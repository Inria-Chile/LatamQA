"""Check a datathon question's answer key against the source its authors cited (Wikipedia and Wikidata).

The datathon app stores each question's source in ``wikidata_qids``: Wikidata ids (``Q123``) and/or Wikipedia article
links, separated by commas (spec §3). For each accepted question this module

- fetches the cited Wikipedia articles and Wikidata items; a bare Wikidata id is read through its Wikipedia article
  in the question's language (Spanish, or Portuguese for Brazil), else English;
- checks that the ids and links agree: a cited id that is not the Wikidata item of the cited article is reported,
  e.g. "Q11291 («Cistella») is not the item of the cited article «Río Colorado (Argentina)»";
- looks for the answer and the distractors in the article text (accents, case and plurals ignored);
- optionally asks a reader model which option the article supports (``reader``), which catches a key the article
  mentions but contradicts.

Verdicts, from the article text: ``key in source``; ``key words in source`` (all the key's content words, not the
exact phrase); ``distractor in source, key not``; ``key not in source``; ``no source`` (only a provenance note, which
the spec allows); ``source unavailable``. The reader adds ``reader agrees``, ``reader picks another option: «…»`` or
``reader: source does not say``. A check is flagged for the committee when the article backs a distractor or not the
key, when ids and links disagree or cannot be fetched, or when the reader picks another option. Flags never change a
score (spec §3.1).

Wikimedia's Action API (``w/api.php``) throttles shared addresses hard, so the article pages (``/wiki/<title>``) and
Wikidata entity files (``Special:EntityData/<id>.json``) are read instead, with a descriptive User-Agent, a pause between
requests, Retry-After on HTTP 429 and a disk cache: every page is fetched once per event.
"""

import datetime as dt
import hashlib
import json
import random
import re
import time
import unicodedata
import zlib
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote, unquote

from structlog import get_logger

logger = get_logger(__name__)

USER_AGENT = "LatamQA-datathon-verifier/0.1 (https://github.com/Inria-Chile/LatamQA)"
CHECKS_FILE = "source_checks.json"
CHECK_VERSION = 3  # bump when the checks change, so saved results are redone
QID = re.compile(r"\bQ[1-9]\d*\b")
WIKI_LINK = re.compile(r"https?://([a-z][a-z-]*)\.(?:m\.)?wikipedia\.org/wiki/([^\s,;|#?]+)")
PROJECT_WIKIS = {"commonswiki", "specieswiki", "metawiki", "mediawikiwiki", "wikidatawiki", "sourceswiki", "incubatorwiki"}
FLAG_VERDICTS = {"distractor in source, key not", "key not in source", "source unavailable"}
END_SECTIONS = {  # article text stops at the first of these headings (references and links, not content)
    "referencias", "notas", "bibliografia", "enlaces externos", "vease tambien", "referencias bibliograficas",
    "referencias e notas", "ligacoes externas", "ver tambem", "notas e referencias", "references", "notes",
    "external links", "see also", "bibliography", "further reading",
}  # fmt: skip
STOPWORDS = set(
    "el la los las lo un una unos unas de del al y e o u en con por para que se su sus es fue the of and a an in on to "
    "for is was by as at from with os as um uma do da dos das no na nos nas em com pelo pela ao aos".split()
)
EXCERPT_CHARS = 6000
READER_SYSTEM = "You check quiz questions against a source text. Reply with a single letter."
READER_TEMPLATE = """Source: Wikipedia, «{title}»

{excerpt}

Question: {question}

A) {a}
B) {b}
C) {c}
D) {d}
E) The source does not say

Using only the source above, which option is correct? Reply with one letter: A, B, C, D or E."""


# ---------------------------------------------------------------------------------------------------------- parsing


def parse_sources(field: str | None) -> tuple[list[str], list[tuple[str, str]]]:
    """Wikidata ids and Wikipedia ``(language, title)`` links cited in a question's ``wikidata_qids`` field.

    >>> parse_sources("Q11291, https://es.wikipedia.org/wiki/R%C3%ADo_Colorado_(Argentina)")
    (['Q11291'], [('es', 'Río Colorado (Argentina)')])
    """
    text = field or ""
    links = [(lang, unquote(title).replace("_", " ")) for lang, title in WIKI_LINK.findall(text)]
    qids = QID.findall(WIKI_LINK.sub(" ", text))  # ids inside a link's title are not citations
    return list(dict.fromkeys(qids)), list(dict.fromkeys(links))


def _fold(text: str) -> str:
    text = unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^0-9a-z]+", " ", text).strip()


def _stem(word: str) -> str:
    if len(word) > 4 and word.endswith("es"):
        return word[:-2]
    if len(word) > 3 and word.endswith("s"):
        return word[:-1]
    return word


def normalize(text: str) -> str:
    """Accents, case, punctuation and plural endings removed: ``"Frijoles negros"`` -> ``"frijol negro"``."""
    return " ".join(_stem(w) for w in _fold(text).split())


def match(phrase: str, text_norm: str) -> str | None:
    """``"phrase"`` if the normalized phrase occurs in the normalized text, ``"words"`` if all its content words do,
    else None.

    >>> t = normalize("La feijoada se prepara con frijoles negros y carne de cerdo.")
    >>> match("Frijol negro", t), match("Carne negra", t), match("Lentejas", t)
    ('phrase', None, None)
    """
    p = normalize(phrase)
    if not p:
        return None
    padded = f" {text_norm} "
    if f" {p} " in padded:
        return "phrase"
    words = [w for w in p.split() if len(w) > 2 and w not in STOPWORDS]
    if words and all(f" {w} " in padded for w in words):
        return "words"
    return None


class _ArticleText(HTMLParser):
    """Collects the readable text of a rendered article: inside ``#mw-content-text``, without styles, scripts,
    footnote marks and edit links, up to the first references-like section, the category links or the footer."""

    BLOCK = frozenset({"p", "div", "li", "tr", "br", "h1", "h2", "h3", "h4", "h5", "h6", "table", "dd", "dt", "caption"})

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.inside = self.done = False
        self.skip: list[str] = []  # tags of the skipped elements we are in
        self.heading: list[str] | None = None
        self.heading_at = 0

    def handle_starttag(self, tag, attrs):
        if self.done:
            return
        a = dict(attrs)
        cls, ident = a.get("class") or "", a.get("id") or ""
        if ident == "mw-content-text":
            self.inside = True
            return
        if not self.inside:
            return
        if ident == "catlinks" or "printfooter" in cls:
            self.done = True
            return
        if self.skip:
            if tag == self.skip[-1]:  # a nested element of the same kind: wait for its end tag too
                self.skip.append(tag)
            return
        if tag in ("style", "script") or (tag == "sup" and "reference" in cls) or "mw-editsection" in cls:
            self.skip.append(tag)
            return
        if tag in self.BLOCK:
            self.parts.append("\n")
        if tag == "h2":
            self.heading, self.heading_at = [], len(self.parts)

    def handle_endtag(self, tag):
        if self.done or not self.inside:
            return
        if self.skip:
            if tag == self.skip[-1]:
                self.skip.pop()
            return
        if tag in self.BLOCK:
            self.parts.append("\n")
        if tag == "h2" and self.heading is not None:
            if _fold("".join(self.heading)) in END_SECTIONS:
                del self.parts[self.heading_at :]
                self.done = True
            self.heading = None

    def handle_data(self, data):
        if self.inside and not self.done and not self.skip:
            self.parts.append(data)
            if self.heading is not None:
                self.heading.append(data)


def article_text(page: str) -> str:
    """Readable text of a rendered Wikipedia article page, without navigation, references and footnote marks."""
    parser = _ArticleText()
    parser.feed(page)
    lines = (re.sub(r"[ \t\u00a0]+", " ", line).strip() for line in "".join(parser.parts).split("\n"))
    return "\n".join(line for line in lines if line)


def is_disambiguation(page: str) -> bool:
    """A disambiguation page (a list of articles with similar titles), which cannot back a specific answer. Read from
    the page's categories: «Wikipedia:Desambiguación» (es), «Desambiguação» (pt), «Disambiguation pages» (en)."""
    m = re.search(r'"wgCategories":\s*(\[.*?\])', page)
    try:
        categories = json.loads(m.group(1)) if m else []
    except ValueError:
        categories = []
    return any(w in _fold(c) for c in categories for w in ("desambiguacion", "desambiguacao", "disambiguation pages"))


# ---------------------------------------------------------------------------------------------------------- fetching


class Wiki:
    """Reads Wikipedia articles and Wikidata items politely, with a disk cache of the extracted results.

    ``fetch(url) -> (status, body)`` replaces the HTTP layer (tests). Pages that do not exist are cached too.
    """

    def __init__(self, cache_dir: Path, pause: float = 0.5, fetch: Callable[[str], tuple[int, str]] | None = None):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.pause = pause
        self.fetch = fetch or self._http
        self._last = 0.0

    def _http(self, url: str) -> tuple[int, str]:
        import httpx

        status = 0
        for attempt in range(5):
            wait = self._last + self.pause - time.time()
            if wait > 0:
                time.sleep(wait)
            try:
                r = httpx.get(url, headers={"User-Agent": USER_AGENT}, timeout=30, follow_redirects=True)
                status = r.status_code
            except httpx.HTTPError:
                status, r = 599, None
            self._last = time.time()
            if status == 429 or status >= 500:
                try:  # Retry-After is seconds or an HTTP date
                    retry_after = float(r.headers.get("retry-after") or 2**attempt) if r is not None else 2**attempt
                except ValueError:
                    retry_after = 2**attempt
                time.sleep(min(retry_after, 60) * (1 + random.random() * 0.2))
                continue
            return status, r.text if r is not None else ""
        return status, ""

    def _cached(self, key: str, build: Callable[[], dict | None]) -> dict | None:
        path = self.cache_dir / f"{hashlib.sha1(key.encode()).hexdigest()}.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))["value"]
        value = build()
        if value is not None and value.get("_transient"):
            return None  # a failed fetch: try again next time
        path.write_text(json.dumps({"key": key, "value": value}, ensure_ascii=False), encoding="utf-8")
        return value

    def article(self, lang: str, title: str) -> dict | None:
        """``{lang, title, qid, text}`` of a Wikipedia article (redirects followed), ``{"missing": True}`` if it
        does not exist, None if it could not be fetched."""

        def build():
            status, page = self.fetch(f"https://{lang}.wikipedia.org/wiki/{quote(title.replace(' ', '_'), safe="()/'")}")
            if status == 404:
                return {"missing": True}
            if status != 200:
                return {"_transient": True}
            qid = re.search(r'"wgWikibaseItemId":\s*"(Q\d+)"', page)
            name = re.search(r'"wgTitle":\s*("(?:[^"\\]|\\.)*")', page)
            text = article_text(page)
            return dict(
                lang=lang,
                title=json.loads(name.group(1)) if name else title,
                qid=qid.group(1) if qid else None,
                text=text,
                disambiguation=is_disambiguation(page),
            )

        return self._cached(f"article:v3:{lang}:{title}", build)

    def entity(self, qid: str) -> dict | None:
        """``{qid, label, sitelinks}`` of a Wikidata item (label in es, pt or en), ``{"missing": True}`` if it does not
        exist, None if it could not be fetched."""

        def build():
            status, body = self.fetch(f"https://www.wikidata.org/wiki/Special:EntityData/{qid}.json")
            if status == 404:
                return {"missing": True}
            if status != 200:
                return {"_transient": True}
            entity = next(iter(json.loads(body)["entities"].values()))
            labels = entity.get("labels", {})
            label = next((labels[lg]["value"] for lg in ("es", "pt", "en") if lg in labels), None)
            label = label or next((v["value"] for v in labels.values()), None)
            sitelinks = {
                k: v["title"] for k, v in entity.get("sitelinks", {}).items() if k.endswith("wiki") and k not in PROJECT_WIKIS
            }
            return dict(qid=entity.get("id", qid), label=label, sitelinks=sitelinks)

        return self._cached(f"entity:{qid}", build)


# ---------------------------------------------------------------------------------------------------------- checking


def question_language(question: dict) -> str:
    return "pt" if _fold(question.get("country") or "") in ("brasil", "brazil") else "es"


def content_hash(question: dict) -> str:
    """Changes when the question's text, options or cited sources change."""
    fields = ["question_text", "answer", "distractor1", "distractor2", "distractor3", "question_en", "answer_en"]
    fields += ["distractor1_en", "distractor2_en", "distractor3_en", "wikidata_qids"]
    payload = [CHECK_VERSION, *(question.get(f) for f in fields)]
    return f"{zlib.crc32(json.dumps(payload, ensure_ascii=False).encode()):08x}"


def _options(question: dict, lang: str) -> list[str]:
    suffix = "_en" if lang == "en" else ""
    return [question.get(f"answer{suffix}") or ""] + [question.get(f"distractor{i}{suffix}") or "" for i in (1, 2, 3)]


def check_question(question: dict, wiki: Wiki) -> dict:
    """Text check of one question against its cited sources (no model involved). A source that could not be fetched
    (Wikimedia throttling or down) sets ``retry``: the check is saved but not used, and the next `check_all` redoes it."""
    qids, links = parse_sources(question.get("wikidata_qids"))
    result: dict[str, Any] = dict(sources=question.get("wikidata_qids") or "", articles=[], issues=[])
    if not qids and not links:
        return dict(result, verdict="no source", flag=False)
    articles = []
    for lang, title in links:
        a = wiki.article(lang, title)
        if a is None:
            result["issues"].append(f"could not fetch {lang}:{title}")
            result["retry"] = True
        elif a.get("missing"):
            result["issues"].append(f"cited article {lang}:{title} does not exist")
        elif a.get("disambiguation"):
            result["issues"].append(f"cited article {lang}:{title} is a disambiguation page; cite the specific article")
        else:
            articles.append(a)
    cited = list(articles)  # the linked articles; a bare id's own article, added below, is not one of them
    linked = {a["qid"] for a in cited if a.get("qid")}
    for qid in qids:
        e = wiki.entity(qid)
        if e is None:
            result["issues"].append(f"could not fetch Wikidata {qid}")
            result["retry"] = True
            continue
        if e.get("missing"):
            result["issues"].append(f"Wikidata {qid} does not exist")
            continue
        if cited and qid not in linked:
            titles = ", ".join(f"«{a['title']}» ({a.get('qid')})" for a in cited)
            result["issues"].append(f"{qid} («{e['label']}») is not the item of the cited article {titles}")
        elif not cited:
            for lang in (question_language(question), "en"):
                title = e["sitelinks"].get(f"{lang}wiki")
                if not title:
                    continue
                a = wiki.article(lang, title)
                if a is None:
                    result["issues"].append(f"could not fetch {lang}:{title}")
                    result["retry"] = True
                    break
                if not a.get("missing") and not a.get("disambiguation"):
                    articles.append(a)
                    break
            else:
                result["issues"].append(
                    f"{qid} («{e['label']}») has no Wikipedia article in {question_language(question)} or en"
                )
    result["articles"] = [f"{a['lang']}:{a['title']} ({a.get('qid')})" for a in articles]
    if not articles:
        return dict(result, verdict="source unavailable", flag=True)
    key_match, distractors = None, set()
    for a in articles:
        text = normalize(a["text"])
        options = _options(question, a["lang"] if a["lang"] == "en" else "regional")
        hit = match(options[0], text)
        if hit == "phrase" or (hit and not key_match):
            key_match = hit
        distractors |= {i for i in (1, 2, 3) if match(options[i], text) == "phrase"}
    regional = _options(question, "regional")
    result["key_match"] = key_match
    result["distractors_found"] = [regional[i] for i in sorted(distractors)]
    if key_match == "phrase":
        verdict = "key in source"
    elif distractors:
        verdict = "distractor in source, key not"
        result["issues"].append(
            f"the source mentions {', '.join(f'«{d}»' for d in result['distractors_found'])} but not the key"
        )
    elif key_match == "words":
        verdict = "key words in source"
    else:
        verdict = "key not in source"
        result["issues"].append(f"the key «{regional[0]}» does not appear in the source")
    flag = verdict in FLAG_VERDICTS or bool(result["issues"])
    return dict(result, verdict=verdict, flag=flag, _articles=articles)


def excerpt(articles: list[dict], question: dict, limit: int = EXCERPT_CHARS) -> str:
    """The article intro plus the paragraphs that share the most words with the question and its options."""
    words = {w for t in [question.get("question_text"), *_options(question, "regional")] for w in normalize(t or "").split()}
    words -= STOPWORDS
    parts = []
    for a in articles:
        paragraphs = [p for p in a["text"].split("\n") if len(p) > 40]
        if not paragraphs:
            continue
        scored = sorted(
            range(1, len(paragraphs)),
            key=lambda i: -len(words & set(normalize(paragraphs[i]).split())),
        )
        keep, size = {0}, len(paragraphs[0])
        for i in scored:
            if size + len(paragraphs[i]) > limit // len(articles):
                continue
            keep.add(i)
            size += len(paragraphs[i])
        parts.append("\n".join(paragraphs[i] for i in sorted(keep)))
    return "\n\n".join(parts)[:limit]


def ask_reader(litellm, spec: dict, question: dict, articles: list[dict], order: list[int], bill_to: str, token: str):
    """Ask the reader model which option the cited source supports. Returns (letter or None, raw reply)."""
    from latamqa import panel as pn

    options = _options(question, "regional")
    shown = [options[i] for i in order]
    prompt = READER_TEMPLATE.format(
        title=", ".join(a["title"] for a in articles),
        excerpt=excerpt(articles, question),
        question=question.get("question_text"),
        a=shown[0],
        b=shown[1],
        c=shown[2],
        d=shown[3],
    )
    kwargs: dict[str, Any] = dict(
        model=pn.litellm_model(spec),
        messages=[{"role": "system", "content": READER_SYSTEM}, {"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=16,
        api_key=token,
        timeout=pn.TIMEOUT_S,
        num_retries=0,
        extra_headers={"X-HF-Bill-To": bill_to},
    )
    if spec.get("extra_body"):
        kwargs["extra_body"] = spec["extra_body"]
    if spec.get("api_base"):
        kwargs["api_base"] = spec["api_base"]
    for attempt in range(4):
        try:
            reply = (litellm.completion(**kwargs).choices[0].message.content or "").strip()
            break
        except Exception as e:  # retried like the panel's requests; anything else leaves the reader verdict empty
            if pn.status_of(e) not in pn.RETRYABLE or attempt == 3:
                return None, f"error: {type(e).__name__}: {str(e)[:200]}"
            time.sleep(2**attempt)
    m = re.match(r"^[\W_]*([A-E])\b", reply)
    return (m.group(1) if m else None), reply


def reader_verdict(letter: str | None, order: list[int], question: dict) -> tuple[str, bool]:
    if letter is None:
        return "reader: no answer", False
    if letter == "E":
        return "reader: source does not say", True
    picked = order["ABCD".index(letter)]
    if picked == 0:
        return "reader agrees", False
    return f"reader picks another option: «{_options(question, 'regional')[picked]}»", True


# ---------------------------------------------------------------------------------------------------------- batch


def load_checks(out_dir: Path, accepted: list[dict]) -> dict[str, dict]:
    """Saved checks that still match the questions' current content, by question id. Checks whose sources could not
    all be fetched (``retry``) are left out: a temporary failure must not flag a question."""
    path = Path(out_dir) / CHECKS_FILE
    if not path.exists():
        return {}
    saved = json.loads(path.read_text(encoding="utf-8"))
    current = {q["id"]: content_hash(q) for q in accepted}
    return {qid: r for qid, r in saved.items() if current.get(qid) == r.get("hash") and not r.get("retry")}


def check_all(
    accepted: list[dict],
    out_dir: Path,
    wiki: Wiki,
    reader: dict | None = None,
    litellm=None,
    bill_to: str = "",
    token: str = "",
    order_of: Callable[[str], list[int]] | None = None,
) -> tuple[dict[str, dict], int]:
    """Check every accepted question not checked yet (or edited since), save ``source_checks.json`` and return all
    current checks plus the number of new ones. With ``reader`` (a panel model entry), questions whose source could be
    read also get a reader verdict; ``order_of(question_id)`` gives the option order shown to the reader."""
    path = Path(out_dir) / CHECKS_FILE
    saved = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    current = load_checks(out_dir, accepted)
    todo = [q for q in accepted if q["id"] not in current or (reader and current[q["id"]].get("reader") != reader["key"])]
    results: dict[str, dict] = {}
    for i, q in enumerate(todo, 1):
        r = check_question(q, wiki)
        r.update(
            hash=content_hash(q), team=q.get("team"), checked_at=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
        )
        results[q["id"]] = r
        if i % 25 == 0:
            logger.info(f"source check: {i}/{len(todo)} questions")
    if reader:
        for r in results.values():  # also for questions without a readable source, so they are not redone
            r["reader"] = reader["key"]
        readable = [q for q in todo if results[q["id"]].get("_articles")]

        def read(q):
            order = order_of(q["id"]) if order_of else [0, 1, 2, 3]
            letter, reply = ask_reader(litellm, reader, q, results[q["id"]]["_articles"], order, bill_to, token)
            verdict, flag = reader_verdict(letter, order, q)
            r = results[q["id"]]
            r.update(reader_reply=reply[:200], reader_verdict=verdict)
            if letter is None and reply.startswith("error: "):  # a failed request: let the next verify ask again
                r.pop("reader", None)
            if flag:
                r["flag"] = True
                r["issues"].append(verdict)

        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(read, readable))
    for r in results.values():
        r.pop("_articles", None)
    retry = sum(bool(r.get("retry")) for r in results.values())
    if retry:
        logger.warning(f"source check: {retry} question(s) with a source that could not be fetched; verify retries them")
    saved.update(results)
    path.write_text(json.dumps(saved, indent=1, ensure_ascii=False), encoding="utf-8")
    return load_checks(out_dir, accepted), len(results)
