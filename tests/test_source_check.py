"""Tests for the datathon source check (`latamqa.source_check`). Wikipedia and Wikidata are replaced by a fake fetch,
so no network access is needed."""

import json
import types

import pytest

from latamqa import source_check as sc


def page(title, qid, body, categories=()):
    """A minimal rendered Wikipedia page, shaped like the real ones."""
    config = json.dumps({"wgTitle": title, "wgWikibaseItemId": qid, "wgCategories": list(categories)})
    return (
        f"<html><head><script>RLCONF={config};</script></head><body><div id='mw-navigation'>Menú principal</div>"
        f'<div id="mw-content-text" class="mw-body-content">{body}</div>'
        '<div class="printfooter">Obtenido de «https://es.wikipedia.org/w/index.php?oldid=1»</div>'
        '<div id="catlinks">Categorías: X</div></body></html>'
    )


AYALA = page(
    "Plan de Ayala",
    "Q1781472",
    '<table data-mw=\'{"wt":"a <br /> b"}\'><tr><td>Infobox</td></tr></table>'
    "<p>El plan de Ayala fue un manifiesto de Emiliano Zapata que reclamaba la tierra para los campesinos y concluía "
    'con el lema «Reforma, Libertad, Justicia y Ley».<sup class="reference">[1]</sup></p>'
    '<div class="mw-heading"><h2 id="Contexto">Contexto<span class="mw-editsection"><span>[</span>editar'
    "<span>]</span></span></h2></div><p>Zapata acusó a Madero de traicionar la causa.</p>"
    '<div class="mw-heading"><h2 id="Referencias">Referencias</h2></div><p>Tierra y libertad, un libro.</p>',
)
COLORADO = page("Río Colorado (Argentina)", "Q270627", "<p>El río Colorado recorre unos 1000 km en Argentina.</p>")
CARABOBO = page(
    "Batalla de Carabobo",
    "Q8244119",
    "<p>Batalla de Carabobo puede referirse a: Batalla de Carabobo (1814), Batalla de Carabobo (1821).</p>",
    categories=["Wikipedia:Desambiguación"],
)
ENTITIES = {
    "Q11291": {"id": "Q11291", "labels": {"ca": {"value": "Cistella"}}, "sitelinks": {}},
    "Q270627": {
        "id": "Q270627",
        "labels": {"es": {"value": "río Colorado"}},
        "sitelinks": {"eswiki": {"title": "Río Colorado (Argentina)"}, "commonswiki": {"title": "x"}},
    },
}


class FakeWeb:
    def __init__(self):
        self.calls = []
        self.fail = set()

    def __call__(self, url):
        self.calls.append(url)
        if url in self.fail:
            return 503, ""
        pages = {
            "https://es.wikipedia.org/wiki/Plan_de_Ayala": AYALA,
            "https://es.wikipedia.org/wiki/R%C3%ADo_Colorado_(Argentina)": COLORADO,
            "https://es.wikipedia.org/wiki/Batalla_de_Carabobo": CARABOBO,
        }
        if url in pages:
            return 200, pages[url]
        if "Special:EntityData/" in url:
            qid = url.rsplit("/", 1)[1].removesuffix(".json")
            if qid in ENTITIES:
                return 200, json.dumps({"entities": {qid: ENTITIES[qid]}})
        return 404, "not found"


@pytest.fixture
def web():
    return FakeWeb()


@pytest.fixture
def wiki(tmp_path, web):
    return sc.Wiki(tmp_path / "cache", pause=0, fetch=web)


def question(qid="Q-1", sources="", answer="Reforma, Libertad, Justicia y Ley", d1="Tierra y libertad", **kw):
    q = dict(id=qid, team="T", country="México", question_text="¿Cuál era el lema del Plan de Ayala?", answer=answer)
    q.update(distractor1=d1, distractor2="Viva la revolución", distractor3="Paz y trabajo", wikidata_qids=sources)
    q.update(question_en="Motto?", answer_en=answer, distractor1_en=d1, distractor2_en="x", distractor3_en="y")
    q.update(kw)
    return q


# ---------------------------------------------------------------------------------------------------------- parsing


def test_parse_sources():
    field = "Q11291, https://es.wikipedia.org/wiki/R%C3%ADo_Colorado_(Argentina) "
    field += "https://pt.m.wikipedia.org/wiki/Q123_(x)"
    assert sc.parse_sources(field) == (["Q11291"], [("es", "Río Colorado (Argentina)"), ("pt", "Q123 (x)")])
    assert sc.parse_sources(None) == ([], [])
    assert sc.parse_sources("Q1 Q1") == (["Q1"], [])


@pytest.mark.parametrize(
    ("phrase", "text", "expected"),
    [
        ("Frijol negro", "La feijoada lleva frijoles negros.", "phrase"),
        ("Río Colorado", "el rio colorado nace en los Andes", "phrase"),
        ("Tierra y libertad para los campesinos", "tierra ... libertad ... los campesinos", "words"),
        ("Lentejas", "frijoles negros", None),
        ("", "anything", None),
    ],
)
def test_match(phrase, text, expected):
    assert sc.match(phrase, sc.normalize(text)) == expected


def test_article_text_keeps_content_only():
    text = sc.article_text(AYALA)
    assert text.startswith("Infobox")
    assert "Reforma, Libertad, Justicia y Ley" in text and "Contexto" in text and "traicionar" in text
    for gone in ("Menú principal", "[1]", "editar", "Referencias", "un libro", "Obtenido de", "Categorías", '{"wt"'):
        assert gone not in text


def test_is_disambiguation():
    assert sc.is_disambiguation(CARABOBO)
    assert not sc.is_disambiguation(AYALA)  # a hatnote linking to a disambiguation page is not one


# ---------------------------------------------------------------------------------------------------------- fetching


def test_wiki_caches_results_but_not_failures(wiki, web):
    first = wiki.article("es", "Plan de Ayala")
    assert first["qid"] == "Q1781472" and first["title"] == "Plan de Ayala" and not first["disambiguation"]
    assert wiki.article("es", "Plan de Ayala") == first and len(web.calls) == 1
    assert wiki.article("es", "No existe") == {"missing": True}
    url = "https://www.wikidata.org/wiki/Special:EntityData/Q11291.json"
    web.fail.add(url)
    assert wiki.entity("Q11291") is None  # a failed fetch is retried next time
    web.fail.clear()
    assert wiki.entity("Q11291")["label"] == "Cistella"
    assert wiki.entity("Q270627")["sitelinks"] == {"eswiki": "Río Colorado (Argentina)"}


# ---------------------------------------------------------------------------------------------------------- checking


def test_key_in_source(wiki):
    r = sc.check_question(question(sources="https://es.wikipedia.org/wiki/Plan_de_Ayala"), wiki)
    assert r["verdict"] == "key in source" and not r["flag"] and r["articles"] == ["es:Plan de Ayala (Q1781472)"]


def test_distractor_in_source_but_not_the_key(wiki):
    q = question(answer="Tierra y libertad para los campesinos", d1="Reforma, Libertad, Justicia y Ley")
    r = sc.check_question(dict(q, wikidata_qids="https://es.wikipedia.org/wiki/Plan_de_Ayala"), wiki)
    assert r["verdict"] == "distractor in source, key not" and r["flag"]
    assert r["distractors_found"] == ["Reforma, Libertad, Justicia y Ley"]


def test_key_not_in_source(wiki):
    q = question(answer="Viva Zapata", d1="Abajo Madero", sources="https://es.wikipedia.org/wiki/Plan_de_Ayala")
    r = sc.check_question(q, wiki)
    assert r["verdict"] == "key not in source" and r["flag"]


def test_wikidata_id_that_is_not_the_cited_article(wiki):
    q = question(answer="Río Colorado", sources="Q11291, https://es.wikipedia.org/wiki/R%C3%ADo_Colorado_(Argentina)")
    r = sc.check_question(q, wiki)
    assert r["verdict"] == "key in source" and r["flag"]
    assert r["issues"] == ["Q11291 («Cistella») is not the item of the cited article «Río Colorado (Argentina)» (Q270627)"]


def test_bare_wikidata_id_is_read_through_its_article(wiki):
    r = sc.check_question(question(answer="Río Colorado", sources="Q270627"), wiki)
    assert r["verdict"] == "key in source" and not r["flag"]
    assert r["articles"] == ["es:Río Colorado (Argentina) (Q270627)"]


def test_unusable_sources(wiki):
    r = sc.check_question(question(sources="Q999"), wiki)
    assert r["verdict"] == "source unavailable" and r["issues"] == ["Wikidata Q999 does not exist"]
    r = sc.check_question(question(sources="https://es.wikipedia.org/wiki/Batalla_de_Carabobo"), wiki)
    assert r["verdict"] == "source unavailable" and "disambiguation page" in r["issues"][0]
    r = sc.check_question(question(sources=""), wiki)
    assert r["verdict"] == "no source" and not r["flag"]


def test_excerpt_keeps_the_intro_and_relevant_paragraphs():
    intro = "Intro paragraph about the plan, long enough to count as a paragraph."
    filler = [f"Unrelated paragraph number {i} about something else entirely." for i in range(50)]
    hit = "This paragraph gives the motto: Reforma, Libertad, Justicia y Ley, as written by Zapata."
    article = {"title": "X", "text": "\n".join([intro, *filler, hit])}
    ex = sc.excerpt([article], question(), limit=400)
    assert ex.startswith(intro) and hit in ex and len(ex) <= 400


# ---------------------------------------------------------------------------------------------------------- reader


def fake_litellm(reply):
    calls = []

    def completion(**kw):
        calls.append(kw)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=reply))])

    return types.SimpleNamespace(completion=completion, calls=calls)


READER = dict(
    key="qwen3.5-397b",
    hub_id="Qwen/Qwen3.5-397B-A17B",
    provider="deepinfra",
    extra_body={"reasoning_effort": "none"},
    api_base="https://router.huggingface.co/deepinfra/v1/openai/chat/completions",
)


def test_ask_reader_and_verdicts(wiki):
    q = question(sources="https://es.wikipedia.org/wiki/Plan_de_Ayala")
    articles = [wiki.article("es", "Plan de Ayala")]
    order = [2, 0, 1, 3]  # the key is shown as B
    llm = fake_litellm("B")
    letter, _ = sc.ask_reader(llm, READER, q, articles, order, "org", "tok")
    kw = llm.calls[0]
    assert letter == "B" and kw["extra_headers"] == {"X-HF-Bill-To": "org"} and kw["extra_body"] == READER["extra_body"]
    assert "B) Reforma, Libertad, Justicia y Ley" in kw["messages"][1]["content"]
    assert "E) The source does not say" in kw["messages"][1]["content"]
    assert sc.reader_verdict("B", order, q) == ("reader agrees", False)
    assert sc.reader_verdict("A", order, q) == ("reader picks another option: «Viva la revolución»", True)
    assert sc.reader_verdict("E", order, q) == ("reader: source does not say", True)
    assert sc.reader_verdict(None, order, q) == ("reader: no answer", False)


def test_check_all_is_incremental_and_follows_edits(tmp_path, wiki, web):
    qs = [
        question("a", sources="https://es.wikipedia.org/wiki/Plan_de_Ayala"),
        question("b", sources=""),
    ]
    checks, new = sc.check_all(qs, tmp_path, wiki)
    assert new == 2 and set(checks) == {"a", "b"} and "_articles" not in checks["a"]
    assert sc.check_all(qs, tmp_path, wiki)[1] == 0  # nothing new
    qs[1]["wikidata_qids"] = "https://es.wikipedia.org/wiki/Batalla_de_Carabobo"  # an edit invalidates the check
    assert set(sc.load_checks(tmp_path, qs)) == {"a"}
    checks, new = sc.check_all(qs, tmp_path, wiki)
    assert new == 1 and checks["b"]["verdict"] == "source unavailable"
    llm = fake_litellm("E")
    checks, new = sc.check_all(qs, tmp_path, wiki, reader=READER, litellm=llm, bill_to="org", token="tok")
    assert new == 2 and len(llm.calls) == 1  # only the readable source is sent to the reader
    assert checks["a"]["reader_verdict"] == "reader: source does not say" and checks["a"]["flag"]
    assert sc.check_all(qs, tmp_path, wiki, reader=READER, litellm=llm)[1] == 0
