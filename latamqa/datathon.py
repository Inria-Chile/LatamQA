#!/usr/bin/env python3
"""Score the LLACA datathon teams with the model panel and rank them.

Reads the teams' questions from the datathon app's SQLite database (a local file, or the app's private Hub backup),
asks every panel model each accepted question in both of its languages under the app's four balanced option orders,
and ranks the teams with the rules of the datathon specification (llaca-datathon-app, docs/datathon-platform-spec.md):

  score(q)   = 1 - correct / received        over the (model, language, order) answers received for question q;
                                              errors are retried and never scored; an unparsable or reasoning reply is
                                              received and wrong (§6.2)
  team score = sum of max(0, score(q) - 0.5) over the team's accepted questions (§7)
  ties       = descending score(q) vectors padded to 60, then the lower sum of submission sequence numbers, then the
               lowest single sequence number (§7, D10)

The ranking also shows each team's mean panel accuracy (mean over its scored questions of correct / received) for
reference; it does not affect the order. Accepted = submitted or evaluated, not excluded or withdrawn (§3.1).

Every run is incremental: requests are sent only for answers not yet received (new questions, edited questions, earlier
errors), then the whole ranking is recomputed from the database snapshot and the logs and written to
``rankings/ranking_<UTC time>.md|.csv`` and ``ranking_latest.md|.csv``.

Stages:
  check   free   panel pre-run checks (token, billing org, routes, gated access)
  run     paid   evaluate what is missing, then rank
  rank    free   re-rank from the logs and a fresh database snapshot (e.g. after committee exclusions)
  verify  free   check new or edited answer keys against their cited Wikipedia/Wikidata sources (paid with --reader)

Usage:
  datathon check
  datathon run  --db hf:inria-chile/db-datathon-test --yes
  datathon run  --db path/to/datathon.db --dry_run
  datathon rank --db hf:inria-chile/db-datathon-test
  datathon run  --db hf:inria-chile/db-datathon-test --publish inria-chile/datathon-results --yes

With ``--publish``, run, rank and verify also upload the results file the datathon app imports (`results_payload`) to
a private Hugging Face dataset.
"""

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import shutil
import sqlite3
import sys
import tempfile
import zlib
from pathlib import Path

from structlog import get_logger

from latamqa import panel as pn
from latamqa import source_check as sc
from latamqa.mcq_core import V2_PROMPT_TEMPLATE, build_prompt

logger = get_logger(__name__)

DEFAULT_RESULTS_DIR = Path(__file__).parent.parent / "results" / "datathon"
DB_FILE = "datathon.db"
ACCEPTED_STATES = ("SUBMITTED", "EVALUATED")
LANGUAGES = {  # language -> (question, answer, distractor1, distractor2, distractor3) columns of the questions table
    "regional": ("question_text", "answer", "distractor1", "distractor2", "distractor3"),
    "english": ("question_en", "answer_en", "distractor1_en", "distractor2_en", "distractor3_en"),
}
# read when present: the cited sources (Wikidata ids and Wikipedia links) and the provenance note (spec §3)
OPTIONAL_COLUMNS = ("wikidata_qids", "cultural_provenance")
N_PERMUTATIONS = 4
SCORE_FLOOR = 0.5  # spec §7: a question earns points only above this score
QUOTA = 60  # spec §3.2: questions per team; the tie-break vectors are padded to this length
NO_ANSWER_FLAG = 0.30  # spec §6.2: questions with more unanswered replies than this go to the committee
# A question where at least this share of the panel's answers picks the same wrong option is flagged for review: the
# key may be wrong or ambiguous (a wrong key "fools" the panel and earns points, as the Sep 2026 test run showed).
CONSENSUS_FLAG = 0.5
# `--publish`: the results file the datathon app imports (see `results_payload`) and the age above which the database
# snapshot is reported as stale (the app backs its database up to the Hub on a timer; the spec asks for 5 minutes)
RESULTS_SCHEMA = 1
RESULTS_FILE = "latest.json"
STALE_DATA_MIN = 30


# ---------------------------------------------------------------------------------------------------------- questions


def unique_stamp(directory: Path, pattern: str) -> str:
    """A UTC timestamp such that ``directory / pattern.format(stamp)`` does not exist yet (adds -2, -3... if needed)."""
    base = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stamp, n = base, 1
    while (directory / pattern.format(stamp)).exists():
        n += 1
        stamp = f"{base}-{n}"
    return stamp


def balanced_permutations(question_id: str) -> list[tuple[list[int], str]]:
    """The app's four balanced option orders for a question (``evaluator.get_balanced_permutations``, spec §6.2).

    Returns, for runs 0-3, the order of ``[answer, distractor1, distractor2, distractor3]`` (as indices) and the
    correct letter. One sha256 of the question id fixes the distractor order and a base offset; run r puts the answer
    at position (offset + r) mod 4, so it appears exactly once at each of A-D.

    >>> [letter for _, letter in balanced_permutations("e57b6436-01d1-4486-99a0-db1bca8deeef")]
    ['A', 'B', 'C', 'D']
    """
    base = int(hashlib.sha256(question_id.encode("utf-8")).hexdigest(), 16)
    distractors = [2, 1, 3] if base % 2 == 1 else [1, 2, 3]
    orders = []
    for r in range(N_PERMUTATIONS):
        target = (base % 4 + r) % 4
        rest = iter(distractors)
        orders.append(([0 if i == target else next(rest) for i in range(4)], "ABCD"[target]))
    return orders


def iso_utc(t: dt.datetime) -> str:
    return t.astimezone(dt.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def snapshot_db(source: str, dest_dir: Path) -> tuple[Path, str]:
    """Copy the app database into ``dest_dir/datathon_<UTC time>.db``; return the copy and when its data was written.

    ``source`` is a local SQLite file or ``hf:<org>/<dataset>`` (the app's private backup repo, file ``datathon.db``).
    The copy uses SQLite's backup API, so a live database in WAL mode is read consistently. The data time is the
    backup's commit time for a Hub source (the copy is of that very commit) and the file time for a local one.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"datathon_{unique_stamp(dest_dir, 'datathon_{}.db')}.db"
    with tempfile.TemporaryDirectory() as tmp:
        if source.startswith("hf:"):
            from huggingface_hub import hf_hub_download

            token = pn.hf_token()
            info = hub_api(token).get_paths_info(source[3:], [DB_FILE], expand=True, repo_type="dataset")
            if not info or info[0].last_commit is None:
                raise pn.PanelError(f"no {DB_FILE} in dataset {source[3:]}")
            commit = info[0].last_commit
            path = hf_hub_download(
                source[3:],
                DB_FILE,
                repo_type="dataset",
                revision=commit.oid,
                local_dir=tmp,
                token=token,
                force_download=True,
            )
            written = commit.date
        else:
            path = source
            if not Path(path).exists():
                raise pn.PanelError(f"database not found: {path}")
            wal = Path(f"{path}-wal")
            mtime = max(Path(path).stat().st_mtime, wal.stat().st_mtime if wal.exists() else 0.0)
            written = dt.datetime.fromtimestamp(mtime, dt.timezone.utc)
        src = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        dst = sqlite3.connect(dest)
        try:
            src.backup(dst)
            dst.execute("PRAGMA journal_mode=DELETE")  # a plain file: no -wal/-shm left beside each snapshot
        finally:
            src.close()
            dst.close()
    return dest, iso_utc(written)


def read_db(path: Path) -> tuple[list[dict], list[dict]]:
    """All teams and all questions (every state) from an app database."""
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        teams = [dict(r) for r in con.execute("SELECT id, name, country FROM teams ORDER BY id")]
        columns = {r[1] for r in con.execute("PRAGMA table_info(questions)")}
        wanted = ["id", "team_id", "state", "submission_sequence_number", *sorted({c for v in LANGUAGES.values() for c in v})]
        missing = [c for c in wanted if c not in columns]
        if missing:
            raise pn.PanelError(f"{path} lacks question columns {missing}: is it a datathon app database?")
        optional = [c for c in OPTIONAL_COLUMNS if c in columns]  # sources and provenance, for `datathon verify`
        questions = [dict(r) for r in con.execute(f"SELECT {', '.join(wanted + optional)} FROM questions")]
    finally:
        con.close()
    return teams, questions


def build_cells(teams: list[dict], questions: list[dict]) -> tuple[list[dict], list[dict], list[str]]:
    """Accepted questions and their evaluation cells (one per language and option order), plus problems found.

    A cell id carries a CRC of the question's text in that language, so a question edited by the committee gets new
    cells and its old answers stop counting (spec §6.2: an edit invalidates the affected cells). A language whose
    question, answer or distractors are empty or not distinct is skipped and reported.
    """
    team_by_id = {t["id"]: t for t in teams}
    accepted, cells, problems = [], [], []
    for q in questions:
        if q["state"] not in ACCEPTED_STATES:
            continue
        team = team_by_id.get(q["team_id"])
        if team is None:
            problems.append(f"question {q['id']}: no team (team_id={q['team_id']}); not ranked")
            continue
        q = dict(q, team=team["name"], country=team["country"])
        accepted.append(q)
        for lang, cols in LANGUAGES.items():
            texts = [str(q[c] or "").strip() for c in cols]
            if not all(texts) or len(set(texts[1:])) < 4:
                problems.append(f"question {q['id']} ({team['name']}): {lang} version incomplete or options not distinct")
                continue
            stem, options = texts[0], texts[1:]
            digest = zlib.crc32(json.dumps(texts, ensure_ascii=False).encode())
            for r, (order, correct) in enumerate(balanced_permutations(str(q["id"]))):
                shown = [options[i] for i in order]
                cells.append(
                    dict(
                        qid=f"{q['id']}|{lang}|p{r}|{digest:08x}",
                        question_id=q["id"],
                        group=team["name"],
                        lang=lang,
                        perm=r,
                        order=order,
                        options=shown,
                        correct=correct,
                        prompt=build_prompt(V2_PROMPT_TEMPLATE, stem, shown),
                    )
                )
    return accepted, cells, problems


# ---------------------------------------------------------------------------------------------------------- scoring


def score_questions(accepted: list[dict], cells: list[dict], logs: dict[str, dict[str, dict]]) -> dict[str, dict]:
    """Per-question panel results over the current cells: received, correct, unanswered, pending, score, accuracy,
    and the wrong option the panel picked most (``consensus_option``, its regional text, and ``consensus_share`` of
    the received answers; wrong picks are counted per option across languages and option orders).

    ``logs`` maps each model key to its latest record per cell id (`panel.latest_records`).
    """
    stats = {q["id"]: dict(received=0, correct=0, no_answer=0, pending=0, picks=[0, 0, 0, 0]) for q in accepted}
    for cell in cells:
        s = stats[cell["question_id"]]
        for recs in logs.values():
            letter, rule = pn.score(recs.get(cell["qid"]))
            if rule in ("error", "missing"):
                s["pending"] += 1
                continue
            s["received"] += 1
            s["correct"] += letter == cell["correct"]
            s["no_answer"] += rule in ("none", "leak")
            if letter and "order" in cell:
                s["picks"][cell["order"]["ABCD".index(letter)]] += 1
    texts = {q["id"]: [q[c] for c in LANGUAGES["regional"][1:]] for q in accepted}
    for qid, s in stats.items():
        n = s["received"]
        s["accuracy"] = s["correct"] / n if n else None
        s["score"] = 1 - s["correct"] / n if n else None
        s["no_answer_rate"] = s["no_answer"] / n if n else None
        top = max(range(1, 4), key=lambda i: s["picks"][i])
        s["consensus_option"] = texts[qid][top] if n and s["picks"][top] else None
        s["consensus_share"] = s["picks"][top] / n if n else None
    return stats


def review_reasons(stats: dict[str, dict], checks: dict[str, dict] | None = None, consensus: float = CONSENSUS_FLAG):
    """Why each question deserves a committee look: ``consensus`` (most answers agree on one wrong option),
    ``unanswered`` (more than 30 % of replies without a letter) and ``source`` (the cited source does not back the
    key, see `latamqa.source_check`). Flags never change a score (spec §3.1: presumption of validity)."""
    reasons = {}
    for qid, s in stats.items():
        r = []
        if (s["consensus_share"] or 0) >= consensus:
            r.append("consensus")
        if (s["no_answer_rate"] or 0) > NO_ANSWER_FLAG:
            r.append("unanswered")
        if (checks or {}).get(qid, {}).get("flag"):
            r.append("source")
        reasons[qid] = r
    return reasons


def rank_teams(
    teams: list[dict],
    accepted: list[dict],
    stats: dict[str, dict],
    floor: float = SCORE_FLOOR,
    quota: int = QUOTA,
    reasons: dict[str, list[str]] | None = None,
) -> list[dict]:
    """Team rows sorted by the spec's team score and tie-breaks, with the mean panel accuracy for reference.

    A question with no answer received yet contributes 0 (spec §3.1). Teams without accepted questions rank last
    among equal scores. ``review`` counts the team's point-earning questions that have a review reason
    (`review_reasons`): the ones the committee should check before results are final.
    """
    reasons = reasons if reasons is not None else review_reasons(stats)
    by_team: dict[int, list[dict]] = {t["id"]: [] for t in teams}
    for q in accepted:
        by_team[q["team_id"]].append(q)
    rows = []
    for t in teams:
        qs = by_team[t["id"]]
        scores = [stats[q["id"]]["score"] for q in qs]
        scored = [s for s in scores if s is not None]
        accuracies = [stats[q["id"]]["accuracy"] for q in qs if stats[q["id"]]["accuracy"] is not None]
        seqs = [q["submission_sequence_number"] for q in qs if q["submission_sequence_number"] is not None]
        vector = sorted((s or 0.0 for s in scores), reverse=True)
        rows.append(
            dict(
                team_id=t["id"],
                team=t["name"],
                country=t["country"],
                team_score=sum(max(0.0, s - floor) for s in scored),
                mean_accuracy=sum(accuracies) / len(accuracies) if accuracies else None,
                accepted=len(qs),
                scored=len(scored),
                earning=sum(s > floor for s in scored),
                pending_questions=sum(stats[q["id"]]["pending"] > 0 for q in qs),
                flagged_no_answer=sum("unanswered" in reasons[q["id"]] for q in qs),
                flagged_consensus=sum("consensus" in reasons[q["id"]] for q in qs),
                flagged_source=sum("source" in reasons[q["id"]] for q in qs),
                review=sum(bool(reasons[q["id"]]) and (stats[q["id"]]["score"] or 0) > floor for q in qs),
                _vector=(vector + [0.0] * quota)[: max(quota, len(vector))],
                _seq_sum=sum(seqs) if seqs else float("inf"),
                _seq_min=min(seqs) if seqs else float("inf"),
            )
        )
    rows.sort(key=lambda r: (-r["team_score"], [-v for v in r["_vector"]], r["_seq_sum"], r["_seq_min"], r["team"]))
    for i, r in enumerate(rows, 1):
        r["rank"] = i
    return rows


# ---------------------------------------------------------------------------------------------------------- outputs


def _cell(text: object) -> str:
    """A value safe inside a Markdown table cell."""
    return str(text if text is not None else "—").replace("|", "\\|").replace("\n", " ")


def _pct(x: float | None) -> str:
    return "—" if x is None else f"{100 * x:.1f}%"


def footnote(floor: float = SCORE_FLOOR) -> str:
    return (
        f"Team score = sum over accepted questions of max(0, score - {floor}), with score = 1 - panel accuracy on the "
        "question. Mean panel accuracy is shown for reference only. Scored = questions with at least one panel answer; "
        "earning points = score above the floor; pending = questions still missing answers; review = point-earning "
        "questions flagged for the committee (panel consensus on one wrong option, unanswered replies, or a cited "
        "source that does not back the key)."
    )


def render_ranking(rows: list[dict], stamp: str, status: str, note: str, floor: float = SCORE_FLOOR):
    """The ranking as a `rich` table for the terminal: the podium in bold, teams without points dimmed, pending counts
    in yellow, and the run status in the title."""
    from rich import box
    from rich.table import Table
    from rich.text import Text

    title = Text.assemble(
        ("Datathon ranking", "bold"),
        f" · {stamp} · ",
        (status, "bold green" if status == "complete" else "bold yellow"),
    )
    table = Table(
        title=title,
        caption=Text(f"{note}\n{footnote(floor)}"),  # Text everywhere: team names are not rich markup ("[/x]" crashes)
        caption_justify="left",
        caption_style="dim",
        box=box.ROUNDED,
        header_style="bold",
        expand=False,
    )
    table.add_column("#", justify="right")
    table.add_column("Team")
    table.add_column("Country")
    table.add_column("Team score", justify="right", style="bold cyan")
    table.add_column("Mean panel accuracy", justify="right")
    for name in ("Accepted", "Scored", "Earning points", "Pending", "Review"):
        table.add_column(name, justify="right")
    for r in rows:
        style = "bold" if r["rank"] <= 3 and r["team_score"] > 0 else "dim" if r["team_score"] == 0 else None
        pending = Text(str(r["pending_questions"]), style="yellow" if r["pending_questions"] else "")
        review = Text(str(r.get("review", 0)), style="bold red" if r.get("review") else "")
        table.add_row(
            str(r["rank"]),
            Text(str(r["team"])),
            Text(str(r["country"] or "—")),
            f"{r['team_score']:.4f}",
            _pct(r["mean_accuracy"]),
            str(r["accepted"]),
            str(r["scored"]),
            str(r["earning"]),
            pending,
            review,
            style=style,
        )
    return table


def write_outputs(
    rows: list[dict],
    accepted: list[dict],
    stats: dict[str, dict],
    out_dir: Path,
    stamp: str,
    status: str,
    note: str,
    floor: float = SCORE_FLOOR,
    reasons: dict[str, list[str]] | None = None,
    checks: dict[str, dict] | None = None,
) -> Path:
    """Write the ranking (Markdown + CSV) and the per-question scores for this run, refresh ``ranking_latest``, and
    show the ranking in the terminal. ``reasons`` are the review flags (`review_reasons`) and ``checks`` the source
    checks (`latamqa.source_check`), both by question id."""
    reasons, checks = reasons or {}, checks or {}
    rank_dir = out_dir / "rankings"
    rank_dir.mkdir(parents=True, exist_ok=True)
    columns = ["rank", "team", "country", "team_score", "mean_accuracy", "accepted", "scored", "earning"]
    columns += ["pending_questions", "review", "flagged_consensus", "flagged_no_answer", "flagged_source"]
    with open(rank_dir / f"ranking_{stamp}.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(dict(r, team_score=round(r["team_score"], 4), mean_accuracy=r["mean_accuracy"]))
    lines = [
        f"# Datathon ranking · {stamp} · {status}",
        "",
        note,
        "",
        "| Rank | Team | Country | Team score | Mean panel accuracy | Accepted | Scored | Earning points | Pending | Review |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    lines += [
        f"| {r['rank']} | {_cell(r['team'])} | {_cell(r['country'])} | {r['team_score']:.4f} | {_pct(r['mean_accuracy'])} | "
        f"{r['accepted']} | {r['scored']} | {r['earning']} | {r['pending_questions']} | {r.get('review', 0)} |"
        for r in rows
    ]
    lines += ["", footnote(floor)]
    md = "\n".join(lines) + "\n"
    (rank_dir / f"ranking_{stamp}.md").write_text(md, encoding="utf-8")
    shutil.copyfile(rank_dir / f"ranking_{stamp}.md", out_dir / "ranking_latest.md")
    shutil.copyfile(rank_dir / f"ranking_{stamp}.csv", out_dir / "ranking_latest.csv")
    qcols = ["team", "country", "question_id", "sequence", "state", "score", "accuracy", "contribution", "received"]
    qcols += ["correct", "pending", "no_answer_rate", "consensus_option", "consensus_share", "source_verdict"]
    qcols += ["source_detail", "reader_verdict", "review"]
    with open(rank_dir / f"questions_{stamp}.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=qcols)
        writer.writeheader()
        for q in sorted(accepted, key=lambda q: (q["team"], q["submission_sequence_number"] or 0)):
            s = stats[q["id"]]
            writer.writerow(
                dict(
                    team=q["team"],
                    country=q["country"],
                    question_id=q["id"],
                    sequence=q["submission_sequence_number"],
                    state=q["state"],
                    score=s["score"],
                    accuracy=s["accuracy"],
                    contribution=max(0.0, s["score"] - floor) if s["score"] is not None else 0.0,
                    received=s["received"],
                    correct=s["correct"],
                    pending=s["pending"],
                    no_answer_rate=s["no_answer_rate"],
                    consensus_option=s["consensus_option"],
                    consensus_share=s["consensus_share"],
                    source_verdict=checks.get(q["id"], {}).get("verdict", "not checked"),
                    source_detail="; ".join(checks.get(q["id"], {}).get("issues", [])),
                    reader_verdict=checks.get(q["id"], {}).get("reader_verdict"),
                    review=",".join(reasons.get(q["id"], [])),
                )
            )
    from rich.console import Console

    console = Console()
    console.print(render_ranking(rows, stamp, status, note, floor))
    console.print(f"-> {rank_dir / f'ranking_{stamp}.md'}", highlight=False, soft_wrap=True)
    return rank_dir / f"ranking_{stamp}.md"


# ---------------------------------------------------------------------------------------------------------- publishing


def hub_api(token: str):
    from huggingface_hub import HfApi

    return HfApi(token=token)


def question_status(s: dict) -> str:
    """``scored`` (every panel answer in), ``partial`` (some still missing), ``pending`` (none yet) or ``unscorable``
    (neither language version is complete, so there is nothing to ask; it contributes 0)."""
    if s["received"]:
        return "partial" if s["pending"] else "scored"
    return "pending" if s["pending"] else "unscorable"


def results_payload(
    event: str,
    data_as_of: str | None,
    status: str,
    rows: list[dict],
    accepted: list[dict],
    stats: dict[str, dict],
    reasons: dict[str, list[str]],
    checks: dict[str, dict],
    floor: float = SCORE_FLOOR,
    dry_run: bool = False,
) -> dict:
    """The results file the datathon app imports (schema ``RESULTS_SCHEMA``), in three parts by audience (spec §6.1):

    - ``ranking``, for the public leaderboard: every team's rank, score and question counts;
    - ``questions``, for each team's own view: the score of each of its accepted questions, never per-model answers;
    - ``review``, for the committee only: the questions with review reasons (`review_reasons`) and why.

    ``data_as_of`` is when the database the ranking reflects was last written (`snapshot_db`): the time to show on a
    provisional leaderboard. No model names or counts that reveal the panel are included (spec §6.1). Questions
    accepted after the snapshot are absent: the app shows them as not evaluated yet.
    """
    by_id = {q["id"]: q for q in accepted}
    return dict(
        schema=RESULTS_SCHEMA,
        event=event,
        published_at=iso_utc(dt.datetime.now(dt.timezone.utc)),
        data_as_of=data_as_of,
        status=status.lower(),
        dry_run=dry_run,
        score_floor=floor,
        ranking=[
            dict(
                rank=r["rank"],
                team_id=r["team_id"],
                team=r["team"],
                country=r["country"],
                team_score=round(r["team_score"], 6),
                mean_accuracy=None if r["mean_accuracy"] is None else round(r["mean_accuracy"], 6),
                accepted=r["accepted"],
                scored=r["scored"],
                earning=r["earning"],
                pending=r["pending_questions"],
            )
            for r in rows
        ],
        questions=[
            dict(
                id=q["id"],
                team_id=q["team_id"],
                sequence=q["submission_sequence_number"],
                score=None if stats[q["id"]]["score"] is None else round(stats[q["id"]]["score"], 6),
                contribution=round(max(0.0, (stats[q["id"]]["score"] or 0.0) - floor), 6),
                status=question_status(stats[q["id"]]),
            )
            for q in sorted(accepted, key=lambda q: (q["team_id"], q["submission_sequence_number"] or 0, q["id"]))
        ],
        review=[
            dict(
                id=qid,
                team_id=by_id[qid]["team_id"],
                reasons=reasons[qid],
                earning=(stats[qid]["score"] or 0.0) > floor,
                consensus_option=stats[qid]["consensus_option"],
                consensus_share=stats[qid]["consensus_share"],
                no_answer_rate=stats[qid]["no_answer_rate"],
                source_verdict=checks.get(qid, {}).get("verdict", "not checked"),
                source_issues=checks.get(qid, {}).get("issues", []),
                reader_verdict=checks.get(qid, {}).get("reader_verdict"),
            )
            for qid in sorted(reasons, key=lambda qid: (by_id[qid]["team_id"], str(qid)))
            if reasons[qid]
        ],
    )


def results_repo(value: str) -> str:
    """``<org>/<dataset>`` from a ``--publish`` value (an ``hf:`` prefix, as in ``--db``, is accepted)."""
    repo = value.removeprefix("hf:")
    if repo.count("/") != 1 or not all(repo.split("/")):
        raise pn.PanelError(f"--publish {value}: expected <org>/<dataset>")
    return repo


def ensure_results_repo(repo: str, token: str) -> str:
    """Make sure ``repo`` is a private dataset this token can reach, creating it (private) if it does not exist; return
    what was found. A public repo is refused: the file holds every team's question scores and the committee's flags."""
    from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError

    api = hub_api(token)
    try:
        info = api.repo_info(repo, repo_type="dataset")
    except RepositoryNotFoundError:
        try:
            api.create_repo(repo, repo_type="dataset", private=True)
        except HfHubHTTPError as e:
            raise pn.PanelError(f"cannot create the results dataset {repo}: {e}") from e
        return "created, private"
    except HfHubHTTPError as e:
        raise pn.PanelError(f"cannot reach the results dataset {repo}: {e}") from e
    if not info.private:
        raise pn.PanelError(
            f"the results dataset {repo} is public; it would expose every team's question scores and the committee's "
            "review flags. Make it private or publish elsewhere."
        )
    return "private"


def publish_results(repo: str, payload: dict, stamp: str, token: str) -> str:
    """Upload ``<event>/latest.json`` and ``<event>/history/results_<stamp>.json`` in one commit; return its URL.

    The app polls ``latest.json``; one commit per publication means it never reads a half-updated file.
    """
    from huggingface_hub import CommitOperationAdd

    data = json.dumps(payload, ensure_ascii=False, indent=1).encode("utf-8")
    event = payload["event"]
    commit = hub_api(token).create_commit(
        repo,
        [
            CommitOperationAdd(f"{event}/{RESULTS_FILE}", data),
            CommitOperationAdd(f"{event}/history/results_{stamp}.json", data),
        ],
        commit_message=f"datathon {event}: {payload['status']} results, data as of {payload['data_as_of']}",
        repo_type="dataset",
    )
    return commit.commit_url


# ---------------------------------------------------------------------------------------------------------- stages


def event_dir(args) -> Path:
    return Path(args.results_dir) / args.event


def load_state(args, specs: list[dict]) -> tuple[list[dict], list[dict], list[dict], Path]:
    """Snapshot the database and build the current cells; write ``items.json`` for the panel workers."""
    out_dir = event_dir(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    snapshot, args.data_as_of = snapshot_db(args.db, out_dir / "snapshots")
    age = dt.datetime.now(dt.timezone.utc) - dt.datetime.fromisoformat(args.data_as_of)
    if age > dt.timedelta(minutes=STALE_DATA_MIN):
        logger.warning(f"the database was last written {age.total_seconds() / 60:.0f} min ago ({args.data_as_of})")
    teams, questions = read_db(snapshot)
    accepted, cells, problems = build_cells(teams, questions)
    for p in problems:
        logger.warning(p)
    (out_dir / "items.json").write_text(json.dumps(cells, ensure_ascii=False))
    print(
        f"{snapshot.name}: {len(teams)} teams, {len(accepted)} accepted questions, {len(cells)} cells "
        f"({len(LANGUAGES)} languages x {N_PERMUTATIONS} orders) per model, {len(specs)} models"
    )
    return teams, accepted, cells, out_dir


def rank_and_write(args, specs: list[dict], teams: list[dict], accepted: list[dict], cells: list[dict], out_dir: Path):
    logs = {s["key"]: pn.latest_records(out_dir / f"{s['key']}.jsonl") for s in specs}
    stats = score_questions(accepted, cells, logs)
    checks = sc.load_checks(out_dir, accepted)
    reasons = review_reasons(stats, checks, args.consensus_flag)
    rows = rank_teams(teams, accepted, stats, args.score_floor, reasons=reasons)
    aborted = [s["key"] for s in specs if (out_dir / f"{s['key']}.ABORT").exists()]
    pending = sum(r["pending_questions"] for r in rows)
    complete = not aborted and not pending
    status = "complete" if complete else "PROVISIONAL"
    note = f"Panel {args.panel_name} ({len(specs)} models), {len(accepted)} accepted questions, {len(teams)} teams."
    if pending:
        note += f" {pending} question(s) still missing panel answers."
    if aborted:
        note += f" Stopped models: {', '.join(aborted)} (see their .ABORT files)."
    refused = sum(pn.is_billing_refusal(recs[c["qid"]]) for recs in logs.values() for c in cells if c["qid"] in recs)
    if refused:  # with --no_billing_check nothing else says why these answers are missing
        note += f" {refused} answer(s) refused for billing (HTTP 402, spending limit): raise the limit or lower --rate."
    unanswered = []
    for key, recs in logs.items():  # a high rate over the whole field can mean a broken reasoning switch
        current = [recs[c["qid"]] for c in cells if recs.get(c["qid"], {}).get("status") == 200]
        if current:
            unanswered.append(f"{key} {sum(pn.score(r)[1] in ('none', 'leak') for r in current) / len(current):.1%}")
    if unanswered:
        note += f" Replies without an answer letter: {', '.join(unanswered)}."
    review = sum(r["review"] for r in rows)
    if review:
        kinds = {k: sum(k in reasons[q["id"]] and (stats[q["id"]]["score"] or 0) > args.score_floor for q in accepted)
                 for k in ("consensus", "unanswered", "source")}  # fmt: skip
        detail = ", ".join(f"{v} {k}" for k, v in kinds.items() if v)
        note += f" {review} point-earning question(s) to review before results are final ({detail}): see questions CSV."
    note += (
        f" Source check: {len(checks)} of {len(accepted)} questions checked (`datathon verify`)."
        if checks
        else " Source check not run (`datathon verify`)."
    )
    if args.dry_run:
        note += " DRY RUN: fake answers, not a real ranking."
    stamp = unique_stamp(out_dir / "rankings", "ranking_{}.md")
    path = write_outputs(rows, accepted, stats, out_dir, stamp, status, note, args.score_floor, reasons, checks)
    if args.publish:
        payload = results_payload(
            args.event, args.data_as_of, status, rows, accepted, stats, reasons, checks, args.score_floor, args.dry_run
        )
        local = out_dir / "rankings" / f"results_{stamp}.json"
        local.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        if args.dry_run:
            print(f"DRY RUN: not published to {args.publish}; the results file is {local}")
        else:
            try:
                url = publish_results(args.publish, payload, stamp, pn.hf_token())
            except Exception as e:  # the ranking is saved: say how to publish it again rather than lose the run
                raise pn.PanelError(
                    f"publishing to {args.publish} failed ({e}); the results file is {local}. "
                    f"Re-publish with `datathon rank --db {args.db} --publish {args.publish}`."
                ) from e
            print(f"published {args.event}/{RESULTS_FILE} to {args.publish} (data as of {args.data_as_of}): {url}")
    return path


def stage_run(args, panel: dict, specs: list[dict]) -> None:
    teams, accepted, cells, out_dir = load_state(args, specs)
    bill_to = args.bill_to or panel["bill_to"]
    done = {s["key"]: pn.latest_records(out_dir / f"{s['key']}.jsonl") for s in specs}
    todo = {s["key"]: sum(done[s["key"]].get(c["qid"], {}).get("status") != 200 for c in cells) for s in specs}
    if sum(todo.values()):
        cost = pn.estimate_cost(specs, todo)
        print(f"{sum(todo.values())} requests to send ({todo}), estimated cost <= ${cost:.2f}, billed to {bill_to}")
        print(f"projected wall time at {args.rate:.1f} req/s per model: {max(todo.values()) / args.rate / 60:.1f} min")
        if cost > args.budget_usd:
            raise pn.PanelError(f"estimate exceeds --budget_usd {args.budget_usd}")
        if not (args.yes or args.dry_run):
            raise pn.PanelError("paid stage: re-run with --yes (or --dry_run to test offline)")
        # Team questions are untrusted: a model that refuses or comments on a malformed question is scored as not
        # answering it, and only unmistakable reasoning counts toward the stop rule (strict_leaks=False). Leaks already
        # in a model's log do not count against this run's --max_leaks.
        previous = {k: sum(pn.is_reasoning_leak(r) for r in recs.values()) for k, recs in done.items()}
        max_leaks = {k: v + args.max_leaks for k, v in previous.items()}
        if args.no_billing_check:
            logger.warning("billing checks off: answers refused for billing (HTTP 402, spending-limit 403) stay pending")
        pn.run_panel_processes(
            specs,
            len(cells),
            out_dir,
            args,
            bill_to,
            max_leaks,
            strict_leaks=False,
            billing_stop=not args.no_billing_check,
            group_label="team",
        )
    else:
        print("every cell already has a panel answer; ranking only")
    rank_and_write(args, specs, teams, accepted, cells, out_dir)


def stage_verify(args, panel: dict, specs: list[dict]) -> None:
    """Check new or edited questions against their cited Wikipedia/Wikidata sources (`latamqa.source_check`), list
    the flagged ones, save ``rankings/sources_<time>.csv``, and re-rank so the review counts include the source flags.
    With ``--reader``, a panel model also reads each cited source and says which option it supports (paid)."""
    teams, accepted, cells, out_dir = load_state(args, specs)
    reader, litellm, token, bill_to = None, None, "", args.bill_to or panel["bill_to"]
    if args.reader:
        known = pn.all_models(panel)  # the run's panel first, then any panel file (e.g. a P6 model for P6-small)
        for path in sorted(pn.PANELS_DIR.glob("*.yaml")):
            try:
                known = pn.all_models(pn.load_panel(path)) | known
            except pn.PanelError as e:  # another panel's broken file must not block this one's reader
                logger.warning(str(e))
        if args.reader not in known:
            raise pn.PanelError(f"--reader {args.reader}: unknown model key; known: {sorted(known)}")
        reader = known[args.reader]
        current = sc.load_checks(out_dir, accepted)
        n = sum(
            any(sc.parse_sources(q.get("wikidata_qids")))
            and (q["id"] not in current or current[q["id"]].get("reader") != reader["key"])
            for q in accepted
        )
        cost = n * ((sc.EXCERPT_CHARS / 3 + 300) * reader["price"]["input"] + 16 * reader["price"]["output"]) / 1e6
        print(f"reader {reader['key']}: up to {n} requests, estimated cost <= ${cost:.2f}, billed to {bill_to}")
        if n and cost > args.budget_usd:
            raise pn.PanelError(f"estimate exceeds --budget_usd {args.budget_usd}")
        if n and not (args.yes or args.dry_run):
            raise pn.PanelError("the reader is paid: re-run with --yes (or --dry_run to test offline)")
        if n:
            litellm, token = pn.setup_litellm(args.dry_run, [reader]), pn.hf_token(args.dry_run)
    wiki = sc.Wiki(out_dir / "sources", pause=args.wiki_pause)
    checks, new = sc.check_all(
        accepted,
        out_dir,
        wiki,
        reader=reader if litellm else None,
        litellm=litellm,
        bill_to=bill_to,
        token=token,
        order_of=lambda qid: balanced_permutations(str(qid))[0][0],
    )
    print(f"checked {new} new or edited question(s); {len(checks)} of {len(accepted)} have a current check")
    write_source_report(checks, accepted, out_dir)
    rank_and_write(args, specs, teams, accepted, cells, out_dir)


def write_source_report(checks: dict[str, dict], accepted: list[dict], out_dir: Path) -> Path:
    """Save every current check to ``rankings/sources_<time>.csv`` and show the flagged ones in the terminal."""
    from rich import box
    from rich.console import Console
    from rich.table import Table
    from rich.text import Text

    rank_dir = out_dir / "rankings"
    rank_dir.mkdir(parents=True, exist_ok=True)
    path = rank_dir / f"sources_{unique_stamp(rank_dir, 'sources_{}.csv')}.csv"
    by_id = {q["id"]: q for q in accepted}
    columns = ["team", "question_id", "verdict", "flag", "issues", "articles", "sources", "reader", "reader_verdict"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for qid, r in sorted(checks.items(), key=lambda kv: (by_id[kv[0]]["team"], kv[0])):
            writer.writerow(
                dict(
                    r,
                    team=by_id[qid]["team"],
                    question_id=qid,
                    issues="; ".join(r["issues"]),
                    articles="; ".join(r["articles"]),
                )
            )
    verdicts: dict[str, int] = {}
    for r in checks.values():
        verdicts[r["verdict"]] = verdicts.get(r["verdict"], 0) + 1
    table = Table(
        title=f"Source check: {sum(r['flag'] for r in checks.values())} flagged of {len(checks)} checked",
        caption=" · ".join(f"{k}: {v}" for k, v in sorted(verdicts.items(), key=lambda kv: -kv[1])),
        caption_justify="left",
        box=box.ROUNDED,
        header_style="bold",
    )
    for name in ("Team", "Question", "Verdict", "Why"):
        table.add_column(name)
    for qid, r in sorted(checks.items(), key=lambda kv: (by_id[kv[0]]["team"], kv[0])):
        if r["flag"]:
            table.add_row(Text(by_id[qid]["team"]), Text(qid), Text(r["verdict"]), Text("\n".join(r["issues"]) or "—"))
    console = Console()
    console.print(table)
    console.print(f"-> {path}", highlight=False, soft_wrap=True)
    return path


def stage_rank(args, panel: dict, specs: list[dict]) -> None:
    teams, accepted, cells, out_dir = load_state(args, specs)
    rank_and_write(args, specs, teams, accepted, cells, out_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="datathon", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["check", "run", "rank", "verify"])
    parser.add_argument(
        "--db",
        default=os.environ.get("DATATHON_DB"),
        help="app database: a local datathon.db or hf:<org>/<dataset> (the app's backup repo); default $DATATHON_DB",
    )
    parser.add_argument("--event", default="llaca-2026", help="name of the results folder for this event")
    parser.add_argument("--panel", default=pn.DEFAULT_PANEL, help="panel name in latamqa/panels/ or a YAML path")
    parser.add_argument("--models", help="comma-separated model keys (default: the panel's models)")
    parser.add_argument("--swap", action="append", help="replace a panel model, e.g. --swap qwen2.5-72b=qwen2.5-72b-di")
    parser.add_argument("--score_floor", type=float, default=SCORE_FLOOR, help="contribution floor of the team score")
    parser.add_argument(
        "--consensus_flag",
        type=float,
        default=CONSENSUS_FLAG,
        help="flag a question for review when this share of the panel's answers picks the same wrong option",
    )
    parser.add_argument("--bill_to", help="HF org billed through X-HF-Bill-To (default: the panel's bill_to)")
    parser.add_argument(
        "--no_billing_check",
        action="store_true",
        help="go ahead when HF refuses requests for billing (HTTP 402, spending-limit 403): leave those answers pending "
        "instead of stopping the run; check does not fail on a billing org that cannot pay",
    )
    parser.add_argument("--rate", type=float, default=pn.DEFAULT_RATE, help="requests/s per model (default 13.3)")
    parser.add_argument("--cap", action="append", help="in-flight cap override, e.g. --cap kimi-k2=120")
    parser.add_argument("--max_leaks", type=int, default=10, help="stop a model after this many new reasoning leaks")
    parser.add_argument("--max_error_rate", type=float, default=0.01, help="stop a model above this error rate")
    parser.add_argument("--retry_rounds", type=int, default=2, help="slower passes for answers that still failed")
    parser.add_argument("--budget_usd", type=float, default=40.0, help="refuse to start if the cost estimate exceeds this")
    parser.add_argument(
        "--progress_s", type=float, default=60.0, help="seconds between progress lines when the output is not a terminal"
    )
    parser.add_argument("--results_dir", default=str(DEFAULT_RESULTS_DIR), help="root folder for datathon results")
    parser.add_argument(
        "--publish",
        metavar="ORG/DATASET",
        help="also upload the results file the datathon app imports to this private HF dataset (created private if "
        "missing; a public one is refused), as <event>/latest.json plus a dated copy",
    )
    parser.add_argument("--reader", help="verify: panel model key that also reads each cited source (paid), e.g. qwen3.5-397b")
    parser.add_argument("--wiki_pause", type=float, default=0.5, help="verify: seconds between Wikipedia/Wikidata requests")
    parser.add_argument("--yes", action="store_true", help="confirm a paid run")
    parser.add_argument("--dry_run", action="store_true", help="offline: fake provider, no network, no cost")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        panel = pn.load_panel(args.panel)
        specs = pn.select_models(panel, args.models, args.swap)
        args.panel_name = panel["name"]
        if args.publish:  # before any paid work: a run whose results cannot be published should not start
            args.publish = results_repo(args.publish)
            if args.db and args.db.startswith("hf:") and results_repo(args.db) == args.publish:
                raise pn.PanelError(f"--publish {args.publish} is the app's backup dataset (--db); use a separate one")
            if not args.dry_run:
                print(f"[PASS] results dataset {args.publish}: {ensure_results_repo(args.publish, pn.hf_token())}")
        if args.stage == "check":
            pn.stage_check(args, panel, specs, billing_check=not args.no_billing_check)
            return
        if not args.db:
            raise pn.PanelError("--db is required (a local datathon.db or hf:<org>/<dataset>), or set $DATATHON_DB")
        {"run": stage_run, "rank": stage_rank, "verify": stage_verify}[args.stage](args, panel, specs)
    except pn.PanelError as e:
        logger.fatal(str(e))
        sys.exit(-1)


if __name__ == "__main__":
    main()
