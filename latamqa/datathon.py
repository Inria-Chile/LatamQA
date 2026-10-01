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

Usage:
  datathon check
  datathon run  --db hf:inria-chile/db-datathon-test --yes
  datathon run  --db path/to/datathon.db --dry_run
  datathon rank --db hf:inria-chile/db-datathon-test
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
from latamqa.mcq_core import V2_PROMPT_TEMPLATE, build_prompt

logger = get_logger(__name__)

DEFAULT_RESULTS_DIR = Path(__file__).parent.parent / "results" / "datathon"
DB_FILE = "datathon.db"
ACCEPTED_STATES = ("SUBMITTED", "EVALUATED")
LANGUAGES = {  # language -> (question, answer, distractor1, distractor2, distractor3) columns of the questions table
    "regional": ("question_text", "answer", "distractor1", "distractor2", "distractor3"),
    "english": ("question_en", "answer_en", "distractor1_en", "distractor2_en", "distractor3_en"),
}
N_PERMUTATIONS = 4
SCORE_FLOOR = 0.5  # spec §7: a question earns points only above this score
QUOTA = 60  # spec §3.2: questions per team; the tie-break vectors are padded to this length
NO_ANSWER_FLAG = 0.30  # spec §6.2: questions with more unanswered replies than this go to the committee


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


def snapshot_db(source: str, dest_dir: Path) -> Path:
    """Copy the app database into ``dest_dir/datathon_<UTC time>.db`` and return the copy.

    ``source`` is a local SQLite file or ``hf:<org>/<dataset>`` (the app's private backup repo, file ``datathon.db``).
    The copy uses SQLite's backup API, so a live database in WAL mode is read consistently.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"datathon_{unique_stamp(dest_dir, 'datathon_{}.db')}.db"
    with tempfile.TemporaryDirectory() as tmp:
        if source.startswith("hf:"):
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(
                source[3:], DB_FILE, repo_type="dataset", local_dir=tmp, token=pn.hf_token(), force_download=True
            )
        else:
            path = source
            if not Path(path).exists():
                raise pn.PanelError(f"database not found: {path}")
        src = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        dst = sqlite3.connect(dest)
        try:
            src.backup(dst)
        finally:
            src.close()
            dst.close()
    return dest


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
        questions = [dict(r) for r in con.execute(f"SELECT {', '.join(wanted)} FROM questions")]
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
                        options=shown,
                        correct=correct,
                        prompt=build_prompt(V2_PROMPT_TEMPLATE, stem, shown),
                    )
                )
    return accepted, cells, problems


# ---------------------------------------------------------------------------------------------------------- scoring


def score_questions(accepted: list[dict], cells: list[dict], logs: dict[str, dict[str, dict]]) -> dict[str, dict]:
    """Per-question panel results over the current cells: received, correct, unanswered, pending, score, accuracy.

    ``logs`` maps each model key to its latest record per cell id (`panel.latest_records`).
    """
    stats = {q["id"]: dict(received=0, correct=0, no_answer=0, pending=0) for q in accepted}
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
    for s in stats.values():
        n = s["received"]
        s["accuracy"] = s["correct"] / n if n else None
        s["score"] = 1 - s["correct"] / n if n else None
        s["no_answer_rate"] = s["no_answer"] / n if n else None
    return stats


def rank_teams(
    teams: list[dict], accepted: list[dict], stats: dict[str, dict], floor: float = SCORE_FLOOR, quota: int = QUOTA
) -> list[dict]:
    """Team rows sorted by the spec's team score and tie-breaks, with the mean panel accuracy for reference.

    A question with no answer received yet contributes 0 (spec §3.1). Teams without accepted questions rank last
    among equal scores.
    """
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
                team=t["name"],
                country=t["country"],
                team_score=sum(max(0.0, s - floor) for s in scored),
                mean_accuracy=sum(accuracies) / len(accuracies) if accuracies else None,
                accepted=len(qs),
                scored=len(scored),
                earning=sum(s > floor for s in scored),
                pending_questions=sum(stats[q["id"]]["pending"] > 0 for q in qs),
                flagged_no_answer=sum((stats[q["id"]]["no_answer_rate"] or 0) > NO_ANSWER_FLAG for q in qs),
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
        "earning points = score above the floor; pending = questions still missing answers."
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
        caption=f"{note}\n{footnote(floor)}",
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
    for name in ("Accepted", "Scored", "Earning points", "Pending"):
        table.add_column(name, justify="right")
    for r in rows:
        style = "bold" if r["rank"] <= 3 and r["team_score"] > 0 else "dim" if r["team_score"] == 0 else None
        pending = Text(str(r["pending_questions"]), style="yellow" if r["pending_questions"] else "")
        table.add_row(
            str(r["rank"]),
            str(r["team"]),
            str(r["country"] or "—"),
            f"{r['team_score']:.4f}",
            _pct(r["mean_accuracy"]),
            str(r["accepted"]),
            str(r["scored"]),
            str(r["earning"]),
            pending,
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
) -> Path:
    """Write the ranking (Markdown + CSV) and the per-question scores for this run, refresh ``ranking_latest``, and
    show the ranking in the terminal."""
    rank_dir = out_dir / "rankings"
    rank_dir.mkdir(parents=True, exist_ok=True)
    columns = ["rank", "team", "country", "team_score", "mean_accuracy", "accepted", "scored", "earning"]
    columns += ["pending_questions", "flagged_no_answer"]
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
        "| Rank | Team | Country | Team score | Mean panel accuracy | Accepted | Scored | Earning points | Pending |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    lines += [
        f"| {r['rank']} | {_cell(r['team'])} | {_cell(r['country'])} | {r['team_score']:.4f} | {_pct(r['mean_accuracy'])} | "
        f"{r['accepted']} | {r['scored']} | {r['earning']} | {r['pending_questions']} |"
        for r in rows
    ]
    lines += ["", footnote(floor)]
    md = "\n".join(lines) + "\n"
    (rank_dir / f"ranking_{stamp}.md").write_text(md, encoding="utf-8")
    shutil.copyfile(rank_dir / f"ranking_{stamp}.md", out_dir / "ranking_latest.md")
    shutil.copyfile(rank_dir / f"ranking_{stamp}.csv", out_dir / "ranking_latest.csv")
    qcols = ["team", "country", "question_id", "sequence", "state", "score", "accuracy", "contribution", "received"]
    qcols += ["correct", "pending", "no_answer_rate", "flag_no_answer"]
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
                    flag_no_answer=(s["no_answer_rate"] or 0) > NO_ANSWER_FLAG,
                )
            )
    from rich.console import Console

    console = Console()
    console.print(render_ranking(rows, stamp, status, note, floor))
    console.print(f"-> {rank_dir / f'ranking_{stamp}.md'}", highlight=False, soft_wrap=True)
    return rank_dir / f"ranking_{stamp}.md"


# ---------------------------------------------------------------------------------------------------------- stages


def event_dir(args) -> Path:
    return Path(args.results_dir) / args.event


def load_state(args, specs: list[dict]) -> tuple[list[dict], list[dict], list[dict], Path]:
    """Snapshot the database and build the current cells; write ``items.json`` for the panel workers."""
    out_dir = event_dir(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    snapshot = snapshot_db(args.db, out_dir / "snapshots")
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
    rows = rank_teams(teams, accepted, stats, args.score_floor)
    aborted = [s["key"] for s in specs if (out_dir / f"{s['key']}.ABORT").exists()]
    pending = sum(r["pending_questions"] for r in rows)
    complete = not aborted and not pending
    status = "complete" if complete else "PROVISIONAL"
    note = f"Panel {args.panel_name} ({len(specs)} models), {len(accepted)} accepted questions, {len(teams)} teams."
    if pending:
        note += f" {pending} question(s) still missing panel answers."
    if aborted:
        note += f" Stopped models: {', '.join(aborted)} (see their .ABORT files)."
    if args.dry_run:
        note += " DRY RUN: fake answers, not a real ranking."
    stamp = unique_stamp(out_dir / "rankings", "ranking_{}.md")
    return write_outputs(rows, accepted, stats, out_dir, stamp, status, note, args.score_floor)


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
        # leaks already in a model's log do not count against this run's --max_leaks
        previous = {k: sum(pn.is_leak(r) for r in recs.values()) for k, recs in done.items()}
        pn.run_panel_processes(specs, len(cells), out_dir, args, bill_to, {k: v + args.max_leaks for k, v in previous.items()})
    else:
        print("every cell already has a panel answer; ranking only")
    rank_and_write(args, specs, teams, accepted, cells, out_dir)


def stage_rank(args, panel: dict, specs: list[dict]) -> None:
    teams, accepted, cells, out_dir = load_state(args, specs)
    rank_and_write(args, specs, teams, accepted, cells, out_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="datathon", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["check", "run", "rank"])
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
    parser.add_argument("--bill_to", help="HF org billed through X-HF-Bill-To (default: the panel's bill_to)")
    parser.add_argument("--rate", type=float, default=pn.DEFAULT_RATE, help="requests/s per model (default 13.3)")
    parser.add_argument("--cap", action="append", help="in-flight cap override, e.g. --cap kimi-k2=120")
    parser.add_argument("--max_leaks", type=int, default=10, help="stop a model after this many new reasoning leaks")
    parser.add_argument("--max_error_rate", type=float, default=0.01, help="stop a model above this error rate")
    parser.add_argument("--retry_rounds", type=int, default=2, help="slower passes for answers that still failed")
    parser.add_argument("--budget_usd", type=float, default=40.0, help="refuse to start if the cost estimate exceeds this")
    parser.add_argument("--progress_s", type=float, default=60.0, help="seconds between progress lines")
    parser.add_argument("--results_dir", default=str(DEFAULT_RESULTS_DIR), help="root folder for datathon results")
    parser.add_argument("--yes", action="store_true", help="confirm a paid run")
    parser.add_argument("--dry_run", action="store_true", help="offline: fake provider, no network, no cost")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        panel = pn.load_panel(args.panel)
        specs = pn.select_models(panel, args.models, args.swap)
        args.panel_name = panel["name"]
        if args.stage == "check":
            pn.stage_check(args, panel, specs)
            return
        if not args.db:
            raise pn.PanelError("--db is required (a local datathon.db or hf:<org>/<dataset>), or set $DATATHON_DB")
        {"run": stage_run, "rank": stage_rank}[args.stage](args, panel, specs)
    except pn.PanelError as e:
        logger.fatal(str(e))
        sys.exit(-1)


if __name__ == "__main__":
    main()
