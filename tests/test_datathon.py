"""Tests for the datathon evaluator (`latamqa.datathon`): the app's option orders, reading the app database, the spec's
scoring and tie-breaks, the ranking files, and an offline end-to-end run. No network access is needed."""

import inspect
import json
import sqlite3
import types

import pytest

from latamqa import datathon as dtn
from latamqa import panel as pn

LANG_COLS = [c for cols in dtn.LANGUAGES.values() for c in cols]


def make_db(path, teams, questions):
    """A minimal datathon app database: the teams and questions columns the evaluator reads."""
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE teams (id INTEGER PRIMARY KEY, name TEXT, country TEXT)")
    cols = ["id TEXT PRIMARY KEY", "team_id INTEGER", "state TEXT", "submission_sequence_number INTEGER"]
    con.execute(f"CREATE TABLE questions ({', '.join(cols + [f'{c} TEXT' for c in LANG_COLS])})")
    con.executemany("INSERT INTO teams VALUES (?, ?, ?)", teams)
    for q in questions:
        con.execute(f"INSERT INTO questions ({', '.join(q)}) VALUES ({', '.join('?' * len(q))})", list(q.values()))
    con.commit()
    con.close()
    return path


def question(qid, team_id, seq, state="SUBMITTED", **overrides):
    q = dict(id=qid, team_id=team_id, state=state, submission_sequence_number=seq)
    q.update(question_text=f"Pregunta {qid}?", answer=f"Sí {qid}", distractor1="No 1", distractor2="No 2", distractor3="No 3")
    q.update(question_en=f"Question {qid}?", answer_en=f"Yes {qid}", distractor1_en="No 1", distractor2_en="No 2")
    q.update(distractor3_en="No 3")
    q.update(overrides)
    return q


TEAMS = [(1, "Equipo Uno", "Chile"), (2, "Equipo | Dos", "Brasil"), (3, "Equipo Tres", "Uruguay")]
QUESTIONS = [
    question("q1", 1, 1),
    question("q2", 1, 2, state="EVALUATED"),
    question("q3", 2, 3),
    question("q4", 2, 4, state="EXCLUDED"),
    question("q5", 2, 5, state="WITHDRAWN"),
    question("q6", 1, None, state="DRAFT"),
    question("q7", 2, 6, distractor3_en="No 2"),  # English options not distinct: only the regional version is asked
]


@pytest.fixture
def db(tmp_path):
    return make_db(tmp_path / "datathon.db", TEAMS, QUESTIONS)


# ---------------------------------------------------------------------------------------------------------- questions


GOLDEN = {  # from llaca-datathon-app evaluator.get_balanced_permutations(qid, ["ANS", "D1", "D2", "D3"], "A")
    "d28b2b6b-41f0-4160-9415-ce8d25dcfd3c": [
        (["D1", "D2", "ANS", "D3"], "C"),
        (["D1", "D2", "D3", "ANS"], "D"),
        (["ANS", "D1", "D2", "D3"], "A"),
        (["D1", "ANS", "D2", "D3"], "B"),
    ],
    "e57b6436-01d1-4486-99a0-db1bca8deeef": [
        (["ANS", "D1", "D2", "D3"], "A"),
        (["D1", "ANS", "D2", "D3"], "B"),
        (["D1", "D2", "ANS", "D3"], "C"),
        (["D1", "D2", "D3", "ANS"], "D"),
    ],
    "q-odd-1": [
        (["D2", "ANS", "D1", "D3"], "B"),
        (["D2", "D1", "ANS", "D3"], "C"),
        (["D2", "D1", "D3", "ANS"], "D"),
        (["ANS", "D2", "D1", "D3"], "A"),
    ],
}


@pytest.mark.parametrize("qid", sorted(GOLDEN))
def test_balanced_permutations_match_the_app(qid):
    labels = ["ANS", "D1", "D2", "D3"]
    got = [([labels[i] for i in order], letter) for order, letter in dtn.balanced_permutations(qid)]
    assert got == GOLDEN[qid]


def test_balanced_permutations_put_the_answer_once_in_each_position():
    for qid in ("a", "b", "c", "d", "e"):
        perms = dtn.balanced_permutations(qid)
        assert sorted(letter for _, letter in perms) == list("ABCD")
        for order, letter in perms:
            assert order["ABCD".index(letter)] == 0 and sorted(order) == [0, 1, 2, 3]


def test_read_db_and_build_cells(db):
    teams, questions = dtn.read_db(db)
    assert len(teams) == 3 and len(questions) == 7
    accepted, cells, problems = dtn.build_cells(teams, questions)
    assert [q["id"] for q in accepted] == ["q1", "q2", "q3", "q7"]  # not excluded, withdrawn or draft
    assert len(cells) == 3 * 2 * 4 + 1 * 4  # q7 has only its regional version
    assert problems == ["question q7 (Equipo | Dos): english version incomplete or options not distinct"]
    first = cells[0]
    assert first["qid"].startswith("q1|regional|p0|") and first["group"] == "Equipo Uno"
    assert first["options"][ord(first["correct"]) - ord("A")] == "Sí q1"
    assert first["prompt"].startswith("Answer the following") and "Pregunta q1?" in first["prompt"]
    english = [c for c in cells if c["question_id"] == "q1" and c["lang"] == "english"]
    assert [c["correct"] for c in english] == [c["correct"] for c in cells if c["question_id"] == "q1"][:4]


def test_an_edit_creates_new_cells(db):
    teams, questions = dtn.read_db(db)
    _, before, _ = dtn.build_cells(teams, questions)
    questions[0]["question_en"] = "Question q1, reworded?"
    _, after, _ = dtn.build_cells(teams, questions)
    changed = {c["qid"] for c in after} - {c["qid"] for c in before}
    assert len(changed) == 4 and all(c.startswith("q1|english|") for c in changed)


def test_read_db_rejects_other_databases(tmp_path):
    path = tmp_path / "other.db"
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE teams (id INTEGER, name TEXT, country TEXT)")
    con.execute("CREATE TABLE questions (id TEXT)")
    con.close()
    with pytest.raises(pn.PanelError, match="lacks question columns"):
        dtn.read_db(path)


def test_snapshot_copies_a_local_database(db, tmp_path):
    copy = dtn.snapshot_db(str(db), tmp_path / "snaps")
    assert copy.parent == tmp_path / "snaps" and dtn.read_db(copy)[1] == dtn.read_db(db)[1]
    with pytest.raises(pn.PanelError, match="not found"):
        dtn.snapshot_db(str(tmp_path / "missing.db"), tmp_path / "snaps")


# ---------------------------------------------------------------------------------------------------------- scoring


def _ok(cell, letter):
    return {"qid": cell["qid"], "status": 200, "content": letter, "finish": "stop", "completion_tokens": 1}


def test_score_questions_follows_the_spec(db):
    teams, questions = dtn.read_db(db)
    accepted, cells, _ = dtn.build_cells(teams, questions)
    q1 = [c for c in cells if c["question_id"] == "q1"]
    wrong = {"A": "B", "B": "C", "C": "D", "D": "A"}
    model_a = {c["qid"]: _ok(c, c["correct"]) for c in q1}  # always right
    model_b = {c["qid"]: _ok(c, wrong[c["correct"]]) for c in q1[:6]}  # wrong on 6 cells
    model_b[q1[6]["qid"]] = {"qid": q1[6]["qid"], "status": 503}  # an error: pending, not scored
    model_b[q1[7]["qid"]] = {"qid": q1[7]["qid"], "status": 200, "content": "No sé", "finish": "stop"}  # unparsable
    stats = dtn.score_questions(accepted, cells, {"a": model_a, "b": model_b})
    s = stats["q1"]
    assert (s["received"], s["correct"], s["pending"], s["no_answer"]) == (15, 8, 1, 1)
    assert s["accuracy"] == pytest.approx(8 / 15) and s["score"] == pytest.approx(7 / 15)
    assert stats["q2"]["score"] is None and stats["q2"]["pending"] == 16  # nothing received yet


def _stats(**scores):
    return {
        qid: dict(score=s, accuracy=None if s is None else 1 - s, pending=0, no_answer_rate=0.0, consensus_share=None)
        for qid, s in scores.items()
    }


def test_rank_teams_uses_the_floored_sum_and_reports_mean_accuracy():
    teams = [{"id": 1, "name": "T1", "country": "CL"}, {"id": 2, "name": "T2", "country": "BR"}]
    accepted = [
        dict(id="a", team_id=1, submission_sequence_number=1),
        dict(id="b", team_id=1, submission_sequence_number=2),
        dict(id="c", team_id=2, submission_sequence_number=3),
    ]
    rows = dtn.rank_teams(teams, accepted, _stats(a=0.9, b=0.4, c=0.8))
    assert [r["team"] for r in rows] == ["T1", "T2"]
    assert rows[0]["team_score"] == pytest.approx(0.4) and rows[1]["team_score"] == pytest.approx(0.3)
    assert rows[0]["mean_accuracy"] == pytest.approx((0.1 + 0.6) / 2) and rows[0]["earning"] == 1


def test_rank_teams_tie_breaks():
    teams = [{"id": i, "name": f"T{i}", "country": "CL"} for i in (1, 2, 3, 4)]
    accepted = [
        dict(id="a1", team_id=1, submission_sequence_number=10),
        dict(id="a2", team_id=1, submission_sequence_number=11),  # same floored score as T2, deeper bench
        dict(id="b1", team_id=2, submission_sequence_number=1),
        dict(id="c1", team_id=3, submission_sequence_number=5),
        dict(id="d1", team_id=4, submission_sequence_number=2),  # same as T3 but submitted earlier
    ]
    stats = _stats(a1=0.8, a2=0.4, b1=0.8, c1=0.3, d1=0.3)
    rows = dtn.rank_teams(teams, accepted, stats)
    assert [r["team"] for r in rows] == ["T1", "T2", "T4", "T3"]
    rows = dtn.rank_teams([*teams, {"id": 5, "name": "T0", "country": "UY"}], accepted, stats)
    assert rows[-1]["team"] == "T0" and rows[-1]["accepted"] == 0  # no questions: last among equal scores


def test_render_ranking():
    import io

    from rich.console import Console

    rows = [
        dict(rank=1, team="T | One", country="CL", team_score=1.25, mean_accuracy=0.2, accepted=3, scored=3, earning=3,
             pending_questions=0),
        dict(rank=2, team="T2", country=None, team_score=0.0, mean_accuracy=None, accepted=0, scored=0, earning=0,
             pending_questions=2),
    ]  # fmt: skip
    console = Console(file=io.StringIO(), record=True, width=160)
    console.print(dtn.render_ranking(rows, "20261003T120000Z", "PROVISIONAL", "a note", floor=0.4))
    text = console.export_text()
    assert "Datathon ranking · 20261003T120000Z · PROVISIONAL" in text
    assert "T | One" in text and "1.2500" in text and "20.0%" in text and "—" in text
    assert "a note" in text and "max(0, score - 0.4)" in text


def test_write_outputs(tmp_path, db):
    teams, questions = dtn.read_db(db)
    accepted, cells, _ = dtn.build_cells(teams, questions)
    stats = dtn.score_questions(accepted, cells, {})
    rows = dtn.rank_teams(teams, accepted, stats)
    path = dtn.write_outputs(rows, accepted, stats, tmp_path, "20261003T120000Z", "PROVISIONAL", "note")
    md = path.read_text()
    assert "Equipo \\| Dos" in md and "PROVISIONAL" in md
    assert (tmp_path / "ranking_latest.md").read_text() == md
    assert (tmp_path / "ranking_latest.csv").read_text().startswith("rank,team,country,team_score,mean_accuracy")
    assert (tmp_path / "rankings" / "questions_20261003T120000Z.csv").exists()
    assert "max(0, score - 0.5)" in md


# ---------------------------------------------------------------------------------------------------------- end to end


def test_datathon_dry_run_end_to_end(db, tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_API_KEY", raising=False)
    out = tmp_path / "out"
    base = ["--db", str(db), "--results_dir", str(out), "--dry_run", "--rate", "500", "--progress_s", "0.2"]
    base += ["--models", "qwen3-4b,kimi-k2"]
    dtn.main(["run", *base])
    event = out / "llaca-2026"
    rows = list(__import__("csv").DictReader(open(event / "ranking_latest.csv", encoding="utf-8")))
    assert {r["team"] for r in rows} == {"Equipo Uno", "Equipo | Dos", "Equipo Tres"}
    assert all(r["pending_questions"] == "0" for r in rows)
    assert "complete" in (event / "ranking_latest.md").read_text()
    assert "Replies without an answer letter: qwen3-4b 0.0%, kimi-k2 0.0%" in (event / "ranking_latest.md").read_text()
    n_lines = sum(1 for _ in open(event / "kimi-k2.jsonl"))
    assert n_lines == 28  # 3 questions x 2 languages x 4 orders + 4 regional-only cells; Kimi's switch sent, no leak
    assert not any(pn.is_leak(json.loads(line)) for line in open(event / "kimi-k2.jsonl"))
    dtn.main(["run", *base])  # incremental: nothing new to ask
    assert sum(1 for _ in open(event / "kimi-k2.jsonl")) == n_lines
    dtn.main(["rank", *base])
    assert len(list((event / "rankings").glob("ranking_*.md"))) == 3


def test_no_billing_check_reaches_the_panel_workers(db, tmp_path, monkeypatch):
    spawned = []

    def process(target, args=(), kwargs=None):  # records each worker's run_model arguments instead of spawning it
        bound = inspect.signature(target).bind(*args, **(kwargs or {}))
        bound.apply_defaults()
        spawned.append(bound.arguments)
        return types.SimpleNamespace(start=lambda: None, is_alive=lambda: False, join=lambda: None)

    monkeypatch.setattr(pn, "mp", types.SimpleNamespace(get_context=lambda method: types.SimpleNamespace(Process=process)))
    base = ["run", "--db", str(db), "--results_dir", str(tmp_path / "out"), "--dry_run", "--models", "qwen3-4b,kimi-k2"]
    dtn.main(base)
    dtn.main([*base, "--no_billing_check"])
    assert [(a["billing_stop"], a["strict_leaks"]) for a in spawned] == [(True, False)] * 2 + [(False, False)] * 2


def test_ranking_counts_answers_refused_for_billing(db, tmp_path):
    base = ["rank", "--db", str(db), "--results_dir", str(tmp_path / "out"), "--models", "qwen3-4b"]
    dtn.main(base)
    event = tmp_path / "out" / "llaca-2026"
    cells = json.loads((event / "items.json").read_text())
    limit = 'HuggingfaceException - {"error":"You have exceeded your monthly spending limit for Inference Providers."}'
    recs = [
        dict(qid=cells[0]["qid"], status=403, error_msg=limit),
        dict(qid=cells[1]["qid"], status=402, error_msg="Payment Required"),
        dict(qid=cells[2]["qid"], status=503, error_msg="busy"),  # an ordinary error
        dict(qid=cells[3]["qid"], status=402, error_msg="Payment Required"),
        _ok(cells[3], cells[3]["correct"]),  # answered later: not refused any more
    ]
    (event / "qwen3-4b.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    dtn.main(base)
    assert "2 answer(s) refused for billing" in (event / "ranking_latest.md").read_text()


def test_check_without_billing_check_only_reports_the_billing_org(monkeypatch, capsys):
    import importlib.metadata

    def fake_http_json(url, token=None, timeout=30):
        if url.endswith("/api/whoami-v2"):
            return 200, {"name": "someone", "orgs": [{"name": "inria-chile", "canPay": False}]}
        return 200, {"inferenceProviderMapping": {"nscale": {"status": "live", "providerId": "x"}}, "gated": False}

    monkeypatch.setattr(pn, "_http_json", fake_http_json)
    monkeypatch.setattr(pn, "TESTED_LITELLM", importlib.metadata.version("litellm"))
    monkeypatch.setenv("HF_TOKEN", "tok")
    for var in ("HF_API_BASE", "HUGGINGFACE_API_BASE"):
        monkeypatch.delenv(var, raising=False)
    base = ["check", "--models", "qwen3-4b"]
    with pytest.raises(SystemExit):
        dtn.main(base)
    assert "[FAIL] org inria-chile can pay" in capsys.readouterr().out
    dtn.main([*base, "--no_billing_check"])
    out = capsys.readouterr().out
    assert "[INFO] org inria-chile can pay: member=True canPay=False" in out and "All checks passed." in out
    with pytest.raises(SystemExit):  # a mistyped org still fails
        dtn.main([*base, "--no_billing_check", "--bill_to", "inria-chlie"])
    assert "[FAIL] org inria-chlie can pay: member=False" in capsys.readouterr().out


def test_datathon_needs_a_database(monkeypatch):
    monkeypatch.delenv("DATATHON_DB", raising=False)
    with pytest.raises(SystemExit):
        dtn.main(["rank"])


def test_consensus_on_one_wrong_option(db):
    teams, questions = dtn.read_db(db)
    accepted, cells, _ = dtn.build_cells(teams, questions)
    q1 = [c for c in cells if c["question_id"] == "q1"]
    pick = {c["qid"]: "ABCD"[c["order"].index(1)] for c in q1}  # always distractor 1, whatever the order or language
    logs = {m: {c["qid"]: _ok(c, pick[c["qid"]]) for c in q1} for m in ("a", "b")}
    stats = dtn.score_questions(accepted, cells, logs)
    assert stats["q1"]["consensus_option"] == "No 1" and stats["q1"]["consensus_share"] == 1.0
    assert stats["q1"]["score"] == 1.0 and stats["q2"]["consensus_share"] is None
    reasons = dtn.review_reasons(stats, {"q3": {"flag": True}})
    assert reasons["q1"] == ["consensus"] and reasons["q3"] == ["source"] and reasons["q2"] == []
    rows = {r["team"]: r for r in dtn.rank_teams(teams, accepted, stats, reasons=reasons)}
    assert rows["Equipo Uno"]["review"] == 1 and rows["Equipo Uno"]["flagged_consensus"] == 1
    assert rows["Equipo | Dos"]["review"] == 0 and rows["Equipo | Dos"]["flagged_source"] == 1  # q3 earns nothing


def test_verify_stage_without_network(db, tmp_path, monkeypatch):
    from latamqa import source_check as sc

    calls = []
    monkeypatch.setattr(sc.Wiki, "_http", lambda self, url: calls.append(url) or (404, ""))
    monkeypatch.delenv("HF_TOKEN", raising=False)
    out = tmp_path / "out"
    dtn.main(["verify", "--db", str(db), "--results_dir", str(out), "--models", "qwen3-4b", "--dry_run"])
    event = out / "llaca-2026"
    checks = json.loads((event / sc.CHECKS_FILE).read_text())
    assert set(checks) == {"q1", "q2", "q3", "q7"} and {c["verdict"] for c in checks.values()} == {"no source"}
    assert calls == []  # no sources cited, nothing fetched
    assert "Source check: 4 of 4 questions checked" in (event / "ranking_latest.md").read_text()
    assert list((event / "rankings").glob("sources_*.csv"))
