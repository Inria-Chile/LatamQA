"""Tests for the live progress of a panel run (`latamqa.progress`): tailing the logs, counting by model and by group,
the rich display and its fall-back to plain lines. No network access is needed."""

import io
import json
import multiprocessing as mp
import os
import types
from collections import Counter

import pytest
from rich.console import Console

from latamqa import progress as pg

SPECS = [{"key": "m1"}, {"key": "m2"}]
LIMIT = 'HuggingfaceException - {"error":"You have exceeded your monthly spending limit for Inference Providers."}'


def _items(teams: dict[str, int]) -> list[dict]:
    return [dict(qid=f"{team}|{i}", group=team) for team, n in teams.items() for i in range(n)]


def _log(path, *recs):
    with open(path, "a", encoding="utf-8") as f:
        f.writelines(json.dumps(r) + "\n" for r in recs)


def _ok(qid, **kw):
    return dict(qid=qid, status=200, content="B", finish="stop", completion_tokens=1, **kw)


def _console(width=100, height=30, interactive=False):
    return Console(file=io.StringIO(), width=width, height=height, record=True, color_system=None,
                   force_terminal=interactive, force_interactive=interactive)  # fmt: skip


@pytest.fixture
def run_dir(tmp_path):
    items = _items({"Equipo Uno": 2, "Equipo Dos": 3, "Equipo Diez": 1})
    (tmp_path / "items.json").write_text(json.dumps(items))
    return tmp_path


def _proc(alive=True, exitcode=0):
    return types.SimpleNamespace(is_alive=lambda: alive, exitcode=None if alive else exitcode)


# ---------------------------------------------------------------------------------------------------------- state


def test_log_tail_reads_only_new_complete_lines(tmp_path):
    path = tmp_path / "m.jsonl"
    tail = pg.LogTail(path)
    assert tail.read() == []  # no log yet
    path.write_text('{"qid": "a"}\n{"qid": "b"}\n{"qid": "c"')  # the last line is still being written
    assert [r["qid"] for r in tail.read()] == ["a", "b"]
    with open(path, "a") as f:
        f.write(', "status": 200}\nnot json\n{"qid": "d"}\n')
    assert tail.read() == [{"qid": "c", "status": 200}, {"qid": "d"}]  # the cut line is skipped
    assert tail.read() == []
    path.write_text('{"qid": "e"}\n')  # rewritten from scratch
    assert tail.read() == [{"qid": "e"}]


def test_names():
    assert sorted(["T 10", "t 9", "T 1"], key=pg.natural_key) == ["T 1", "t 9", "T 10"]
    assert pg.short_names(["Equipo Chile 3", "Equipo Chile 12", "Equipo A"]) == {
        "Equipo Chile 3": "Chile 3",
        "Equipo Chile 12": "Chile 12",
        "Equipo A": "A",
    }
    assert pg.short_names(["Equipo", "Equipo B"]) == {"Equipo": "Equipo", "Equipo B": "Equipo B"}  # no whole word
    assert pg.short_names(["Solo"]) == {"Solo": "Solo"}
    assert pg._pct(1167, 1168) == "99%" and pg._pct(0, 0) == "—"


def test_counts_by_model_and_group(run_dir):
    _log(run_dir / "m1.jsonl", _ok("Equipo Uno|0"), dict(qid="Equipo Uno|1", status=503), _ok("old|0"))
    watch = pg.RunWatch(SPECS, run_dir, 6, "team", console=_console())
    assert watch.groups == ["Equipo Diez", "Equipo Dos", "Equipo Uno"]  # natural order
    assert watch.answered("m1") == 1 and watch.counts["m1"]["error"] == 1  # the cell no longer asked is left out
    _log(
        run_dir / "m1.jsonl",
        _ok("Equipo Uno|1"),  # answered after an error
        dict(qid="Equipo Uno|0", status=503),  # an error after a 200 does not undo it
        dict(qid="Equipo Dos|0", status=402),
        dict(qid="Equipo Dos|1", status=403, error_msg=LIMIT),
        _ok("Equipo Dos|2", reasoning_len=900),  # a leak is answered, and counted
    )
    m2 = [_ok("Equipo Uno|0"), _ok("Equipo Uno|1"), _ok("Equipo Uno|0"), dict(qid="Equipo Diez|0", status=403, error_msg="no")]
    _log(run_dir / "m2.jsonl", *m2)  # a cell answered twice counts once
    watch.update()
    assert +watch.counts["m1"] == Counter(ok=2, leak=1, refused=2) and +watch.counts["m2"] == Counter(ok=2, error=1)
    assert watch.answered("m1") == 3 and watch.answered("m2") == 2
    assert watch.group_done == {"Equipo Uno": 4, "Equipo Dos": 1}
    assert watch.group_complete("Equipo Uno") and not watch.group_complete("Equipo Dos")
    assert watch.line().endswith("m1 3/6 err 2 leak 1 | m2 2/6 err 1 leak 0 || teams fully evaluated 1/3")


def test_rates_use_the_last_answers(run_dir, monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(pg.time, "time", lambda: now[0])
    watch = pg.RunWatch(SPECS, run_dir, 6, console=_console())
    now[0] += 10
    _log(run_dir / "m1.jsonl", _ok("Equipo Uno|0"), _ok("Equipo Uno|1"))
    watch.update()
    assert watch.rate("m1") == pytest.approx(0.2) and watch.rate("m2") == 0
    now[0] += 60  # nothing new for a minute: the rate falls to zero
    watch.update()
    assert watch.rate("m1") == 0


def test_only_stops_of_this_run_count(run_dir):
    (run_dir / "m1.ABORT").write_text("billing: earlier run\n")
    (run_dir / "m2.ABORT").write_text("billing: earlier run\n")
    os.utime(run_dir / "m1.ABORT", (1_000_000_000, 1_000_000_000))
    os.utime(run_dir / "m2.ABORT", (1_000_000_000, 1_000_000_000))
    watch = pg.RunWatch(SPECS, run_dir, 6, console=_console())
    assert watch.status("m1", _proc()) == ("running", "cyan", "")
    (run_dir / "m2.ABORT").write_text("stopped: m1: HTTP 402 (billing)\n")  # rewritten by this run's worker
    assert watch.status("m2", _proc(alive=False)) == ("stopped", "red", "m1: HTTP 402 (billing)")
    assert "m1 0/6 err 0 leak 0 | m2 0/6 err 0 leak 0 ABORTED ||" in watch.line()
    assert watch.status("m1", _proc(alive=False)) == ("done", "green", "")
    assert watch.status("m1", _proc(alive=False, exitcode=1))[:2] == ("exited", "red")


# ---------------------------------------------------------------------------------------------------------- display


def _frame(watch, procs, width=100, height=30):
    watch.console = _console(width, height, interactive=True)
    watch.console.print(watch.render(procs))
    return watch.console.export_text()


def test_render_shows_models_and_teams(run_dir):
    _log(run_dir / "m1.jsonl", _ok("Equipo Uno|0"), _ok("Equipo Uno|1"), dict(qid="Equipo Dos|0", status=402))
    _log(run_dir / "m2.jsonl", _ok("Equipo Uno|0"), _ok("Equipo Uno|1"), _ok("Equipo Dos|1"))
    (run_dir / "m2.ABORT").touch()  # an earlier run's
    watch = pg.RunWatch(SPECS, run_dir, 6, "team", console=_console())
    (run_dir / "m2.ABORT").write_text("HTTP 403: forbidden\n")
    text = _frame(watch, {"m1": _proc(), "m2": _proc(alive=False)})
    assert "2 models x 6 answers" in text and "all models" in text and "5/12" in text
    m1, m2 = (next(line for line in text.splitlines() if f" {k} " in line) for k in ("m1", "m2"))
    assert "2/6" in m1 and "33%" in m1 and "refused 1" in m1 and "running" in m1
    assert "3/6" in m2 and "stopped" in m2 and "m2 stopped: HTTP 403: forbidden" in text
    assert "Teams fully evaluated" in text and "1/3" in text
    uno, dos, diez = (next(line for line in text.splitlines() if f" {t} " in line) for t in ("Uno", "Dos", "Diez"))
    assert "✓" in uno and "16%" in dos and "0%" in diez  # short names, in natural order
    assert len(text.splitlines()) <= 30


def test_render_makes_room_in_small_terminals(tmp_path):
    teams = {f"Equipo {i}": 1 for i in range(1, 61)}
    (tmp_path / "items.json").write_text(json.dumps(_items(teams)))
    _log(tmp_path / "m1.jsonl", *(_ok(f"Equipo {i}|0") for i in range(1, 31)))
    _log(tmp_path / "m2.jsonl", *(_ok(f"Equipo {i}|0") for i in range(1, 41)))
    watch = pg.RunWatch(SPECS, tmp_path, 60, "team", console=_console())
    text = _frame(watch, {}, width=80, height=16)
    assert len(text.splitlines()) <= 16 and "Teams fully evaluated" in text
    assert "+30 fully evaluated" in text and "more not shown" in text and "✓" not in text
    assert "Equipo" not in text.split("Teams fully evaluated")[1]  # the shared word is dropped
    text = _frame(watch, {}, width=80, height=9)  # no room for the teams' bars
    assert "Teams fully evaluated" in text and "+30" not in text
    (tmp_path / "items.json").write_text(json.dumps(_items({"all": 60})))  # a single group: no group view
    text = _frame(pg.RunWatch(SPECS, tmp_path, 60, console=_console()), {})
    assert "fully evaluated" not in text and "m1" in text


# ---------------------------------------------------------------------------------------------------------- watch


class _Procs(dict):
    """Fake worker processes that each log one answer per check, then end."""

    def __init__(self, run_dir, answers):
        todo = {k: list(qids) for k, qids in answers.items()}

        def alive(key):
            if not todo[key]:
                return False
            _log(run_dir / f"{key}.jsonl", _ok(todo[key].pop(0)))
            return True

        super().__init__({k: types.SimpleNamespace(is_alive=lambda k=k: alive(k), exitcode=0) for k in answers})


def test_watch_prints_lines_when_not_a_terminal(run_dir, capsys):
    watch = pg.RunWatch(SPECS, run_dir, 6, console=_console())
    assert not watch.live
    watch.watch(_Procs(run_dir, {"m1": ["Equipo Uno|0"], "m2": []}), progress_s=0)
    out = capsys.readouterr().out
    assert out.startswith("[  0.0 min] m1 1/6 err 0 leak 0 | m2 0/6 err 0 leak 0 || groups fully evaluated 0/3")


def test_watch_says_how_to_resume_after_ctrl_c(run_dir, capsys):
    def interrupted():
        raise KeyboardInterrupt

    watch = pg.RunWatch(SPECS, run_dir, 6, console=_console())
    with pytest.raises(KeyboardInterrupt):
        watch.watch({"m1": types.SimpleNamespace(is_alive=interrupted)}, progress_s=0)
    assert "run the same command again to resume" in capsys.readouterr().out


def test_watch_shows_a_live_display_in_a_terminal(run_dir, monkeypatch):
    monkeypatch.setattr(pg, "REFRESH_S", 0)
    console = _console(interactive=True)
    watch = pg.RunWatch(SPECS, run_dir, 6, "team", console=console)
    assert watch.live
    (run_dir / "m1.worker.log").touch()
    (run_dir / "m2.worker.log").write_text("a warning\n")
    watch.watch(_Procs(run_dir, {"m1": ["Equipo Uno|0", "Equipo Uno|1"], "m2": ["Equipo Uno|0", "Equipo Uno|1"]}), 60)
    final = console.export_text().split("╭")[-1]  # the last frame
    assert "4/12" in final and "done" in final and "running" not in final and "1/3" in final and "/s" not in final
    assert not (run_dir / "m1.worker.log").exists() and (run_dir / "m2.worker.log").exists()  # empty logs go


def test_worker_output_goes_to_its_log_with_the_live_display(run_dir):
    watch = pg.RunWatch(SPECS, run_dir, 6, console=_console(interactive=True))
    stdout = os.fstat(1).st_ino
    proc = mp.get_context("spawn").Process(target=print, args=("a warning from the worker",))
    with watch.worker_output("m1"):
        proc.start()
    assert os.fstat(1).st_ino == stdout  # the parent's own output is back
    proc.join(timeout=60)
    assert (run_dir / "m1.worker.log").read_text().strip() == "a warning from the worker"
    plain = pg.RunWatch(SPECS, run_dir, 6, console=_console())
    with plain.worker_output("m2"):  # without the live display the worker keeps the parent's output
        assert os.fstat(1).st_ino == stdout
    assert not (run_dir / "m2.worker.log").exists()


def test_watch_ends_soon_after_the_workers(run_dir, monkeypatch, capsys):
    import time

    monkeypatch.setattr(pg, "REFRESH_S", 0.01)
    watch = pg.RunWatch(SPECS, run_dir, 6, console=_console())
    t0 = time.time()
    watch.watch(_Procs(run_dir, {"m1": ["Equipo Uno|0"], "m2": []}), progress_s=60)
    assert time.time() - t0 < 5  # not a whole --progress_s after the last worker ended
    assert capsys.readouterr().out.splitlines()[-1].split("] ")[1].startswith("m1 1/6")
