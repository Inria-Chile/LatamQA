"""Live progress of a panel run (`panel run`, `datathon run`): one bar per model, and one per group of items (the
teams, for the datathon).

The workers append one JSON record per request to ``<model>.jsonl``. `RunWatch` tails those logs, reading only the
bytes added since its last look, and counts answers by model and by group with the rule of `panel.latest_records` (a
200 supersedes any error before or after it). In a terminal it shows a `rich` live display refreshed every second and
sends the workers' own output (warnings, tracebacks) to ``<model>.worker.log``, so that it cannot break the display;
otherwise (a file, ``tee``, CI) it prints the plain progress line every ``progress_s`` seconds.
"""

import contextlib
import datetime as dt
import json
import math
import os
import re
import sys
import time
from collections import Counter, deque
from pathlib import Path

from latamqa import panel as pn

RATE_WINDOW_S = 30.0  # rates and ETAs use the answers of the last 30 s
REFRESH_S = 1.0
NAME_W = 18  # widest group name shown
MIN_BAR_W, MAX_BAR_W = 8, 20  # width of a group's bar
NOTES = (("err", "error", "yellow"), ("leak", "leak", "magenta"), ("refused", "refused", "yellow"))  # shown when > 0


class LogTail:
    """The records appended to a JSONL log since the last `read`; a line still being written waits for the next."""

    def __init__(self, path: Path):
        self.path, self.offset, self.partial = path, 0, b""

    def read(self) -> list[dict]:
        try:
            with open(self.path, "rb") as f:
                if os.fstat(f.fileno()).st_size < self.offset:  # rewritten from scratch: start over
                    self.offset, self.partial = 0, b""
                f.seek(self.offset)
                data = f.read()
        except FileNotFoundError:
            return []
        self.offset += len(data)
        *lines, self.partial = (self.partial + data).split(b"\n")
        records = []
        for line in lines:
            with contextlib.suppress(ValueError):  # a line cut short by a killed worker
                records.append(json.loads(line))
        return records


def natural_key(name: str) -> list:
    """Sort key that puts "Team 9" before "Team 10"."""
    return [int(t) if t.isdigit() else t.casefold() for t in re.split(r"(\d+)", name)]


def short_names(names: list[str]) -> dict[str, str]:
    """Drop the leading words that every name shares ("Equipo Chile 3" -> "Chile 3")."""
    prefix = os.path.commonprefix(names) if len(names) > 1 else ""
    prefix = prefix[: prefix.rfind(" ") + 1]
    return {n: n[len(prefix) :] or n for n in names}


def _hms(seconds: float | None) -> str:
    return "—" if seconds is None else str(dt.timedelta(seconds=round(seconds)))


def _pct(done: int, total: int) -> str:
    """Rounded down, so that 100% means done."""
    return f"{100 * done // total}%" if total else "—"


def _kind(rec: dict, leak=pn.is_leak) -> str:
    """ok / leak (answered; ``leak`` is the stop rule's test), refused (for billing) / error (pending)."""
    if rec.get("status") == 200:
        return "leak" if leak(rec) else "ok"
    return "refused" if pn.is_billing_refusal(rec) else "error"


class RunWatch:
    """Progress of the models of one run directory, by model and by group of items.

    Create it before starting the workers: it counts the answers already in the logs (a resumed run) and notes the
    ``<model>.ABORT`` files of earlier runs, which a worker removes or rewrites when it ends. Leaks are counted as the
    workers' stop rule counts them (`run_model`'s ``strict_leaks``), so a refusal the rule ignores is not shown as one.
    """

    def __init__(
        self,
        specs: list[dict],
        out_dir: Path,
        n_items: int,
        group_label: str = "group",
        console=None,
        strict_leaks: bool = True,
    ):
        from rich.console import Console

        self.out_dir, self.n_items, self.group_label = Path(out_dir), n_items, group_label
        self.leak = pn.is_leak if strict_leaks else pn.is_reasoning_leak
        self.keys = [s["key"] for s in specs]
        items_file = self.out_dir / "items.json"
        items = json.loads(items_file.read_text(encoding="utf-8")) if items_file.exists() else []
        self.group_of = {it["qid"]: it["group"] for it in items}  # also leaves out the cells of edited questions
        self.group_size = Counter(self.group_of.values())
        self.groups = sorted(self.group_size, key=natural_key)
        self.short = short_names(self.groups)
        self.tails = {k: LogTail(self.out_dir / f"{k}.jsonl") for k in self.keys}
        self.latest: dict[str, dict[str, dict]] = {k: {} for k in self.keys}
        self.counts = {k: Counter() for k in self.keys}
        self.group_done = Counter()  # answered cells of each group, summed over the models
        self.earlier_abort = {k: self.abort_stamp(k) for k in self.keys}
        self.console = console or Console()
        self.live = self.console.is_interactive
        self.start = time.time()
        self.read_logs()
        self.samples = {k: deque([(self.start, self.answered(k))]) for k in self.keys}

    # ------------------------------------------------------------------------------------------------------ state

    def read_logs(self) -> None:
        for key, tail in self.tails.items():
            latest, counts = self.latest[key], self.counts[key]
            for rec in tail.read():
                qid = rec.get("qid")
                if self.group_of and qid not in self.group_of:
                    continue
                old = latest.get(qid)
                was_answered = old is not None and old.get("status") == 200
                if was_answered and rec.get("status") != 200:
                    continue  # a 200 supersedes any error before or after it
                latest[qid] = rec
                if old is not None:
                    counts[_kind(old, self.leak)] -= 1
                counts[_kind(rec, self.leak)] += 1
                if rec.get("status") == 200 and not was_answered and self.group_of:
                    self.group_done[self.group_of[qid]] += 1

    def update(self) -> None:
        """Read what the workers logged since the last call."""
        self.read_logs()
        now = time.time()
        for key, samples in self.samples.items():
            samples.append((now, self.answered(key)))
            while len(samples) > 2 and samples[1][0] <= now - RATE_WINDOW_S:
                samples.popleft()

    def answered(self, key: str) -> int:
        return self.counts[key]["ok"] + self.counts[key]["leak"]

    def rate(self, key: str) -> float:
        """Answers per second over the last `RATE_WINDOW_S`."""
        (t0, a0), (t1, a1) = self.samples[key][0], self.samples[key][-1]
        return (a1 - a0) / (t1 - t0) if t1 > t0 else 0.0

    def abort_stamp(self, key: str) -> int | None:
        try:
            return (self.out_dir / f"{key}.ABORT").stat().st_mtime_ns
        except FileNotFoundError:
            return None

    def abort_reason(self, key: str) -> str | None:
        """Why the model stopped in this run; None if it did not (an ABORT file left by an earlier run does not count)."""
        if self.abort_stamp(key) in (None, self.earlier_abort[key]):
            return None
        with contextlib.suppress(FileNotFoundError):
            return (self.out_dir / f"{key}.ABORT").read_text(encoding="utf-8").strip()
        return None

    def status(self, key: str, proc) -> tuple[str, str, str]:
        """(state, style, detail) of a model: running, done, stopped (with the reason) or exited (a crash)."""
        reason = self.abort_reason(key)
        if reason:
            return "stopped", "red", reason.removeprefix("stopped: ")
        if proc is None or proc.is_alive():
            return "running", "cyan", ""
        code = getattr(proc, "exitcode", 0)
        if code:
            return "exited", "red", f"exit code {code}" + (f", see {key}.worker.log" if self.live else "")
        return "done", "green", ""

    def group_complete(self, group: str) -> bool:
        return self.group_done[group] >= self.group_size[group] * len(self.keys)

    # ---------------------------------------------------------------------------------------------------- output

    def line(self) -> str:
        """The plain progress line, for output that is not a terminal."""
        parts = []
        for k in self.keys:
            c = self.counts[k]
            aborted = " ABORTED" if self.abort_reason(k) else ""
            parts.append(f"{k} {self.answered(k)}/{self.n_items} err {c['refused'] + c['error']} leak {c['leak']}{aborted}")
        line = f"[{(time.time() - self.start) / 60:5.1f} min] " + " | ".join(parts)
        if len(self.groups) > 1:
            complete = sum(self.group_complete(g) for g in self.groups)
            line += f" || {self.group_label}s fully evaluated {complete}/{len(self.groups)}"
        return line

    def render(self, procs: dict | None = None):
        """The live display: a bar per model and for all of them, then the groups if there are several."""
        from rich.console import Group
        from rich.panel import Panel
        from rich.progress_bar import ProgressBar
        from rich.table import Table
        from rich.text import Text

        procs = procs or {}
        total = self.n_items * len(self.keys)
        models = Table.grid(padding=(0, 1), expand=True)
        models.add_column(no_wrap=True, min_width=max(len(k) for k in [*self.keys, "all models"]))
        models.add_column(ratio=1, min_width=10)  # the bars take the width left
        models.add_column(justify="right", no_wrap=True, min_width=len(f"{total:,}/{total:,}"))
        models.add_column(justify="right", no_wrap=True, min_width=4)  # %
        models.add_column(justify="right", no_wrap=True, min_width=6)  # rate
        models.add_column(justify="right", no_wrap=True, min_width=7)  # ETA
        models.add_column(no_wrap=True, overflow="ellipsis")  # errors, leaks and billing refusals, when there are any
        models.add_column(no_wrap=True)  # state
        etas, total_done, stopped = [], 0, []
        for k in self.keys:
            c, done, rate = self.counts[k], self.answered(k), self.rate(k)
            total_done += done
            state, style, detail = self.status(k, procs.get(k))
            if detail:
                stopped.append(Text(f"{k} {state}: {detail}", style=style, overflow="ellipsis", no_wrap=True))
            eta = None
            if state == "running":
                eta = (self.n_items - done) / rate if rate > 0 else None
                etas.append(eta)
            notes = [(f"{name} {c[kind]}", hue) for name, kind, hue in NOTES if c[kind]]
            running = state == "running"
            models.add_row(
                Text(k, style="bold"),
                ProgressBar(total=max(self.n_items, 1), completed=done),
                f"{done:,}/{self.n_items:,}",
                _pct(done, self.n_items),
                f"{rate:.1f}/s" if running else "",
                _hms(eta) if running else "",
                Text(" · ").join(Text(text, style=hue) for text, hue in notes),
                Text(state, style=style),
            )
        models.add_row(
            Text("all models", style="bold"),
            ProgressBar(total=max(total, 1), completed=total_done),
            f"{total_done:,}/{total:,}",
            _pct(total_done, total),
            f"{sum(self.rate(k) for k in self.keys):.1f}/s" if etas else "",
            ("—" if None in etas else _hms(max(etas))) if etas else "",  # the slowest model sets the end
            "",
            "",
        )
        parts = [models, *stopped]
        width, height = self.console.size
        if len(self.groups) > 1:
            complete = sum(self.group_complete(g) for g in self.groups)
            summary = Table.grid(padding=(0, 1))
            summary.add_column(no_wrap=True)
            summary.add_column(width=24)
            summary.add_column(no_wrap=True)
            summary.add_row(
                Text(f"{self.group_label.capitalize()}s fully evaluated", style="bold"),
                ProgressBar(total=len(self.groups), completed=complete),
                f"{complete}/{len(self.groups)}",
            )
            parts += ["", summary]
            grid = self.group_grid(width - 4, height - len(self.keys) - len(stopped) - 6)  # rows left in the frame
            if grid is not None:
                parts.append(grid)
        title = f"{len(self.keys)} models x {self.n_items:,} answers · {_hms(time.time() - self.start)}"
        return Panel(Group(*parts), title=title, title_align="left", border_style="dim")

    def group_grid(self, width: int, max_rows: int):
        """Every group's bar, in columns read top to bottom; fully evaluated groups make room first when space runs out."""
        from rich.console import Group
        from rich.progress_bar import ProgressBar
        from rich.table import Table
        from rich.text import Text

        if max_rows < 2:
            return None
        name_w = min(NAME_W, max(len(self.short[g]) for g in self.groups))
        fixed = name_w + 4 + 2  # name, %, and the two gaps of a column
        ncols = max(1, (width + 3) // (fixed + MIN_BAR_W + 3))  # 3 columns of space between two columns
        bar_w = min(MAX_BAR_W, (width + 3) // ncols - fixed - 3)
        shown, note = list(self.groups), ""
        if math.ceil(len(shown) / ncols) > max_rows:
            shown = [g for g in self.groups if not self.group_complete(g)]
            note = f"+{len(self.groups) - len(shown)} fully evaluated"
            if math.ceil(len(shown) / ncols) > max_rows - 1:
                keep = (max_rows - 1) * ncols
                note += f" · {len(shown) - keep} more not shown"
                shown = shown[:keep]
        nrows = max(1, math.ceil(len(shown) / ncols))
        grid = Table.grid(padding=(0, 1))
        for c in range(ncols):
            if c:
                grid.add_column(width=1)
            grid.add_column(width=name_w, no_wrap=True, overflow="ellipsis")
            grid.add_column(width=bar_w)
            grid.add_column(width=4, justify="right", no_wrap=True)
        for r in range(nrows):
            row = []
            for c in range(ncols):
                if c:
                    row.append("")
                i = c * nrows + r
                if i >= len(shown):
                    row += ["", "", ""]
                    continue
                g = shown[i]
                total, done = self.group_size[g] * len(self.keys), self.group_done[g]
                if self.group_complete(g):
                    pct = Text("✓", style="green")
                else:
                    pct = Text(_pct(done, total), style="" if done else "dim")
                row += [
                    Text(self.short[g], style="" if done else "dim"),
                    ProgressBar(total=total, completed=done, width=bar_w),
                    pct,
                ]
            grid.add_row(*row)
        return Group(grid, Text(note, style="dim")) if note else grid

    # ----------------------------------------------------------------------------------------------------- watch

    @contextlib.contextmanager
    def worker_output(self, key: str):
        """With the live display on, send the output of a worker started in this block to ``<key>.worker.log``: a
        spawned process inherits the parent's stdout and stderr as they are when it starts."""
        if not self.live:
            yield
            return
        sys.stdout.flush()
        sys.stderr.flush()
        saved = os.dup(1), os.dup(2)
        try:
            with open(self.out_dir / f"{key}.worker.log", "ab") as f:
                os.dup2(f.fileno(), 1)
                os.dup2(f.fileno(), 2)
                yield
        finally:
            os.dup2(saved[0], 1)
            os.dup2(saved[1], 2)
            os.close(saved[0])
            os.close(saved[1])

    def watch(self, procs: dict, progress_s: float) -> None:
        """Show progress until every worker process has ended."""
        try:
            if self.live:
                self.watch_live(procs)
            else:
                while any(p.is_alive() for p in procs.values()):
                    time.sleep(progress_s)
                    self.update()
                    print(self.line(), flush=True)
        except KeyboardInterrupt:
            print("Interrupted: the answers so far are in the logs; run the same command again to resume.", flush=True)
            raise

    def watch_live(self, procs: dict) -> None:
        from rich.live import Live

        with Live(self.render(procs), console=self.console, auto_refresh=False) as live:
            while any(p.is_alive() for p in procs.values()):
                time.sleep(REFRESH_S)
                self.update()
                live.update(self.render(procs), refresh=True)
            self.update()
            live.update(self.render(procs), refresh=True)
        for key in self.keys:  # keep only the logs of workers that wrote something
            log = self.out_dir / f"{key}.worker.log"
            if log.exists() and not log.stat().st_size:
                log.unlink()
