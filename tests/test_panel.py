"""Tests for the panel runner (`latamqa.panel`): panel config, question items, request spec, scoring, the per-model
worker (guards, resume, billing stop) and the offline end-to-end command. No network access is needed."""

import copy
import csv
import json
import types

import pytest

from latamqa import panel as pn
from latamqa.mcq_core import V2_MAX_TOKENS, V2_PROMPT_TEMPLATE, V2_SYSTEM_MESSAGE

ROWS = [
    {
        "article_id": i,
        "question": f"Question {i}?",
        "answer": f"right {i}",
        "distractor1": f"w1 {i}",
        "distractor2": f"w2 {i}",
        "distractor3": f"w3 {i}",
        "lang": "es-la" if i % 2 else "pt-br",
    }
    for i in range(40)
]


@pytest.fixture
def p6():
    return pn.load_panel("p6")


@pytest.fixture(autouse=True)
def _reset_billing_abort():
    pn._BILLING_ABORT.clear()
    yield
    pn._BILLING_ABORT.clear()


# ---------------------------------------------------------------------------------------------------------- panel


def test_p6_panel_is_valid_and_pins_the_verified_spec(p6):
    assert p6["name"] == "P6" and p6["protocol"] == "v2" and p6["bill_to"] == "inria-chile"
    by_key = {m["key"]: m for m in p6["models"]}
    assert list(by_key) == ["qwen3-4b", "llama-3.1-8b", "qwen3.5-9b", "qwen3.5-27b", "qwen2.5-72b", "qwen3.5-397b", "kimi-k2"]
    # Reasoning switches verified in the pilot.
    for key in ("qwen3.5-9b", "qwen3.5-27b", "qwen3.5-397b"):
        assert by_key[key]["extra_body"] == {"reasoning_effort": "none"}
        assert by_key[key]["api_base"] == "https://router.huggingface.co/deepinfra/v1/openai/chat/completions"
    assert by_key["kimi-k2"]["extra_body"] == {"thinking": {"type": "disabled"}}
    for key in ("qwen3-4b", "llama-3.1-8b", "qwen2.5-72b"):
        assert by_key[key]["extra_body"] == {}
    # Qwen2.5-72B replaced Qwen3.5-122B, which is only an alternative now.
    assert by_key["qwen2.5-72b"]["provider"] == "novita"
    assert "qwen3.5-122b" not in by_key and "qwen3.5-122b" in pn.all_models(p6)


def test_validate_panel_reports_problems(p6):
    raw = {k: v for k, v in copy.deepcopy(p6).items() if k != "path"}
    assert pn.validate_panel(raw) == []
    bad = copy.deepcopy(raw)
    bad["models"][0]["tokens"] = 5
    del bad["models"][1]["price"]
    bad["protocol"] = "v1"
    bad["alternatives"][0]["key"] = "qwen3-4b"
    errors = pn.validate_panel(bad)
    assert any("Additional properties" in e for e in errors)
    assert any("'price' is a required property" in e for e in errors)
    assert any("protocol" in e for e in errors)
    dup = copy.deepcopy(raw)
    dup["alternatives"][0]["key"] = "qwen3-4b"
    assert pn.validate_panel(dup) == ["duplicate model key «qwen3-4b»"]


def test_load_panel_errors(tmp_path):
    with pytest.raises(pn.PanelError, match="not found"):
        pn.load_panel("no-such-panel")
    path = tmp_path / "bad.yaml"
    path.write_text("name: X\nprotocol: v2\nbill_to: org\nmodels: []\n")
    with pytest.raises(pn.PanelError, match="invalid panel"):
        pn.load_panel(path)


def test_select_models(p6):
    assert [m["key"] for m in pn.select_models(p6)] == [m["key"] for m in p6["models"]]
    assert [m["key"] for m in pn.select_models(p6, "kimi-k2,qwen3-4b")] == ["kimi-k2", "qwen3-4b"]
    swapped = [m["key"] for m in pn.select_models(p6, swaps=["qwen2.5-72b=qwen2.5-72b-di"])]
    assert "qwen2.5-72b-di" in swapped and "qwen2.5-72b" not in swapped
    with pytest.raises(pn.PanelError, match="unknown model"):
        pn.select_models(p6, "gpt-9")
    with pytest.raises(pn.PanelError, match="--swap"):
        pn.select_models(p6, swaps=["qwen3.5-122b=qwen2.5-72b-di"])
    with pytest.raises(pn.PanelError, match="duplicates"):
        pn.select_models(p6, swaps=["qwen2.5-72b=qwen3-4b"])


def test_litellm_model_routes(p6):
    by_key = pn.all_models(p6)
    assert pn.litellm_model(by_key["qwen2.5-72b"]) == "huggingface/novita/Qwen/Qwen2.5-72B-Instruct"
    assert pn.litellm_model(by_key["qwen3.5-9b"]) == "huggingface/Qwen/Qwen3.5-9B"  # route set by api_base


# ---------------------------------------------------------------------------------------------------------- questions


def test_build_items(p6):
    items = pn.build_items(ROWS, order=1, col_group="lang")
    assert len(items) == 40
    first = items[0]
    assert first["qid"].startswith("0:") and first["group"] == "pt-br"
    assert first["prompt"].startswith("Answer the following") and first["prompt"] == V2_PROMPT_TEMPLATE.replace(
        "{question}", "Question 0?"
    ).replace("{option_a}", first["options"][0]).replace("{option_b}", first["options"][1]).replace(
        "{option_c}", first["options"][2]
    ).replace("{option_d}", first["options"][3])
    assert first["options"]["ABCD".index(first["correct"])] == "right 0"
    # identical on every call; a different order moves the options; duplicates are dropped
    assert pn.build_items(ROWS, order=1, col_group="lang") == items
    assert pn.build_items(ROWS, order=2)[0]["options"] != first["options"]
    assert len(pn.build_items(ROWS + ROWS[:5], order=1)) == 40
    assert {it["group"] for it in pn.build_items(ROWS, order=1)} == {"all"}


def test_build_items_limit_is_a_stable_subset():
    a = pn.build_items(ROWS, order=1, limit=10)
    assert len(a) == 10
    assert [it["qid"] for it in a] == [it["qid"] for it in pn.build_items(ROWS, order=3, limit=10)]
    assert [it["qid"] for it in a] != [it["qid"] for it in pn.build_items(ROWS, order=1, limit=10, seed=7)]


def test_build_items_missing_columns():
    with pytest.raises(pn.PanelError, match="lacks columns"):
        pn.build_items([{"q": "x"}], order=1)
    renamed = [{"id": r["article_id"], "q": r["question"], "a": r["answer"], "x": r["distractor1"],
                "y": r["distractor2"], "z": r["distractor3"]} for r in ROWS]  # fmt: skip
    items = pn.build_items(renamed, order=1, col_id="id", col_question="q", col_answer="a", col_distractors=("x", "y", "z"))
    assert len(items) == 40


def test_read_questions_formats(tmp_path):
    csv_path = tmp_path / "q.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(ROWS[0]))
        w.writeheader()
        w.writerows(ROWS)
    jsonl_path = tmp_path / "q.jsonl"
    jsonl_path.write_text("\n".join(json.dumps(r) for r in ROWS) + "\n", encoding="utf-8")
    json_path = tmp_path / "q.json"
    json_path.write_text(json.dumps({"data": ROWS}), encoding="utf-8")
    for path in (csv_path, jsonl_path, json_path):
        rows = pn.read_questions(str(path))
        assert len(rows) == 40 and rows[3]["question"] == "Question 3?"
    with pytest.raises(pn.PanelError, match="unsupported"):
        pn.read_questions(str(tmp_path / "q.xlsx"))


# ---------------------------------------------------------------------------------------------------------- requests


def _response(content="B", finish="stop", completion_tokens=1, reasoning=None):
    message = types.SimpleNamespace(content=content, reasoning_content=reasoning, provider_specific_fields=None)
    return types.SimpleNamespace(
        choices=[types.SimpleNamespace(message=message, finish_reason=finish)],
        usage=types.SimpleNamespace(prompt_tokens=250, completion_tokens=completion_tokens, completion_tokens_details=None),
        model="provider/model-id",
        _hidden_params={"additional_headers": {"llm_provider-x-inference-provider": "novita"}},
    )


class _ApiError(Exception):
    def __init__(self, status_code, msg="err"):
        super().__init__(msg)
        self.status_code = status_code


class FakeLiteLLM:
    """Stands in for the litellm module: `script` maps a call number to an exception or a response."""

    def __init__(self, script=None, default=None):
        self.calls, self.script, self.default = [], script or {}, default or _response()

    def completion(self, **kwargs):
        self.calls.append(kwargs)
        out = self.script.get(len(self.calls), self.default)
        if callable(out):
            out = out(kwargs)
        if isinstance(out, Exception):
            raise out
        return out


ITEM = dict(qid="q1", group="es-la", correct="B", prompt="Q?")


@pytest.fixture
def no_sleep(monkeypatch):
    monkeypatch.setattr(pn.time, "sleep", lambda s: None)


def test_ask_builds_the_v2_request(p6):
    fake = FakeLiteLLM()
    spec = pn.all_models(p6)["qwen3.5-9b"]
    rec = pn.ask(fake, spec, ITEM, "inria-chile", "tok")
    kw = fake.calls[0]
    assert kw["model"] == "huggingface/Qwen/Qwen3.5-9B" and kw["api_base"] == spec["api_base"]
    assert kw["messages"] == [{"role": "system", "content": V2_SYSTEM_MESSAGE}, {"role": "user", "content": "Q?"}]
    assert kw["max_tokens"] == V2_MAX_TOKENS and kw["temperature"] == 0.0 and kw["num_retries"] == 0
    assert kw["extra_headers"] == {"X-HF-Bill-To": "inria-chile"} and kw["extra_body"] == {"reasoning_effort": "none"}
    assert rec["status"] == 200 and rec["content"] == "B" and rec["provider_hdr"] == "novita" and rec["attempts"] == 1
    novita = FakeLiteLLM()
    pn.ask(novita, pn.all_models(p6)["qwen2.5-72b"], ITEM, "inria-chile", "tok")
    assert novita.calls[0]["model"] == "huggingface/novita/Qwen/Qwen2.5-72B-Instruct"
    assert "api_base" not in novita.calls[0] and "extra_body" not in novita.calls[0]


def test_ask_retries_transient_errors(p6, no_sleep):
    fake = FakeLiteLLM({1: _ApiError(429), 2: _ApiError(503)})
    rec = pn.ask(fake, p6["models"][0], ITEM, "org", "tok")
    assert rec["status"] == 200 and rec["attempts"] == 3
    fake = FakeLiteLLM({1: _ApiError(400, "This model is cold, retry later")})
    assert pn.ask(fake, p6["models"][0], ITEM, "org", "tok")["attempts"] == 2


def test_ask_records_permanent_errors(p6, no_sleep):
    rec = pn.ask(FakeLiteLLM(default=_ApiError(400, "unknown field")), p6["models"][0], ITEM, "org", "tok")
    assert rec["status"] == 400 and rec["attempts"] == 1 and "unknown field" in rec["error_msg"]
    rec = pn.ask(FakeLiteLLM(default=_ApiError(500)), p6["models"][0], ITEM, "org", "tok")
    assert rec["status"] == 500 and rec["attempts"] == pn.MAX_ATTEMPTS
    assert not pn._BILLING_ABORT.is_set()
    rec = pn.ask(FakeLiteLLM(default=_ApiError(402)), p6["models"][0], ITEM, "org", "tok")
    assert rec["status"] == 402 and rec["attempts"] == 1 and pn._BILLING_ABORT.is_set()


# ---------------------------------------------------------------------------------------------------------- scoring


@pytest.mark.parametrize(
    ("rec", "expected"),
    [
        (None, (None, "missing")),
        ({"status": 503}, (None, "error")),
        ({"status": 200, "content": "C", "finish": "stop", "completion_tokens": 1}, ("C", "only")),
        ({"status": 200, "content": "B) Lima", "finish": "stop", "completion_tokens": 4}, ("B", "lead")),
        ({"status": 200, "content": "B) Lima, la capital del", "finish": "length", "completion_tokens": 16}, ("B", "lead")),
        ({"status": 200, "content": "La respuesta es D", "finish": "stop", "completion_tokens": 5}, ("D", "cue")),
        ({"status": 200, "content": "No sé", "finish": "stop", "completion_tokens": 3}, (None, "none")),
        ({"status": 200, "content": "B", "reasoning_len": 900}, (None, "leak")),
        ({"status": 200, "content": "B", "reasoning_tokens": 12}, (None, "leak")),
        ({"status": 200, "content": "<think>x</think>B"}, (None, "leak")),
        ({"status": 200, "content": "B", "completion_tokens": 3718}, (None, "leak")),
        ({"status": 200, "content": "The user wants me to answer", "finish": "length"}, (None, "leak")),
    ],
)
def test_score(rec, expected):
    assert pn.score(rec) == expected


def test_latest_records_prefers_a_200(tmp_path):
    path = tmp_path / "m.jsonl"
    lines = [{"qid": "a", "status": 503}, {"qid": "a", "status": 200}, {"qid": "a", "status": 429}, {"qid": "b", "status": 500}]
    path.write_text("".join(json.dumps(x) + "\n" for x in lines))
    last = pn.latest_records(path)
    assert last["a"]["status"] == 200 and last["b"]["status"] == 500


def test_estimate_cost(p6):
    spec = pn.all_models(p6)["qwen2.5-72b"]
    cost = pn.estimate_cost([spec], {"qwen2.5-72b": 20_000})
    assert cost == pytest.approx(20_000 * (300 * 0.38 + 16 * 0.40) / 1e6)


# ---------------------------------------------------------------------------------------------------------- worker


def _run_worker(tmp_path, monkeypatch, spec, fake, n=20, **kw):
    items = pn.build_items(ROWS[:n], order=1)
    (tmp_path / "items.json").write_text(json.dumps(items))
    monkeypatch.setattr(pn, "setup_litellm", lambda dry_run, specs: fake)
    monkeypatch.setenv("HF_TOKEN", "tok")
    args = dict(rate=10_000, cap=4, bill_to="org", dry_run=False, max_leaks=3, max_error_rate=0.01, retry_rounds=2)
    args.update(kw)
    pn.run_model(spec, str(tmp_path / "items.json"), str(tmp_path), **args)
    return items


def test_run_model_answers_everything_and_resumes(tmp_path, monkeypatch, p6, no_sleep):
    spec = p6["models"][0]
    fake = FakeLiteLLM()
    items = _run_worker(tmp_path, monkeypatch, spec, fake)
    recs = pn.latest_records(tmp_path / f"{spec['key']}.jsonl")
    assert len(fake.calls) == 20 and all(r["status"] == 200 for r in recs.values()) and len(recs) == len(items)
    assert not (tmp_path / f"{spec['key']}.ABORT").exists() and (tmp_path / f"{spec['key']}.timing.json").exists()
    again = FakeLiteLLM()
    _run_worker(tmp_path, monkeypatch, spec, again)
    assert again.calls == []  # resume: nothing left to send


def test_run_model_retries_failed_questions_in_later_passes(tmp_path, monkeypatch, p6, no_sleep):
    failing = {"n": 0}

    def flaky(kwargs):
        failing["n"] += 1
        return _ApiError(400, "bad gateway body") if failing["n"] <= 3 else _response()

    spec = p6["models"][0]
    _run_worker(tmp_path, monkeypatch, spec, FakeLiteLLM(default=flaky))
    recs = pn.latest_records(tmp_path / f"{spec['key']}.jsonl")
    assert len(recs) == 20 and all(r["status"] == 200 for r in recs.values())


def test_run_model_stops_on_reasoning_leaks(tmp_path, monkeypatch, p6, no_sleep):
    spec = p6["models"][0]
    prose = _response("Okay, so the question asks which option", finish="length", completion_tokens=16)
    fake = FakeLiteLLM(default=prose)
    _run_worker(tmp_path, monkeypatch, spec, fake, n=40, cap=1)
    reason = (tmp_path / f"{spec['key']}.ABORT").read_text()
    assert "reasoning leaks" in reason and len(fake.calls) < 40
    assert not (tmp_path / "STOP").exists()  # a leak stops this model only


def test_run_model_billing_error_stops_the_whole_run(tmp_path, monkeypatch, p6, no_sleep):
    spec = p6["models"][0]
    limit = _ApiError(403, "You have exceeded your monthly spending limit for Inference Providers")
    fake = FakeLiteLLM(default=limit)
    _run_worker(tmp_path, monkeypatch, spec, fake, cap=1)
    assert "spending limit" in (tmp_path / f"{spec['key']}.ABORT").read_text()
    assert (tmp_path / "STOP").exists() and len(fake.calls) < 20


def test_other_models_stop_when_the_run_is_stopped(tmp_path, monkeypatch, p6, no_sleep):
    (tmp_path / "STOP").write_text("kimi-k2: HTTP 402\n")
    spec = p6["models"][0]
    fake = FakeLiteLLM()
    _run_worker(tmp_path, monkeypatch, spec, fake)
    assert fake.calls == [] and "stopped: kimi-k2" in (tmp_path / f"{spec['key']}.ABORT").read_text()


def test_report_counts_errors_as_wrong(tmp_path, monkeypatch, p6, no_sleep):
    spec = p6["models"][0]
    items = _run_worker(tmp_path, monkeypatch, spec, FakeLiteLLM(default=_ApiError(400, "no")), retry_rounds=0)
    (s,) = pn.report([spec], items, tmp_path, "unit", 1, "P6", dry_run=False)
    assert s["accuracy"] == 0 and s["coverage"] == 0 and s["errors"] == 20 and s["protocol"] == "v2"
    assert "**Publishable**: NO for qwen3-4b" in (tmp_path / "report.md").read_text()
    summary = (tmp_path / "mcq_eval_summary_unit-all_run1_qwen3-4b.txt").read_text()
    assert "protocol: v2" in summary and "accuracy: 0.0" in summary


# ---------------------------------------------------------------------------------------------------------- end to end


def test_panel_command_dry_run_end_to_end(tmp_path, monkeypatch):
    questions = tmp_path / "set.jsonl"
    questions.write_text("\n".join(json.dumps(r) for r in ROWS) + "\n", encoding="utf-8")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_API_KEY", raising=False)
    base = ["run", "--questions", str(questions), "--results_dir", str(tmp_path / "out"), "--dry_run", "--rate", "500"]
    base += ["--progress_s", "0.2", "--models", "qwen3-4b,kimi-k2", "--col_group", "lang"]
    pn.main(base)
    out = tmp_path / "out" / "p6" / "set" / "run1"
    report = json.loads((out / "report.json").read_text())
    assert [s["key"] for s in report] == ["qwen3-4b", "kimi-k2"]
    # Kimi-K2 reasons in the fake unless its switch is sent: no leaks means the switch went out.
    assert all(s["coverage"] == 1 and s["leaks"] == 0 and not s["aborted"] for s in report)
    assert set(report[0]["groups"]) == {"es-la", "pt-br"}
    meta = json.loads((out / "run_meta.json").read_text())
    assert meta["protocol"] == "v2" and meta["bill_to"] == "inria-chile" and meta["n"] == 40
    # a different question subset may not be mixed into the same run directory
    with pytest.raises(SystemExit):
        pn.main([*base, "--limit", "10"])
    # a paid run needs --yes
    with pytest.raises(SystemExit):
        pn.main(["run", "--questions", str(questions), "--results_dir", str(tmp_path / "paid")])


def test_check_needs_no_question_set(monkeypatch):
    called = []
    monkeypatch.setattr(pn, "stage_check", lambda args, panel, specs: called.append([s["key"] for s in specs]))
    pn.main(["check", "--swap", "qwen2.5-72b=qwen2.5-72b-di"])
    assert called and "qwen2.5-72b-di" in called[0]


def test_p6_small_is_the_three_smallest_p6_models(p6):
    small = pn.load_panel("p6-small")
    assert small["name"] == "P6-small" and small["protocol"] == p6["protocol"] and small["bill_to"] == p6["bill_to"]
    assert [m["key"] for m in small["models"]] == ["qwen3-4b", "llama-3.1-8b", "qwen3.5-9b"]
    by_key = {m["key"]: m for m in p6["models"]}
    for m in small["models"]:  # identical copies, so their answers are interchangeable with P6's
        assert m == by_key[m["key"]]


def test_is_reasoning_leak_ignores_refusals():
    refusal = {
        "status": 200,
        "content": "I'm not able to see the question. Can you",
        "finish": "length",
        "completion_tokens": 16,
    }
    assert pn.is_leak(refusal) and not pn.is_reasoning_leak(refusal)
    for strong in ({"reasoning_len": 40}, {"reasoning_tokens": 3}, {"content": "<think>x"}, {"completion_tokens": 900}):
        rec = {"status": 200, "content": "B", "finish": "stop", "completion_tokens": 1, **strong}
        assert pn.is_reasoning_leak(rec) and pn.is_leak(rec)
    assert not pn.is_reasoning_leak({"status": 503})


def test_lenient_leak_rule_lets_refusals_through(tmp_path, monkeypatch, p6, no_sleep):
    spec = p6["models"][1]
    refusal = _response("I'm not able to see the question. Can you please provide", finish="length", completion_tokens=16)
    fake = FakeLiteLLM(default=refusal)
    _run_worker(tmp_path, monkeypatch, spec, fake, n=40, cap=1, strict_leaks=False)
    assert len(fake.calls) == 40 and not (tmp_path / f"{spec['key']}.ABORT").exists()
    reasoning = _response("B", reasoning="Let me think about which option")
    fake = FakeLiteLLM(default=reasoning)
    (tmp_path / "x").mkdir()
    _run_worker(tmp_path / "x", monkeypatch, spec, fake, n=40, cap=1, strict_leaks=False)
    assert "reasoning leaks" in (tmp_path / "x" / f"{spec['key']}.ABORT").read_text() and len(fake.calls) < 40
