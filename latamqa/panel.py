#!/usr/bin/env python3
"""Run a panel of open-weight models on Hugging Face Inference Providers under protocol v2.

A panel (``latamqa/panels/<name>.yaml``) pins each model to one provider, with its route, reasoning switch, price and
in-flight cap. ``panel run`` evaluates every model on a question set at the same time, one process per model, at a
fixed request rate (13.3 req/s by default: 20,000 questions in 25 minutes). The request and scoring spec is protocol
v2 (see :mod:`latamqa.mcq_core`); the leaderboard path (``eval_mcq`` / ``model_eval``) keeps protocol v1.

Stages:
  check    free   LiteLLM version, token, billing org, each model's Hub mapping and gated access, DeepInfra retirements
  run      paid   evaluate the panel on a question set; re-running the same command resumes it
  report   free   rebuild a run's summaries and report from its logs

Usage:
  panel check
  panel run --questions new_set.parquet --run 1 --limit 2000 --yes   # dress rehearsal on a stable subset
  panel run --questions new_set.parquet --run 1 --yes                # full run 1; then --run 2 ... --run 5
  panel report --set_name new_set --run 1
Add --dry_run to run everything offline against a fake provider (no network, no cost).

Guards: a model stops after --max_leaks reasoning leaks, above --max_error_rate after 500 requests, after 50
consecutive errors, or on any HTTP 403; HTTP 402 (billing) and the spending-limit 403 stop every model. Errors get
--retry_rounds slower passes. A run is publishable when every model answers at least 99.5 % of the questions with no
leak and no stop.
"""

import argparse
import csv
import datetime as dt
import json
import math
import multiprocessing as mp
import os
import random
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator
from structlog import get_logger

from latamqa.mcq_core import (
    PROTOCOL_V2,
    V2_MAX_TOKENS,
    V2_OPTION_ORDERS,
    V2_PROMPT_TEMPLATE,
    V2_SYSTEM_MESSAGE,
    build_prompt,
    has_think_tag,
    parse_answer_v2,
    permuted_options,
    question_id,
)

logger = get_logger(__name__)

PANELS_DIR = Path(__file__).parent / "panels"
DEFAULT_PANEL = "p6"
DEFAULT_RESULTS_DIR = Path(__file__).parent.parent / "results" / "panel"
HUB = "https://huggingface.co"
# The panel was verified with this LiteLLM release; the two patches in `setup_litellm` target its internals.
TESTED_LITELLM = "1.103.1"
DEFAULT_RATE = 20_000 / 1_500  # req/s per model: 20,000 questions in 25 minutes (the target is 30)
TEMPERATURE = 0.0
TIMEOUT_S = 60
MAX_ATTEMPTS = 4
RETRYABLE = {408, 429, 500, 502, 503, 504, 599}
MIN_COVERAGE = 0.995
EST_PROMPT_TOKENS = 300  # upper bound for cost estimates (LatamQA prompts are ~260 tokens with the system message)

_MODEL_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["key", "hub_id", "provider", "price", "max_in_flight"],
    "properties": {
        "key": {"type": "string", "pattern": r"^[a-z0-9][a-z0-9._-]*$"},
        "hub_id": {"type": "string", "pattern": r"^[^/\s]+/[^/\s]+$"},
        "provider": {"type": "string", "minLength": 1},
        "api_base": {"type": "string", "pattern": r"^https://"},
        "size": {"type": "string"},
        "price": {
            "type": "object",
            "additionalProperties": False,
            "required": ["input", "output"],
            "properties": {"input": {"type": "number", "minimum": 0}, "output": {"type": "number", "minimum": 0}},
        },
        "max_in_flight": {"type": "integer", "minimum": 1},
        "extra_body": {"type": "object"},
        "note": {"type": "string"},
    },
}
PANEL_SCHEMA: dict = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "LatamQA model panel",
    "type": "object",
    "additionalProperties": False,
    "required": ["name", "protocol", "bill_to", "models"],
    "properties": {
        "name": {"type": "string", "minLength": 1},
        "description": {"type": "string"},
        "protocol": {"enum": [PROTOCOL_V2]},
        "bill_to": {"type": "string", "minLength": 1},
        "models": {"type": "array", "minItems": 1, "items": _MODEL_SCHEMA},
        "alternatives": {"type": "array", "items": _MODEL_SCHEMA},
    },
}
Draft202012Validator.check_schema(PANEL_SCHEMA)
_PANEL_VALIDATOR = Draft202012Validator(PANEL_SCHEMA)


class PanelError(Exception):
    """A panel file or command-line selection is invalid."""


# ---------------------------------------------------------------------------------------------------------- panel


def validate_panel(config: object) -> list[str]:
    """Return the schema violations of a panel config (empty when valid), including duplicate model keys."""
    errors = [
        f"{'.'.join(str(p) for p in e.path) or '<document>'}: {e.message}"
        for e in sorted(_PANEL_VALIDATOR.iter_errors(config), key=lambda e: list(e.path))
    ]
    if not errors:
        keys = [m["key"] for m in config["models"] + config.get("alternatives", [])]  # type: ignore[index]
        errors += [f"duplicate model key «{k}»" for k in sorted({k for k in keys if keys.count(k) > 1})]
    return errors


def load_panel(name_or_path: str | Path = DEFAULT_PANEL) -> dict:
    """Load and validate a panel by name (``latamqa/panels/<name>.yaml``) or path."""
    path = Path(name_or_path)
    if not path.suffix:
        path = PANELS_DIR / f"{name_or_path}.yaml"
    if not path.exists():
        raise PanelError(f"panel file not found: {path}")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    errors = validate_panel(config)
    if errors:
        raise PanelError(f"invalid panel {path.name}:\n" + "\n".join(f"  • {e}" for e in errors))
    for m in config["models"] + config.get("alternatives", []):
        m.setdefault("extra_body", {})
    config["path"] = str(path)
    return config


def all_models(panel: dict) -> dict[str, dict]:
    """Panel models and alternatives by key."""
    return {m["key"]: m for m in panel["models"] + panel.get("alternatives", [])}


def select_models(panel: dict, models: str | None = None, swaps: list[str] | None = None) -> list[dict]:
    """The models to run: the panel (or a comma-separated subset/list of keys), with ``old=new`` swaps applied."""
    known = all_models(panel)
    keys = models.split(",") if models else [m["key"] for m in panel["models"]]
    unknown = [k for k in keys if k not in known]
    if unknown:
        raise PanelError(f"unknown model key(s) {unknown}; known: {sorted(known)}")
    for swap in swaps or []:
        old, sep, new = swap.partition("=")
        if not sep or old not in keys or new not in known:
            raise PanelError(f"--swap {swap}: expected old=new with «old» in the selection and «new» a known model key")
        keys[keys.index(old)] = new
    if len(set(keys)) != len(keys):
        raise PanelError(f"model selection has duplicates: {keys}")
    return [known[k] for k in keys]


def litellm_model(spec: dict) -> str:
    """LiteLLM model string: provider-prefixed router route, or the bare Hub id when ``api_base`` sets the route."""
    return f"huggingface/{spec['hub_id']}" if spec.get("api_base") else f"huggingface/{spec['provider']}/{spec['hub_id']}"


# ---------------------------------------------------------------------------------------------------------- questions


def read_questions(source: str) -> list[dict]:
    """Rows from a local .parquet/.csv/.jsonl/.json file, or ``hf:org/name[:split]`` (a Hub dataset, default split
    ``train``; private sets need HF_TOKEN)."""
    if source.startswith("hf:"):
        from datasets import load_dataset

        name, _, split = source[3:].partition(":")
        return load_dataset(name, split=split or "train").to_list()
    path = Path(source)
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        return pq.read_table(path).to_pylist()
    if path.suffix == ".csv":
        with open(path, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    if path.suffix == ".jsonl":
        with open(path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    if path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else data.get("data") or data.get("rows") or []
    raise PanelError(f"unsupported question file: {source} (use .parquet, .csv, .jsonl, .json or hf:org/name[:split])")


def build_items(
    rows: list[dict],
    order: int,
    seed: int = 42,
    limit: int | None = None,
    col_id: str | None = "article_id",
    col_question: str = "question",
    col_answer: str = "answer",
    col_distractors: tuple[str, str, str] = ("distractor1", "distractor2", "distractor3"),
    col_group: str | None = None,
) -> list[dict]:
    """Questions with their option order ``order`` (1-5) and prompt. ``limit`` keeps a stable random subset.

    Duplicate questions (same row id and text) are kept once. Items are identical for every model and every re-run.
    """
    need = [col_question, col_answer, *col_distractors]
    missing = [c for c in need if rows and c not in rows[0]]
    if missing:
        raise PanelError(f"question set lacks columns {missing}; available: {sorted(rows[0])}. Map them with --col_*.")
    d1, d2, d3 = col_distractors
    items, seen = [], set()
    for i, row in enumerate(rows):
        rid = str(row[col_id]) if col_id and row.get(col_id) is not None else f"row{i}"
        qid = question_id(rid, row[col_question])
        if qid in seen:
            continue
        seen.add(qid)
        group = str(row[col_group]) if col_group and row.get(col_group) is not None else "all"
        options, correct = permuted_options(qid, row[col_answer], row[d1], row[d2], row[d3], order, seed)
        items.append(
            dict(
                qid=qid,
                id=rid,
                group=group,
                question=row[col_question],
                answer=row[col_answer],
                options=options,
                correct=correct,
                prompt=build_prompt(V2_PROMPT_TEMPLATE, row[col_question], options),
            )
        )
    if limit:
        rnd = random.Random(f"{seed}:limit")
        items = sorted(rnd.sample(items, min(limit, len(items))), key=lambda x: x["qid"])
    return items


# ---------------------------------------------------------------------------------------------------------- requests


def hf_token(dry_run: bool = False) -> str:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_API_KEY")
    if not token:
        if dry_run:
            return "hf_dry_run"
        raise PanelError("HF_TOKEN is not set")
    return token


def check_environment(dry_run: bool = False) -> str:
    """Fail early on settings that would misroute or break every request; return the HF token."""
    for var in ("HF_API_BASE", "HUGGINGFACE_API_BASE"):
        if os.environ.get(var):
            raise PanelError(f"{var} is set; it would override the router routes. Unset it.")
    return hf_token(dry_run)


def setup_litellm(dry_run: bool = False, specs: list[dict] | None = None):
    """Import and configure LiteLLM for the router: no LiteLLM-side retries, and two patches for its HF provider.

    - LiteLLM probes ``config.json`` on the Hub with two uncached GETs before every request; at 13 req/s per model
      that trips the Hub's rate limit, so the probe is disabled.
    - With ``api_base`` set, LiteLLM leaves ``max_retries`` in the request body, which providers may reject.
    """
    import importlib.metadata as md

    import litellm
    from litellm.llms.huggingface.chat.transformation import HuggingFaceChatConfig

    token = check_environment(dry_run)
    version = md.version("litellm")
    if version != TESTED_LITELLM:
        logger.warning(f"litellm {version} is installed; the panel was verified with {TESTED_LITELLM}.")
    if not hasattr(litellm.utils, "_get_max_position_embeddings"):
        raise PanelError(f"litellm {version} lacks utils._get_max_position_embeddings; update the patch in setup_litellm")
    os.environ.setdefault("HUGGINGFACE_API_KEY", token)  # authenticates LiteLLM's Hub mapping lookup
    litellm.suppress_debug_info = True
    litellm.drop_params = False
    litellm.num_retries = 0
    litellm.utils._get_max_position_embeddings = lambda *a, **k: None
    if not getattr(HuggingFaceChatConfig, "_latamqa_patched", False):
        original = HuggingFaceChatConfig.transform_request

        def transform_request(self, model, messages, optional_params, litellm_params, headers):
            optional_params.pop("max_retries", None)
            return original(self, model, messages, optional_params, litellm_params, headers)

        HuggingFaceChatConfig.transform_request = transform_request
        HuggingFaceChatConfig._latamqa_patched = True
    if dry_run:
        install_fake_transport(specs or [])
    return litellm


def install_fake_transport(specs: list[dict]):
    """Replace the router with an offline fake (no network, no cost) and return a function that restores it.

    The fake answers with a random letter. A model whose panel entry has a reasoning switch (non-empty
    ``extra_body``) answers with reasoning prose unless the request carries one of the known switches, so a missing
    switch shows up as leaks. A request without ``X-HF-Bill-To`` gets HTTP 402.
    """
    import httpx
    from litellm.llms.custom_httpx import http_handler
    from litellm.llms.huggingface.chat import transformation

    needs_switch = {s["hub_id"] for s in specs if s.get("extra_body")}
    mapping = {s["hub_id"]: s["provider"] for s in specs}
    original_post, original_mapping = http_handler.HTTPHandler.post, transformation._fetch_inference_provider_mapping
    lock, rnd = threading.Lock(), random.Random(1)

    def fake_mapping(model_id):
        provider = mapping.get(model_id, "novita")
        return {provider: {"status": "live", "providerId": model_id.lower() if provider == "novita" else model_id}}

    def fake_post(self, url, data=None, json=None, headers=None, **kwargs):
        body = json if json is not None else __import__("json").loads(data)
        provider = str(url).split("router.huggingface.co/")[1].split("/")[0]

        def fail(code, msg):  # the real HTTPHandler.post raises HTTPStatusError for non-2xx responses
            httpx.Response(code, json={"error": msg}, request=httpx.Request("POST", str(url))).raise_for_status()

        if not (headers or {}).get("X-HF-Bill-To"):
            fail(402, "fake: missing X-HF-Bill-To")
        if "max_retries" in body:
            fail(400, "fake: unexpected max_retries in body")
        model = body.get("model", "")
        switched = (
            body.get("enable_thinking") is False
            or body.get("reasoning_effort") == "none"
            or (body.get("reasoning") or {}).get("enabled") is False
            or (body.get("chat_template_kwargs") or {}).get("enable_thinking") is False
            or (body.get("thinking") or {}).get("type") == "disabled"
        )
        thinker = any(h.lower() == model.lower() or h.lower() in model.lower() for h in needs_switch)
        with lock:
            letter, latency = rnd.choice("ABCD"), rnd.uniform(0.02, 0.06)
        time.sleep(latency)
        if thinker and not switched:
            content, finish, tokens = "Okay, so the question asks which option A, B", "length", body.get("max_tokens", 16)
        else:
            content, finish, tokens = letter, "stop", 1
        resp = {
            "id": "fake",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": finish}],
            "usage": {"prompt_tokens": 250, "completion_tokens": tokens, "total_tokens": 250 + tokens},
        }
        hdr = {"x-inference-provider": provider if provider != "v1" else "novita", "x-ratelimit-limit-requests": "6000"}
        return httpx.Response(200, json=resp, headers=hdr, request=httpx.Request("POST", str(url)))

    http_handler.HTTPHandler.post = fake_post
    transformation._fetch_inference_provider_mapping = fake_mapping

    def restore():
        http_handler.HTTPHandler.post = original_post
        transformation._fetch_inference_provider_mapping = original_mapping

    return restore


_BILLING_ABORT = threading.Event()  # set on HTTP 402 in this process


def status_of(error: Exception) -> int:
    code = getattr(error, "status_code", None)
    if isinstance(code, int):
        return code
    return {"Timeout": 408, "APIConnectionError": 599, "RateLimitError": 429, "ServiceUnavailableError": 503}.get(
        type(error).__name__, 0
    )


def ask(litellm, spec: dict, item: dict, bill_to: str, token: str, max_attempts: int = MAX_ATTEMPTS) -> dict:
    """Send one question and return a flat record (content, usage, reasoning fields, provider headers, errors).

    429, 5xx, timeouts and connection errors back off exponentially up to ``max_attempts``; a cold-model 400 waits
    20 s; any other error is recorded, not retried; 402 also sets the process-wide billing abort.
    """
    kwargs: dict[str, Any] = dict(
        model=litellm_model(spec),
        messages=[{"role": "system", "content": V2_SYSTEM_MESSAGE}, {"role": "user", "content": item["prompt"]}],
        temperature=TEMPERATURE,
        max_tokens=V2_MAX_TOKENS,
        api_key=token,
        timeout=TIMEOUT_S,
        num_retries=0,
        extra_headers={"X-HF-Bill-To": bill_to},
    )
    if spec.get("extra_body"):
        kwargs["extra_body"] = spec["extra_body"]
    if spec.get("api_base"):
        kwargs["api_base"] = spec["api_base"]
    rec: dict[str, Any] = dict(ts=_now(), model=spec["key"], qid=item["qid"], group=item["group"], correct=item["correct"])
    attempts, backoff, t0 = 0, 1.0, time.time()
    while True:
        attempts += 1
        ts = time.time()
        try:
            resp = litellm.completion(**kwargs)
            latency = round(time.time() - ts, 3)
            break
        except Exception as e:  # every failure is recorded; the caller decides what is fatal
            code, msg = status_of(e), str(e)[:400]
            cold = code == 400 and "cold" in msg.lower()
            if code == 402:
                _BILLING_ABORT.set()
            if attempts < max_attempts and (code in RETRYABLE or cold) and not _BILLING_ABORT.is_set():
                time.sleep((20 if cold else backoff) * (1 + random.random() * 0.25))
                backoff = min(backoff * 2, 16)
                continue
            rec.update(status=code, error=type(e).__name__, error_msg=msg, attempts=attempts)
            rec.update(latency_s=round(time.time() - ts, 3), total_s=round(time.time() - t0, 3))
            return rec
    choice, usage = resp.choices[0], resp.usage
    message = choice.message
    content = message.content or ""
    psf = getattr(message, "provider_specific_fields", None) or {}
    reasoning = getattr(message, "reasoning_content", None) or psf.get("reasoning_content") or psf.get("reasoning")
    details = getattr(usage, "completion_tokens_details", None)
    headers = (getattr(resp, "_hidden_params", {}) or {}).get("additional_headers") or {}
    rec.update(
        status=200,
        attempts=attempts,
        latency_s=latency,
        total_s=round(time.time() - t0, 3),
        content=content,
        finish=choice.finish_reason,
        returned_model=resp.model,
        prompt_tokens=getattr(usage, "prompt_tokens", None),
        completion_tokens=getattr(usage, "completion_tokens", None),
        reasoning_tokens=getattr(details, "reasoning_tokens", None) if details else None,
        reasoning_len=len(reasoning) if reasoning else 0,
        reasoning_head=(reasoning or "")[:300] or None,
        think_tag=has_think_tag(content),
        provider_hdr=headers.get("llm_provider-x-inference-provider") or headers.get("x-inference-provider"),
    )
    return rec


# ---------------------------------------------------------------------------------------------------------- scoring


def is_leak(rec: dict) -> bool:
    """A 200 reply that reasoned: reasoning fields or tokens, a think tag, more tokens than the cap (Novita ignores
    ``max_tokens`` while a model reasons), or a reply cut at the cap that does not start with a letter."""
    if rec.get("status") != 200:
        return False
    cut_prose = rec.get("finish") == "length" and parse_answer_v2(rec.get("content"))[1] not in ("only", "lead")
    return bool(
        rec.get("reasoning_len")
        or (rec.get("reasoning_tokens") or 0) > 0
        or rec.get("think_tag")
        or has_think_tag(rec.get("content"))
        or (rec.get("completion_tokens") or 0) > V2_MAX_TOKENS
        or cut_prose
    )


def score(rec: dict | None) -> tuple[str | None, str]:
    """(letter, rule). Rules: only / lead / cue / none from `parse_answer_v2`, or error / leak / missing (no letter)."""
    if rec is None:
        return None, "missing"
    if rec.get("status") != 200:
        return None, "error"
    if is_leak(rec):
        return None, "leak"
    return parse_answer_v2(rec.get("content"))


def latest_records(path: Path) -> dict[str, dict]:
    """Last record per question; a 200 supersedes any error before or after it."""
    last: dict[str, dict] = {}
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r["qid"] not in last or r["status"] == 200:
                    last[r["qid"]] = r
    return last


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95 % interval for k successes out of n."""
    if not n:
        return float("nan"), float("nan")
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


# ---------------------------------------------------------------------------------------------------------- worker


def run_model(
    spec: dict,
    items_file: str,
    out_dir: str,
    rate: float,
    cap: int,
    bill_to: str,
    dry_run: bool,
    max_leaks: int,
    max_error_rate: float,
    retry_rounds: int,
) -> None:
    """Evaluate one model (in its own process), resuming from its JSONL log.

    Writes ``<key>.ABORT`` with the reason if the model stops early. A billing stop (402 or the spending-limit 403)
    also writes ``STOP``, which halts every other model of the run.
    """
    litellm = setup_litellm(dry_run, [spec])
    token = hf_token(dry_run)
    key = spec["key"]
    items = json.loads(Path(items_file).read_text(encoding="utf-8"))
    log_path, abort_file, stop_file = Path(out_dir) / f"{key}.jsonl", Path(out_dir) / f"{key}.ABORT", Path(out_dir) / "STOP"
    done = latest_records(log_path)
    state: dict[str, Any] = dict(leaks=sum(1 for r in done.values() if is_leak(r)), errors=0, answered=0, consec=0)
    state["reason"] = None
    lock, stop = threading.Lock(), threading.Event()

    def check(rec: dict) -> None:
        with lock:
            if rec["status"] == 200:
                state["answered"] += 1
                state["consec"] = 0
                state["leaks"] += is_leak(rec)
            else:
                state["errors"] += 1
                state["consec"] += 1
            n = state["answered"] + state["errors"]
            spending_limit = rec["status"] == 403 and "spending limit" in rec.get("error_msg", "")
            if _BILLING_ABORT.is_set() or spending_limit:
                state["reason"] = (
                    "billing: monthly spending limit for Inference Providers reached. HF holds a provisional $0.01 per "
                    "request for ~2 min, so a full-speed panel carries ~$60-120 of holds; raise the limit or top up, "
                    "then re-run the same command to resume"
                    if spending_limit
                    else "HTTP 402 (billing): check the org credits and the X-HF-Bill-To header"
                )
                stop_file.write_text(f"{key}: {state['reason']}\n")
            elif rec["status"] == 403:
                state["reason"] = f"HTTP 403: {rec.get('error_msg', '')[:200]}"
            elif state["leaks"] >= max_leaks:
                state["reason"] = f"{state['leaks']} reasoning leaks (limit {max_leaks}); check the reasoning switch"
            elif state["consec"] >= 50:
                state["reason"] = "50 consecutive errors"
            elif n >= 500 and state["errors"] / n > max_error_rate:
                state["reason"] = f"error rate {state['errors'] / n:.1%} > {max_error_rate:.1%}"
            if state["reason"]:
                stop.set()

    def one_pass(todo: list[dict], rate_: float, cap_: int, f) -> None:
        slots = threading.Semaphore(cap_)
        t0 = time.time()

        def job(item: dict) -> None:
            try:
                try:
                    rec = ask(litellm, spec, item, bill_to, token)
                except Exception as e:  # e.g. a malformed response: record it so it is retried
                    rec = dict(
                        ts=_now(),
                        model=key,
                        qid=item["qid"],
                        group=item["group"],
                        correct=item["correct"],
                        status=0,
                        error=type(e).__name__,
                        error_msg=str(e)[:400],
                    )
                with lock:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f.flush()
                check(rec)
            finally:
                slots.release()

        with ThreadPoolExecutor(max_workers=cap_) as pool:
            for i, item in enumerate(todo):
                if stop.is_set() or stop_file.exists():
                    break
                delay = t0 + i / rate_ - time.time()
                if delay > 0:
                    time.sleep(delay)
                slots.acquire()
                if stop.is_set():
                    slots.release()
                    break
                pool.submit(job, item)

    t_start = time.time()
    with open(log_path, "a", encoding="utf-8") as f:
        one_pass([it for it in items if done.get(it["qid"], {}).get("status") != 200], rate, cap, f)
        for _ in range(retry_rounds):  # slower passes for questions still without a 200
            if stop.is_set() or stop_file.exists():
                break
            latest = latest_records(log_path)
            todo = [it for it in items if latest.get(it["qid"], {}).get("status") != 200]
            if not todo:
                break
            time.sleep(0 if dry_run else 10)
            one_pass(todo, max(1.0, rate / 5), max(2, cap // 5), f)
    if not state["reason"] and stop_file.exists():
        state["reason"] = "stopped: " + stop_file.read_text(encoding="utf-8").strip()
    if state["reason"]:
        abort_file.write_text(state["reason"] + "\n")
    elif abort_file.exists():
        abort_file.unlink()
    (Path(out_dir) / f"{key}.timing.json").write_text(json.dumps({"wall_s": round(time.time() - t_start, 1)}))


# ---------------------------------------------------------------------------------------------------------- reports


def summarize_model(spec: dict, items: list[dict], out_dir: Path, set_name: str, run: int, panel_name: str) -> dict:
    """Score one model's log: per-question CSV (harness columns + parse rule), per-group summary files, JSON summary."""
    key = spec["key"]
    recs = latest_records(out_dir / f"{key}.jsonl")
    rows, n_ok, rules, groups, latencies, cost = [], 0, {}, {}, [], 0.0
    for it in items:
        r = recs.get(it["qid"])
        letter, rule = score(r)
        ok = letter == it["correct"]
        n_ok += ok
        rules[rule] = rules.get(rule, 0) + 1
        g = groups.setdefault(it["group"], [0, 0, 0])  # correct, total, errors
        g[0] += ok
        g[1] += 1
        g[2] += rule in ("error", "missing")
        if r and r.get("status") == 200:
            latencies.append(r["latency_s"])
            cost += (r.get("prompt_tokens") or 0) * spec["price"]["input"] / 1e6
            cost += (r.get("completion_tokens") or 0) * spec["price"]["output"] / 1e6
        rows.append(
            {
                "article_id": it["id"],
                "question": it["question"],
                "correct_answer": it["answer"],
                "option_A": it["options"][0],
                "option_B": it["options"][1],
                "option_C": it["options"][2],
                "option_D": it["options"][3],
                "correct_letter": it["correct"],
                "model_response": (r or {}).get("content"),
                "predicted_letter": letter,
                "is_correct": ok,
                "parse_rule": rule,
                "group": it["group"],
                "provider": (r or {}).get("provider_hdr"),
                "returned_model": (r or {}).get("returned_model"),
                "latency_s": (r or {}).get("latency_s"),
            }
        )
    n = len(items)
    tag = f"{set_name}_run{run}"
    with open(out_dir / f"mcq_eval_results_{tag}_{key}.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    errors = rules.get("error", 0) + rules.get("missing", 0)
    answered = n - errors
    ok_recs = [r for r in recs.values() if r.get("status") == 200]
    abort_file, timing = out_dir / f"{key}.ABORT", out_dir / f"{key}.timing.json"
    summary = dict(
        protocol=PROTOCOL_V2,
        panel=panel_name,
        model=spec["hub_id"],
        provider=spec["provider"],
        key=key,
        set=set_name,
        run=run,
        total=n,
        correct=n_ok,
        accuracy=n_ok / n if n else float("nan"),
        ci95=list(wilson(n_ok, n)),
        errors=errors,
        coverage=answered / n if n else 0.0,
        accuracy_answered=n_ok / answered if answered else float("nan"),
        parse_rules=rules,
        leaks=rules.get("leak", 0),
        latency_p50=statistics.median(latencies) if latencies else None,
        latency_p95=sorted(latencies)[int(0.95 * (len(latencies) - 1))] if latencies else None,
        cost_usd=round(cost, 4),
        groups={k: dict(correct=v[0], total=v[1], errors=v[2], accuracy=v[0] / v[1]) for k, v in groups.items()},
        providers=sorted({str(r.get("provider_hdr")) for r in ok_recs}),
        returned_models=sorted({str(r.get("returned_model")) for r in ok_recs}),
        aborted=abort_file.read_text(encoding="utf-8").strip() if abort_file.exists() else None,
        wall_s=json.loads(timing.read_text())["wall_s"] if timing.exists() else None,
    )
    (out_dir / f"summary_{key}.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    # Harness-style summaries, one per group. Accuracy is over ALL questions of the group (errors count as wrong).
    for name, g in summary["groups"].items():
        fields = dict(
            model=spec["hub_id"],
            region=f"{set_name}-{name}",
            lang=f"run{run}",
            protocol=PROTOCOL_V2,
            total=g["total"],
            correct=g["correct"],
            errors=g["errors"],
            accuracy=g["accuracy"],
        )
        (out_dir / f"mcq_eval_summary_{set_name}-{name}_run{run}_{key}.txt").write_text(
            "".join(f"{k}: {v}\n" for k, v in fields.items())
        )
    return summary


def report(specs: list[dict], items: list[dict], out_dir: Path, set_name: str, run: int, panel_name: str, dry_run: bool):
    """Summarize every model, then write ``report.md`` / ``report.json`` with the publishability verdict."""
    summaries = [summarize_model(s, items, out_dir, set_name, run, panel_name) for s in specs]

    def fmt(x):
        return "—" if x is None else f"{x:.2f}"

    lines = [
        f"# {panel_name} · {set_name} · run {run} ({len(items)} questions, protocol {PROTOCOL_V2})"
        + (" — DRY RUN" if dry_run else ""),
        "",
        "| model | provider | accuracy | 95% CI | answered | errors | leaks | parse rules | p50 s | p95 s | wall min "
        "| cost $ | status |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for s in summaries:
        rules = ", ".join(f"{k} {v}" for k, v in sorted(s["parse_rules"].items()) if k not in ("error", "missing"))
        wall = f"{s['wall_s'] / 60:.1f}" if s["wall_s"] else "—"
        status = "ABORTED: " + s["aborted"] if s["aborted"] else "ok"
        lines.append(
            f"| {s['key']} | {s['provider']} | {100 * s['accuracy']:.1f}% | {100 * s['ci95'][0]:.1f}-"
            f"{100 * s['ci95'][1]:.1f} | {100 * s['coverage']:.2f}% | {s['errors']} | {s['leaks']} | {rules} | "
            f"{fmt(s['latency_p50'])} | {fmt(s['latency_p95'])} | {wall} | {s['cost_usd']:.2f} | {status} |"
        )
    groups = sorted({g for s in summaries for g in s["groups"]})
    if len(groups) > 1:
        lines += ["", "Accuracy by group:", "", "| model | " + " | ".join(groups) + " |", "|---" * (len(groups) + 1) + "|"]
        for s in summaries:
            cells = [f"{100 * s['groups'][g]['accuracy']:.1f}%" if g in s["groups"] else "—" for g in groups]
            lines.append(f"| {s['key']} | " + " | ".join(cells) + " |")
    bad = [s["key"] for s in summaries if s["aborted"] or s["coverage"] < MIN_COVERAGE or s["leaks"]]
    slowest = max((s["wall_s"] or 0) for s in summaries) / 60
    lines += [
        "",
        f"Slowest model: {slowest:.1f} min (target < 30). Cost at list prices: ${sum(s['cost_usd'] for s in summaries):.2f}.",
        "Accuracy counts errors, unparsable and leaked answers as wrong; `accuracy_answered` in the JSON is the v1 definition.",
        "",
        "**Publishable**: "
        + (
            f"yes: every model answered ≥ {MIN_COVERAGE:.1%} with no leak and no stop"
            if not bad
            else "NO for " + ", ".join(bad)
        ),
    ]
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / "report.json").write_text(json.dumps(summaries, indent=1, ensure_ascii=False))
    print("\n".join(lines))
    print(f"-> {out_dir / 'report.md'}")
    return summaries


# ---------------------------------------------------------------------------------------------------------- stages


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _http_json(url: str, token: str | None = None, timeout: float = 30):
    import httpx

    r = httpx.get(url, headers={"Authorization": f"Bearer {token}"} if token else {}, timeout=timeout, follow_redirects=True)
    try:
        return r.status_code, r.json()
    except ValueError:
        return r.status_code, r.text[:500]


def usage_snapshot(bill_to: str, dry_run: bool = False) -> dict:
    """Today's Inference Providers usage for the personal account and the billing org (free GETs)."""
    if dry_run:
        return {"ts": _now(), "personal": {"n": 0, "usd": 0.0}, "org": {"n": 0, "usd": 0.0}}
    token, day0 = hf_token(), int(time.time()) // 86400 * 86400
    snap: dict[str, Any] = {"ts": _now()}
    for label, scope in (("personal", "settings"), ("org", f"organizations/{bill_to}")):
        code, body = _http_json(f"{HUB}/api/{scope}/billing/usage-v2?startDate={day0}&endDate={int(time.time())}", token)
        if code != 200:
            snap[label] = {"error": code}
            continue
        ip = body["usage"]["inferenceProviders"]
        snap[label] = {"n": ip["numRequests"], "usd": ip["usedNanoUsd"] / 1e9}
    return snap


def estimate_cost(specs: list[dict], counts: dict[str, int]) -> float:
    """Upper bound: every answer uses the whole token cap."""
    by_key = {s["key"]: s for s in specs}
    return sum(
        n * (EST_PROMPT_TOKENS * by_key[k]["price"]["input"] + V2_MAX_TOKENS * by_key[k]["price"]["output"]) / 1e6
        for k, n in counts.items()
    )


def run_dir(args, panel: dict) -> Path:
    return Path(args.results_dir) / panel["name"].lower() / args.set_name / f"run{args.run}"


def prepare(args, panel: dict, specs: list[dict]) -> tuple[list[dict], Path]:
    """Build the run's items and write ``items.json`` / ``run_meta.json``; refuse to mix two question sets or orders."""
    items = build_items(
        read_questions(args.questions),
        order=args.run,
        seed=args.seed,
        limit=args.limit,
        col_id=args.col_id,
        col_question=args.col_question,
        col_answer=args.col_answer,
        col_distractors=tuple(args.col_distractors.split(",")),
        col_group=args.col_group,
    )
    if not items:
        raise PanelError("no questions loaded")
    out_dir = run_dir(args, panel)
    out_dir.mkdir(parents=True, exist_ok=True)
    items_file = out_dir / "items.json"
    if items_file.exists():
        old = {it["qid"]: it["correct"] for it in json.loads(items_file.read_text(encoding="utf-8"))}
        if old != {it["qid"]: it["correct"] for it in items}:
            raise PanelError(f"{out_dir} already holds a run on different questions or option orders; use another --set_name")
    items_file.write_text(json.dumps(items, ensure_ascii=False))
    meta = dict(
        protocol=PROTOCOL_V2,
        panel=panel["name"],
        panel_file=panel["path"],
        questions=args.questions,
        n=len(items),
        run=args.run,
        seed=args.seed,
        limit=args.limit,
        rate=args.rate,
        system=V2_SYSTEM_MESSAGE,
        max_tokens=V2_MAX_TOKENS,
        temperature=TEMPERATURE,
        template=V2_PROMPT_TEMPLATE,
        bill_to=args.bill_to or panel["bill_to"],
        dry_run=args.dry_run,
        models=specs,
        ts=_now(),
    )
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    shares = ", ".join(f"{x} {sum(it['correct'] == x for it in items) / len(items):.1%}" for x in "ABCD")
    print(f"{len(items)} questions, groups {sorted({it['group'] for it in items})}, correct-letter shares {shares}")
    return items, out_dir


def stage_run(args, panel: dict, specs: list[dict]) -> None:
    items, out_dir = prepare(args, panel, specs)
    bill_to = args.bill_to or panel["bill_to"]
    todo = {
        s["key"]: len(items) - sum(r["status"] == 200 for r in latest_records(out_dir / f"{s['key']}.jsonl").values())
        for s in specs
    }
    cost = estimate_cost(specs, todo)
    print(f"run {args.run}: {sum(todo.values())} requests to send ({todo}), estimated cost <= ${cost:.2f}, billed to {bill_to}")
    print(f"projected wall time at {args.rate:.1f} req/s per model: {max(todo.values()) / args.rate / 60:.1f} min")
    if cost > args.budget_usd:
        raise PanelError(f"estimate exceeds --budget_usd {args.budget_usd}")
    if not (args.yes or args.dry_run):
        raise PanelError("paid stage: re-run with --yes (or --dry_run to test offline)")
    run_panel_processes(specs, len(items), out_dir, args, bill_to)
    report(specs, items, out_dir, args.set_name, args.run, panel["name"], args.dry_run)


def run_panel_processes(specs: list[dict], n_items: int, out_dir: Path, args, bill_to: str, max_leaks=None) -> None:
    """Run every model on ``out_dir/items.json`` in parallel (one process each), print progress until all finish,
    and record the billing usage before and after. ``max_leaks`` (an int, or a dict by model key) overrides
    ``args.max_leaks``."""
    check_environment(args.dry_run)
    (out_dir / "STOP").unlink(missing_ok=True)
    caps = dict(kv.split("=") for kv in args.cap or [])
    snap0 = usage_snapshot(bill_to, args.dry_run)
    ctx = mp.get_context("spawn")
    procs = {}
    for s in specs:
        cap = int(caps.get(s["key"], s["max_in_flight"]))
        leaks = args.max_leaks if max_leaks is None else max_leaks
        procs[s["key"]] = ctx.Process(
            target=run_model,
            args=(
                s,
                str(out_dir / "items.json"),
                str(out_dir),
                args.rate,
                cap,
                bill_to,
                args.dry_run,
                leaks[s["key"]] if isinstance(leaks, dict) else leaks,
                args.max_error_rate,
                args.retry_rounds,
            ),
        )
        procs[s["key"]].start()
    t0 = time.time()
    while any(p.is_alive() for p in procs.values()):
        time.sleep(args.progress_s)
        parts = []
        for s in specs:
            recs = latest_records(out_dir / f"{s['key']}.jsonl")
            ok = sum(r["status"] == 200 for r in recs.values())
            leaks_now = sum(is_leak(r) for r in recs.values())
            aborted = " ABORTED" if (out_dir / f"{s['key']}.ABORT").exists() else ""
            parts.append(f"{s['key']} {ok}/{n_items} err {len(recs) - ok} leak {leaks_now}{aborted}")
        print(f"[{(time.time() - t0) / 60:5.1f} min] " + " | ".join(parts), flush=True)
    for p in procs.values():
        p.join()
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%H%M%S")
    (out_dir / f"billing_{stamp}.json").write_text(
        json.dumps({"before": snap0, "after": usage_snapshot(bill_to, args.dry_run)}, indent=1)
    )


def stage_report(args, panel: dict, specs: list[dict]) -> None:
    out_dir = run_dir(args, panel)
    items_file = out_dir / "items.json"
    if not items_file.exists():
        raise PanelError(f"no run found in {out_dir}")
    meta = json.loads((out_dir / "run_meta.json").read_text(encoding="utf-8"))
    items = json.loads(items_file.read_text(encoding="utf-8"))
    if not (args.models or args.swap):  # rebuild the report for the models the run actually used
        specs = meta["models"]
    report(specs, items, out_dir, args.set_name, args.run, panel["name"], meta.get("dry_run", False))


def stage_check(args, panel: dict, specs: list[dict]) -> None:
    """Free pre-run checks; exits non-zero if any check fails."""
    import importlib.metadata as md

    results: list[tuple[str, str, str]] = []

    def add(name: str, ok: bool | None, detail: str) -> None:
        results.append(("INFO" if ok is None else "PASS" if ok else "FAIL", name, detail))
        print(f"[{results[-1][0]}] {name}: {detail}")

    version = md.version("litellm")
    add("litellm version", version == TESTED_LITELLM, f"{version} (verified: {TESTED_LITELLM})")
    for var in ("HF_API_BASE", "HUGGINGFACE_API_BASE"):
        add(f"{var} unset", not os.environ.get(var), os.environ.get(var) or "unset")
    token, bill_to = hf_token(), args.bill_to or panel["bill_to"]
    code, who = _http_json(f"{HUB}/api/whoami-v2", token)
    if code != 200:
        add("token", False, f"whoami HTTP {code}")
        sys.exit(1)
    org = next((o for o in who.get("orgs", []) if o.get("name") == bill_to), None)
    add("token owner", None, str(who.get("name")))
    add(
        f"org {bill_to} can pay",
        bool(org and org.get("canPay")),
        f"member={org is not None} canPay={org and org.get('canPay')}",
    )
    for s in specs:
        code, body = _http_json(f"{HUB}/api/models/{s['hub_id']}?expand[]=inferenceProviderMapping&expand[]=gated", token)
        ipm = body.get("inferenceProviderMapping", {}) if code == 200 else {}
        if isinstance(ipm, list):
            ipm = {x.get("provider"): x for x in ipm}
        entry = ipm.get(s["provider"], {})
        add(
            f"{s['key']} on {s['provider']}",
            entry.get("status") == "live",
            f"status={entry.get('status')} providerId={entry.get('providerId')} "
            f"gated={body.get('gated') if code == 200 else code}",
        )
        if code == 200 and body.get("gated"):
            gcode, _ = _http_json(f"{HUB}/{s['hub_id']}/resolve/main/config.json", token)
            add(f"{s['key']} gated access", gcode == 200, f"HTTP {gcode}")
        if s["provider"] == "deepinfra":
            dcode, d = _http_json(f"https://api.deepinfra.com/models/{s['hub_id']}")
            deprecated = d.get("deprecated") if dcode == 200 else None
            add(
                f"{s['key']} not retired on deepinfra",
                dcode == 200 and not deprecated,
                f"HTTP {dcode} quantization={d.get('quantization') if dcode == 200 else None} deprecated={deprecated}",
            )
    failed = [name for status, name, _ in results if status == "FAIL"]
    print(f"{len(failed)} check(s) failed: {', '.join(failed)}" if failed else "All checks passed.")
    if failed:
        sys.exit(1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="panel", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["check", "run", "report"])
    parser.add_argument(
        "--panel", default=DEFAULT_PANEL, help=f"panel name in latamqa/panels/ or a YAML path (default {DEFAULT_PANEL})"
    )
    parser.add_argument("--questions", help="question set: .parquet/.csv/.jsonl/.json file, or hf:org/name[:split]")
    parser.add_argument("--set_name", help="name used in output paths (default: the question file's stem)")
    parser.add_argument(
        "--run", type=int, default=1, choices=list(range(1, V2_OPTION_ORDERS + 1)), help="run = option order 1-5"
    )
    parser.add_argument("--seed", type=int, default=42, help="seed of the option orders and of --limit")
    parser.add_argument("--limit", type=int, help="use a stable random subset of N questions (dress rehearsal)")
    parser.add_argument("--models", help="comma-separated model keys (default: the panel's models)")
    parser.add_argument("--swap", action="append", help="replace a panel model, e.g. --swap qwen2.5-72b=qwen2.5-72b-di")
    parser.add_argument("--bill_to", help="HF org billed through X-HF-Bill-To (default: the panel's bill_to)")
    parser.add_argument("--rate", type=float, default=DEFAULT_RATE, help="requests/s per model (default 13.3)")
    parser.add_argument("--cap", action="append", help="in-flight cap override, e.g. --cap kimi-k2=120")
    parser.add_argument("--max_leaks", type=int, default=10, help="stop a model after this many reasoning leaks")
    parser.add_argument("--max_error_rate", type=float, default=0.01, help="stop a model above this error rate")
    parser.add_argument("--retry_rounds", type=int, default=2, help="slower passes for questions that still failed")
    parser.add_argument("--budget_usd", type=float, default=40.0, help="refuse to start if the cost estimate exceeds this")
    parser.add_argument("--progress_s", type=float, default=60.0, help="seconds between progress lines")
    parser.add_argument("--results_dir", default=str(DEFAULT_RESULTS_DIR), help="root folder for panel results")
    parser.add_argument("--col_id", default="article_id", help="question id column (optional)")
    parser.add_argument("--col_question", default="question")
    parser.add_argument("--col_answer", default="answer")
    parser.add_argument("--col_distractors", default="distractor1,distractor2,distractor3")
    parser.add_argument("--col_group", help="column to report accuracy by, e.g. a region or language column")
    parser.add_argument("--yes", action="store_true", help="confirm a paid run")
    parser.add_argument("--dry_run", action="store_true", help="offline: fake provider, no network, no cost")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        panel = load_panel(args.panel)
        specs = select_models(panel, args.models, args.swap)
        if args.stage == "run" and not args.questions:
            raise PanelError("--questions is required")
        if args.stage != "check" and not args.set_name:
            if not args.questions:
                raise PanelError("--set_name (or --questions) is required")
            args.set_name = Path(args.questions.removeprefix("hf:").split(":")[0]).stem
        {"check": stage_check, "run": stage_run, "report": stage_report}[args.stage](args, panel, specs)
    except PanelError as e:
        logger.fatal(str(e))
        sys.exit(-1)


if __name__ == "__main__":
    main()
