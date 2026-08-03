#!/usr/bin/env python3
"""
Universal script to evaluate models through API (OpenAI, Mistral, Anthropic, Ollama, vLLM, etc.) using LiteLLM.
"""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, List

import pandas as pd
import structlog
from datasets import load_dataset
from litellm.exceptions import AuthenticationError, BadRequestError
from rich.console import Console
from rich.markdown import Markdown
from tqdm.auto import tqdm

# The MCQ core (prompt template, letter parser, prompt builder, deterministic
# shuffle, and the single-question LLM call) lives in the dependency-light
# `mcq_core` leaf module so it can be imported without dragging in pandas /
# datasets / rich / tqdm. Re-exported here so existing imports keep working:
#   from latamqa.eval_mcq import evaluate_mcq, extract_answer, build_prompt, ...
from latamqa.mcq_core import (  # noqa: F401  (re-exported for backwards compatibility)
    DEFAULT_MAX_TOKENS,
    DEFAULT_NUM_RETRIES,
    DEFAULT_PROPMT_TEMPLATE,
    build_prompt,
    evaluate_mcq,
    evaluate_mcq_dict,
    extract_answer,
    shuffle_options,
)

logger = structlog.get_logger()

DEFAULT_RESULTS_DIR = Path(__file__).parent.parent / "results"

REGIONAL_DATASETS = ["es-la", "es-es", "pt-br"]
TARGET_LANGUAGES = ["regional", "english"]

# Number of MCQ requests kept in flight at once. Concurrency turns the
# otherwise-sequential eval loop into a batch of parallel calls, which is a
# large speedup against API providers and lets a self-hosted vLLM server use
# its continuous batching.
DEFAULT_BATCH_SIZE = 16

# Endpoints exposing an OpenAI-compatible *asynchronous* Batch API. When an
# evaluation targets one of these hosts, every question is packed into a single
# batch job (upload file -> create batch -> poll -> download) instead of firing
# many concurrent live requests. This sidesteps the per-second rate limits that
# concurrency hits and, on Maritaca, costs ~50% less. Detection is by host so the
# batch path auto-engages for the endpoint pinned in the model YAML — no flag.
BATCH_API_HOSTS = ("chat.maritaca.ai",)
# How often (seconds) to poll a submitted batch job for completion.
DEFAULT_BATCH_POLL_INTERVAL = 30
# Batch statuses that mean "keep waiting"; every other status is terminal.
BATCH_IN_PROGRESS_STATUSES = frozenset({"validating", "in_progress", "finalizing", "cancelling"})


def sanitize(s: str) -> str:
    """Sanitize string for use in filenames."""
    return s.replace("/", "-").replace(":", "-")


def _run_live_requests(
    model: str,
    prompt_template: str,
    tasks: List[dict],
    temperature: float,
    llm_api_key: str | None,
    llm_uri: str | None,
    num_retries: int,
    batch_size: int,
    desc: str,
) -> list:
    """Issue one live LLM call per task concurrently, at most ``batch_size`` in flight.

    Each answer is written back to its own slot so the returned list stays aligned
    with ``tasks`` (dataset order). A per-question failure that survives LiteLLM's
    retries is stored as the exception; a misconfiguration (auth / bad request)
    aborts the whole run, since every remaining request would fail identically.
    """
    responses: list = [None] * len(tasks)
    with ThreadPoolExecutor(max_workers=max(1, batch_size)) as executor:
        future_to_idx = {
            executor.submit(
                evaluate_mcq,
                model,
                prompt_template,
                task["question"],
                task["options"],
                temperature,
                llm_api_key=llm_api_key,
                llm_uri=llm_uri,
                num_retries=num_retries,
            ): idx
            for idx, task in enumerate(tasks)
        }
        for future in tqdm(as_completed(future_to_idx), total=len(tasks), desc=desc, leave=False):
            idx = future_to_idx[future]
            try:
                responses[idx] = future.result()
            except (AuthenticationError, BadRequestError) as e:
                # Misconfiguration: every request will fail the same way, so
                # there is no point finishing the batch — abort the whole run.
                executor.shutdown(wait=False, cancel_futures=True)
                logger.fatal(f"{type(e).__name__}: {e}")
                exit(-1)
            except Exception as e:
                # Transient/other failure for this one question (survived the
                # retries): record it and let the rest of the batch finish.
                logger.error(f"article_id={tasks[idx]['item'].get('article_id')}: {e}.")
                responses[idx] = e
    return responses


def is_batch_endpoint(llm_uri: str | None) -> bool:
    """Return True when ``llm_uri`` targets an OpenAI-compatible Batch API host.

    Detection is by host (see :data:`BATCH_API_HOSTS`) so the asynchronous batch
    path auto-engages for e.g. the Maritaca endpoint pinned in a model YAML,
    without the caller having to remember an extra flag.
    """
    return bool(llm_uri) and any(host in llm_uri for host in BATCH_API_HOSTS)


def _batch_client(llm_api_key: str | None, llm_uri: str | None) -> Any:
    """Build an OpenAI SDK client pointed at the Batch API endpoint (e.g. Maritaca).

    The batch lifecycle is driven with the OpenAI SDK directly rather than
    LiteLLM's batch wrapper: LiteLLM re-validates the provider response through
    its stricter ``LiteLLMBatch`` model, which rejects Maritaca's ``errors: []``
    (an empty list where an object is expected) and crashes on retrieve. The
    OpenAI client — which LiteLLM installs anyway and which Maritaca's own docs
    use — parses the same response without complaint. The key falls back to the
    usual environment variables when not passed explicitly.
    """
    from openai import OpenAI

    # `llm_api_key` is an empty string when e.g. `--llm_api_key "$MARITACA_API_KEY"`
    # is passed from a shell where the variable is unset; treat that as "absent"
    # and fall back to the environment. Fail fast with an actionable message
    # rather than letting the OpenAI client raise a generic error deep in the run.
    api_key = llm_api_key or os.environ.get("MARITACA_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.fatal(
            f"No API key for the Batch API endpoint «{llm_uri}». Pass a non-empty --llm_api_key, "
            "or export MARITACA_API_KEY (verify with `echo $MARITACA_API_KEY` — it may be empty in this shell)."
        )
        exit(-1)
    return OpenAI(api_key=api_key, base_url=llm_uri)


def _custom_id_to_idx(custom_id: Any) -> int | None:
    """Recover a task index from a ``req-<idx>`` custom_id, or None if it doesn't fit."""
    if not isinstance(custom_id, str) or not custom_id.startswith("req-"):
        return None
    try:
        return int(custom_id[len("req-") :])
    except ValueError:
        return None


def _content_to_text(resp: Any) -> str:
    """Read a LiteLLM/OpenAI file-content response body as text."""
    text = getattr(resp, "text", None)
    if isinstance(text, str):
        return text
    content = getattr(resp, "content", None)
    if isinstance(content, (bytes, bytearray)):
        return content.decode("utf-8")
    return str(content if content is not None else resp)


def _iter_jsonl(text: str):
    """Yield each non-blank line of ``text`` parsed as JSON."""
    for raw in text.splitlines():
        raw = raw.strip()
        if raw:
            yield json.loads(raw)


def _extract_batch_content(record: dict) -> Any:
    """Pull the answer text out of one batch output line, or an Exception on failure."""
    if record.get("error"):
        return RuntimeError(f"batch error: {record['error']}")
    response = record.get("response") or {}
    status = response.get("status_code")
    if status not in (None, 200):
        return RuntimeError(f"batch HTTP {status}: {response.get('body')}")
    try:
        return response["body"]["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError) as e:
        return RuntimeError(f"malformed batch response: {e!r}")


def _run_batch_requests(
    model: str,
    prompt_template: str,
    tasks: List[dict],
    temperature: float,
    llm_api_key: str | None,
    llm_uri: str | None,
    results_dir: str | Path,
    tag: str,
    poll_interval: int = DEFAULT_BATCH_POLL_INTERVAL,
) -> list:
    """Answer every task through an OpenAI-compatible asynchronous Batch API.

    Rather than firing one live request per question — which hammers the
    provider's rate limit — all questions are packed into a single JSONL file and
    submitted as ONE batch job via the OpenAI-compatible Batch API (see
    :func:`_batch_client` for why the OpenAI SDK is used directly). The provider
    answers them asynchronously within the completion window; this polls until the
    job reaches a terminal state, then maps each answer back to its dataset slot by
    ``custom_id`` (batch output order is not guaranteed). The input and output
    JSONL files are kept in ``results_dir`` for traceability. The returned list is
    aligned with ``tasks``: a string answer, an Exception for a failed question,
    or None when the provider returned nothing for it.
    """
    client = _batch_client(llm_api_key, llm_uri)
    # Maritaca's OpenAI-compatible body expects the bare model name, without the
    # LiteLLM "openai/…" routing prefix used for live completions.
    body_model = model.split("/", 1)[1] if model.startswith("openai/") else model

    # 1 — write one request line per task, tagged with a positional custom_id so
    #     out-of-order results can be mapped back to their dataset slot.
    input_path = Path(results_dir) / f"batch_input_{tag}.jsonl"
    with open(input_path, "w", encoding="utf-8") as f:
        for idx, task in enumerate(tasks):
            prompt = build_prompt(prompt_template, task["question"], task["options"])
            line = {
                "custom_id": f"req-{idx}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": body_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                    "max_tokens": DEFAULT_MAX_TOKENS,  # a letter, plus room for a thinking model's reasoning
                },
            }
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

    # 2 — upload the file and create a single batch job for the whole slice.
    logger.info(f"Uploading {len(tasks)} requests as one batch job to «{llm_uri}».")
    with open(input_path, "rb") as f:
        batch_input_file = client.files.create(file=f, purpose="batch")
    batch = client.batches.create(
        input_file_id=batch_input_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": f"LatamQA {tag}"},
    )
    logger.info(f"Batch job {batch.id} created; polling every {poll_interval}s (completion window: 24h).")

    # 3 — poll until the job reaches a terminal state.
    while batch.status in BATCH_IN_PROGRESS_STATUSES:
        time.sleep(poll_interval)
        batch = client.batches.retrieve(batch.id)
        counts = getattr(batch, "request_counts", None)
        done = getattr(counts, "completed", 0) or 0
        failed = getattr(counts, "failed", 0) or 0
        logger.info(f"Batch {batch.id}: status={batch.status} completed={done} failed={failed} / {len(tasks)}.")

    responses: list = [None] * len(tasks)

    if batch.status == "failed":
        # A whole-batch failure (e.g. validation, bad key) mirrors the live
        # path's fatal handling: every question would fail identically.
        detail = getattr(batch, "errors", None) or "no error details provided"
        logger.fatal(f"Batch job {batch.id} failed: {detail}")
        exit(-1)

    # 4 — download the output (and error) files and map answers back by custom_id.
    output_file_id = getattr(batch, "output_file_id", None)
    if output_file_id:
        output_text = _content_to_text(client.files.content(output_file_id))
        (Path(results_dir) / f"batch_output_{tag}.jsonl").write_text(output_text, encoding="utf-8")
        for record in _iter_jsonl(output_text):
            idx = _custom_id_to_idx(record.get("custom_id"))
            if idx is not None:
                responses[idx] = _extract_batch_content(record)

    # Requests that errored land in a separate error file; surface them as
    # per-question failures (scored as errors) rather than silent None answers.
    error_file_id = getattr(batch, "error_file_id", None)
    if error_file_id:
        error_text = _content_to_text(client.files.content(error_file_id))
        for record in _iter_jsonl(error_text):
            idx = _custom_id_to_idx(record.get("custom_id"))
            if idx is not None:
                responses[idx] = RuntimeError(f"batch error: {record.get('error') or record.get('response')}")

    if batch.status in {"expired", "cancelled"}:
        logger.error(f"Batch job {batch.id} ended as «{batch.status}»; unanswered questions are recorded as errors.")

    return responses


def run_evaluation(
    model: str,
    region: str = "es-la",
    lang: str = "regional",
    max_results: int | None = None,
    seed: int = 42,
    temperature: float = 0.0,
    prompt_template: str | None = None,
    results_dir: str | Path | None = None,
    llm_api_key: str | None = None,
    llm_uri: str | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_retries: int = DEFAULT_NUM_RETRIES,
    batch_poll_interval: int = DEFAULT_BATCH_POLL_INTERVAL,
):
    """Run the MCQ evaluation."""

    if results_dir is None:
        logger.info(f"No results directory specified, using default: {DEFAULT_RESULTS_DIR}")
        results_dir = DEFAULT_RESULTS_DIR

    os.makedirs(results_dir, exist_ok=True)

    dataset_name = f"inria-chile/latamqa_mcq_{region}"

    logger.info(f"Loading dataset «{dataset_name}».")
    ds = load_dataset(dataset_name)

    if lang == "english":
        q, a, d1, d2, d3 = "question_en", "answer_en", "distractor1_en", "distractor2_en", "distractor3_en"
    elif lang == "regional":
        q, a, d1, d2, d3 = "question", "answer", "distractor1", "distractor2", "distractor3"
    else:
        raise ValueError(f"lang={lang} is not supported.")

    data = ds["train"]

    if max_results:
        data = data.select(range(min(max_results, len(data))))

    results = []
    correct = 0
    total = 0
    total_err = 0

    if prompt_template:
        with open(prompt_template, "r", encoding="utf-8") as f:
            prompt_template = f.read()
    else:
        prompt_template = DEFAULT_PROPMT_TEMPLATE

    # Phase 1 — prepare every question deterministically (sequential).
    # `shuffle_options` seeds the global RNG, so this step must not run
    # concurrently or the shuffles would race and become non-reproducible.
    tasks = []
    for item in data:
        row_seed = seed + hash(str(item["article_id"])) % 10000
        options, correct_letter = shuffle_options(item[a], item[d1], item[d2], item[d3], row_seed)
        tasks.append({"item": item, "question": item[q], "options": options, "correct_letter": correct_letter})

    # Phase 2 — obtain a model answer for every task. Endpoints exposing an
    # OpenAI-compatible Batch API (e.g. Maritaca) submit ALL questions as one
    # asynchronous batch job — this dodges the rate limits that firing many
    # concurrent live requests would hit, and is ~50% cheaper. Every other
    # endpoint fans the calls out across a thread pool, at most `batch_size` in
    # flight. Either way `responses` ends up aligned with `tasks` (dataset order).
    model_tag = sanitize(model)
    if is_batch_endpoint(llm_uri):
        logger.info(
            f"Endpoint «{llm_uri}» supports the Batch API; submitting {len(tasks)} questions as one "
            f"batch job (--batch_size and --num_retries do not apply to this path)."
        )
        responses = _run_batch_requests(
            model,
            prompt_template,
            tasks,
            temperature,
            llm_api_key=llm_api_key,
            llm_uri=llm_uri,
            results_dir=results_dir,
            tag=f"{region}_{lang}_{model_tag}",
            poll_interval=batch_poll_interval,
        )
    else:
        responses = _run_live_requests(
            model,
            prompt_template,
            tasks,
            temperature,
            llm_api_key=llm_api_key,
            llm_uri=llm_uri,
            num_retries=num_retries,
            batch_size=batch_size,
            desc=f"Evaluating «{dataset_name}» (lang={lang})",
        )

    # Phase 3 — assemble results in the original dataset order (sequential).
    for task, response in zip(tasks, responses):
        item = task["item"]
        options = task["options"]
        correct_letter = task["correct_letter"]

        if isinstance(response, Exception) or response is None:
            model_response = None
            predicted = None
            is_correct = False
            total_err += 1
        else:
            model_response = response
            predicted = extract_answer(response)
            is_correct = predicted == correct_letter
            if is_correct:
                correct += 1
            total += 1

        results.append(
            {
                "article_id": item["article_id"],
                "question": item["question"],
                "correct_answer": item["answer"],
                "option_A": options[0],
                "option_B": options[1],
                "option_C": options[2],
                "option_D": options[3],
                "correct_letter": correct_letter,
                "model_response": model_response,
                "predicted_letter": predicted,
                "is_correct": is_correct,
            }
        )

    accuracy = correct / total if total > 0 else 0

    df_results = pd.DataFrame(results)
    out_name = f"mcq_eval_results_{region}_{lang}_{model_tag}.csv"
    out_path = Path(results_dir) / out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(out_path, index=False, encoding="utf-8")
    logger.info(f"Results saved to: {out_path}")

    summary = {
        "model": model,
        "region": region,
        "lang": lang,
        "total": total,
        "correct": correct,
        "errors": total_err,
        "accuracy": accuracy,
    }
    summary_path = Path(results_dir) / f"mcq_eval_summary_{region}_{lang}_{model_tag}.txt"
    with open(summary_path, "w") as f:
        lines = [f"{k}: {v}\n" for k, v in summary.items()]
        f.writelines(lines)
    logger.info(f"Summary saved to: {summary_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(
        prog="eval_mcq",
        description="LatamQA evaluation script for MCQ datasets.",
        epilog="See <https://github.com/Inria-Chile/LatamQA> for additional information.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model name to evaluate (e.g., 'gpt-4o', 'ollama/llama3.1', 'anthropic/claude-3') see <https://docs.litellm.ai/docs/providers> for details.",  # noqa: E501
    )
    parser.add_argument("--region", choices=REGIONAL_DATASETS, default="es-la", help="Regional dataset selection")
    parser.add_argument(
        "--lang",
        choices=TARGET_LANGUAGES,
        default="regional",
        help="Language: 'regional' for regional language, 'english' for English",
    )
    parser.add_argument("--max_results", type=int, default=None, help="Maximum number of rows to process")
    parser.add_argument("--seed", type=int, default=42, help="Seed for shuffling options")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Number of concurrent requests kept in flight (1 = sequential)",
    )
    parser.add_argument(
        "--num_retries",
        type=int,
        default=DEFAULT_NUM_RETRIES,
        help="LiteLLM retries for transient failures (rate limits, timeouts)",
    )
    parser.add_argument(
        "--batch_poll_interval",
        type=int,
        default=DEFAULT_BATCH_POLL_INTERVAL,
        help=(
            "Seconds between status checks for endpoints that use the async Batch API "
            "(e.g. Maritaca). Ignored for ordinary live-request providers."
        ),
    )
    parser.add_argument("--prompt_template", type=str, default=None, help="File name of custom prompt template")
    parser.add_argument("--results_dir", type=str, default=DEFAULT_RESULTS_DIR, help="Folder for storing results")
    parser.add_argument(
        "--llm_api_key",
        type=str,
        default=None,
        help="API key for LLM (if needed)",
    )
    parser.add_argument(
        "--llm_uri",
        type=str,
        default=None,
        help="URL for local/custom LLM provider (if needed)",
    )

    args = parser.parse_args()

    logger.info("LatamQA multiple choice question (MCQ) evaluation (Universal)")

    console = Console()

    table: list[tuple[str, float | int | str]] = [
        ("Model", args.model),
        ("Region", args.region),
        ("Language", args.lang),
        ("Temperature", args.temperature),
        ("Seed", args.seed),
        ("Batch size", args.batch_size),
        ("Results directory", args.results_dir),
    ]

    if args.llm_api_key:
        table.append(("LLM API key", "Provided"))
    if args.llm_uri:
        table.append(("LLM URI", args.llm_uri))
    if args.max_results:
        table.append(("Max. results", args.max_results))

    if args.prompt_template:
        table.append(("Custom prompt template", args.prompt_template))

    config_df = pd.DataFrame(table, columns=["Configuration", "Value"])
    console.print(Markdown("### Configuration"))
    console.print(Markdown(config_df.to_markdown(index=False)))

    summary = run_evaluation(
        model=args.model,
        region=args.region,
        lang=args.lang,
        max_results=args.max_results,
        seed=args.seed,
        temperature=args.temperature,
        prompt_template=args.prompt_template,
        results_dir=args.results_dir,
        llm_api_key=args.llm_api_key,
        llm_uri=args.llm_uri,
        batch_size=args.batch_size,
        num_retries=args.num_retries,
        batch_poll_interval=args.batch_poll_interval,
    )

    results_table = [
        ("Model", args.model),
        ("Region", args.region),
        ("Language", args.lang),
        ("Temperature", args.temperature),
        ("Seed", args.seed),
        ["Total", summary["total"]],
        ["Correct", summary["correct"]],
        ["Errors", summary["errors"]],
        ["Accuracy", f"{summary['accuracy']:.6%}"],
    ]

    results_df = pd.DataFrame(results_table, columns=["Metric", "Value"])
    console.print(Markdown("### Evaluation results"))
    console.print(Markdown(results_df.to_markdown(index=False)))


if __name__ == "__main__":
    main()
