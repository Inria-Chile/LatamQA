#!/usr/bin/env python3
"""
Universal script to evaluate models through API (OpenAI, Mistral, Anthropic, Ollama, vLLM, etc.) using LiteLLM.
"""

import argparse
import os
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, List, Tuple

import pandas as pd
import structlog
from datasets import load_dataset
from litellm import completion
from litellm.exceptions import AuthenticationError, BadRequestError
from rich.console import Console
from rich.markdown import Markdown
from tqdm.auto import tqdm

logger = structlog.get_logger()

DEFAULT_RESULTS_DIR = Path(__file__).parent.parent / "results"

DEFAULT_PROPMT_TEMPLATE: str = """"Answer the following multiple-choice question by selecting ONLY the letter (A, B, C, or D) of the correct answer.

Question: {question}

A) {option_a}
B) {option_b}
C) {option_c}
D) {option_d}

Answer:
"""  # noqa: E501

REGIONAL_DATASETS = ["es-la", "es-es", "pt-br"]
TARGET_LANGUAGES = ["regional", "english"]

# Number of MCQ requests kept in flight at once. Concurrency turns the
# otherwise-sequential eval loop into a batch of parallel calls, which is a
# large speedup against API providers and lets a self-hosted vLLM server use
# its continuous batching.
DEFAULT_BATCH_SIZE = 16
# LiteLLM-side retries for transient failures (rate limits, timeouts, blips).
DEFAULT_NUM_RETRIES = 3


def sanitize(s: str) -> str:
    """Sanitize string for use in filenames."""
    return s.replace("/", "-").replace(":", "-")


def shuffle_options(
    answer: str,
    d1: str,
    d2: str,
    d3: str,
    seed: int,
) -> Tuple[List[str], str]:
    """Shuffle options and return (options_list, correct_letter)."""
    options = [
        ("answer", answer),
        ("d1", d1),
        ("d2", d2),
        ("d3", d3),
    ]
    random.seed(seed)
    random.shuffle(options)

    correct_idx = next(i for i, (label, _) in enumerate(options) if label == "answer")
    correct_letter = chr(ord("A") + correct_idx)

    return [opt[1] for opt in options], correct_letter


def extract_answer(response: str) -> str | None:
    """Extract the letter (A, B, C, D) from the model's response."""
    response_ini = response.strip()
    response = response.upper().strip()
    if response and response[0] in "ABCD":
        return response[0]
    match = re.search(r"\b([ABCD])[\).]", response)
    if match:
        return match.group(1)
    match = re.search(r"\b([ABCD])\b", response)
    if match:
        if match.group(1) == "A":
            start, end = match.span()
            if response_ini[start:end] == "a":
                return None
            else:
                return "A"
        else:
            return match.group(1)
    return None


def build_prompt(prompt_template: str, question: str, options: List[str]) -> str:
    """Fill the prompt template with the question and its (shuffled) options."""
    prompt = prompt_template.replace("{question}", question)
    prompt = prompt.replace("{option_a}", options[0])
    prompt = prompt.replace("{option_b}", options[1])
    prompt = prompt.replace("{option_c}", options[2])
    prompt = prompt.replace("{option_d}", options[3])
    return prompt


def evaluate_mcq(
    model: str,
    prompt_template: str,
    question: str,
    options: List[str],
    temperature: float,
    llm_api_key: str | None = None,
    llm_uri: str | None = None,
    num_retries: int = DEFAULT_NUM_RETRIES,
) -> str:
    """Ask the LLM to answer a single MCQ via LiteLLM and return its raw response.

    Transient failures (rate limits, timeouts, connection blips) are retried
    internally by LiteLLM up to ``num_retries`` times. Any error that survives
    the retries propagates to the caller, which decides whether it is fatal for
    the whole run or should be recorded as a single failed question. This keeps
    the function thread-safe so it can be fanned out across a thread pool.
    """
    prompt = build_prompt(prompt_template, question, options)

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": 10,  # We only need one letter
        "num_retries": num_retries,
    }

    if llm_api_key:
        kwargs["api_key"] = llm_api_key
    if llm_uri:
        kwargs["api_base"] = llm_uri

    resp = completion(**kwargs)
    return resp.choices[0].message.content.strip()  # type: ignore


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

    # Phase 2 — issue the LLM calls concurrently, at most `batch_size` in flight.
    # Each result is written back to its own slot so dataset order is preserved.
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
        for future in tqdm(
            as_completed(future_to_idx),
            total=len(tasks),
            desc=f"Evaluating «{dataset_name}» (lang={lang})",
            leave=False,
        ):
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
    model_tag = sanitize(model)
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
