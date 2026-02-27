#!/usr/bin/env python3
"""
Universal script to evaluate models through API (OpenAI, Mistral, Anthropic, Ollama, vLLM, etc.) using LiteLLM.
"""

import argparse
import random
import re
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import structlog
from datasets import load_dataset
from litellm import completion
from tabulate import tabulate
from tqdm.auto import tqdm

logger = structlog.get_logger()

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


def extract_answer(response: str) -> str:
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


def evaluate_mcq(
    model: str,
    prompt_template: str,
    question: str,
    options: List[str],
    temperature: float,
    api_key: str | None = None,
    base_url: str | None = None,
) -> str:
    """Ask the LLM to answer the MCQ via LiteLLM and return its response."""
    prompt = prompt_template.replace("{question}", question)
    prompt = prompt.replace("{option_a}", options[0])
    prompt = prompt.replace("{option_b}", options[1])
    prompt = prompt.replace("{option_c}", options[2])
    prompt = prompt.replace("{option_d}", options[3])

    kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": 10,  # We only need one letter
    }
    if api_key:
        kwargs["api_key"] = api_key
    if base_url:
        kwargs["api_base"] = base_url

    resp = completion(**kwargs)
    return resp.choices[0].message.content.strip()


def run_evaluation(
    model: str,
    region: str = "es-la",
    lang: str = "regional",
    max_results: int | None = None,
    seed: int = 42,
    temperature: float = 0.0,
    prompt_template_path: str | None = None,
    results_dir: str | Path = Path().cwd() / "results",
    llm_api_key: str | None = None,
    llm_url: str | None = None,
):
    """Run the MCQ evaluation."""
    logger.info("LatamQA multiple choice question (MCQ) evaluation (Universal)")

    table = [
        ["Model:", model],
        ["Region:", region],
        ["Language:", lang],
        ["Temperature:", temperature],
        ["Seed:", seed],
        ["LLM URL:", llm_url or "not set"],
    ]
    if max_results:
        table.append(["Max. results:", max_results])

    if prompt_template_path:
        table.append(["Custom prompt template:", prompt_template_path])

    logger.info("Configuration:\n" + tabulate(table, tablefmt="rounded_grid", colalign=["left", "right"]))

    dataset_name = f"inria-chile/latamqa_mcq_{region}"

    logger.info(f"Loading dataset «{dataset_name}».")
    ds = load_dataset(dataset_name)

    if lang == "en":
        q, a, d1, d2, d3 = "question_en", "answer_en", "distractor1_en", "distractor2_en", "distractor3_en"
    else:
        q, a, d1, d2, d3 = "question", "answer", "distractor1", "distractor2", "distractor3"

    data = ds["train"]

    if max_results:
        data = data.select(range(min(max_results, len(data))))

    results = []
    correct = 0
    total = 0
    total_err = 0

    if prompt_template_path:
        with open(prompt_template_path, "r", encoding="utf-8") as f:
            prompt_template = f.read()
    else:
        prompt_template = DEFAULT_PROPMT_TEMPLATE

    for item in tqdm(data, desc="Evaluating"):
        question = item[q]
        row_seed = seed + hash(str(item["article_id"])) % 10000
        options, correct_letter = shuffle_options(item[a], item[d1], item[d2], item[d3], row_seed)
        try:
            response = evaluate_mcq(
                model,
                prompt_template,
                question,
                options,
                temperature,
                api_key=llm_api_key,
                base_url=llm_url,
            )
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
                    "model_response": response,
                    "predicted_letter": predicted,
                    "is_correct": is_correct,
                }
            )

        except Exception as e:
            logger.error(f"article_id={item.get('article_id')}: {e}.")
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
                    "model_response": None,
                    "predicted_letter": None,
                    "is_correct": False,
                }
            )
            total_err += 1

    accuracy = correct / total if total > 0 else 0

    results_table = [
        ["Model:", model],
        ["Region:", region],
        ["Language:", lang],
        ["Total:", total],
        ["Correct:", correct],
        ["Errors:", total_err],
        ["Accuracy:", f"{accuracy:.6%}"],
    ]

    logger.info("Evaluation results:\n" + tabulate(results_table, tablefmt="rounded_grid", colalign=["left", "right"]))

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
    parser.add_argument("--lang", choices=TARGET_LANGUAGES, default="regional", help="Language: 'regional' for regional language, 'en' for English")  # noqa: E501
    parser.add_argument("--max_results", type=int, default=None, help="Maximum number of rows to process")
    parser.add_argument("--seed", type=int, default=42, help="Seed for shuffling options")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature")
    parser.add_argument("--prompt_template", type=str, default=None, help="File name of custom prompt template")
    parser.add_argument("--results_dir", type=str, default=Path().cwd() / "results", help="Folder for storing results")
    parser.add_argument(
        "--llm_api_key",
        type=str,
        default=None,
        help="API key for LLM (if needed)",
    )
    parser.add_argument(
        "--llm_url",
        type=str,
        default=None,
        help="URL for local/custom LLM provider (if needed)",
    )

    args = parser.parse_args()

    run_evaluation(
        model=args.model,
        region=args.region,
        lang=args.lang,
        max_results=args.max_results,
        seed=args.seed,
        temperature=args.temperature,
        prompt_template_path=args.prompt_template,
        results_dir=args.results_dir,
        llm_api_key=args.llm_api_key,
        llm_url=args.llm_url,
    )


if __name__ == "__main__":
    main()
