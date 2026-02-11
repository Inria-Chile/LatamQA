#!/usr/bin/env python3
"""
Example of script to evaluate models through API (OpenAI and Mistral)
"""

import argparse
import os
import random
import re
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import structlog
from datasets import load_dataset
from tabulate import SEPARATING_LINE, tabulate
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

API_KEY: str = os.environ.get("API_LLM")
BASE_URL: str = os.environ.get("URL_LLM")


def sanitize(s: str) -> str:
    """Sanitize string for use in filenames."""
    return s.replace("/", "-").replace(":", "-")


def select_provider(provider):
    if provider == "mistral":
        try:
            from mistralai import Mistral

            if not API_KEY:
                logger.error("API_KEY enviroment variable is not set.")
                exit(1)

            client = Mistral(api_key=API_KEY)
        except ImportError:
            raise ImportError("mistralai package not installed.")
    elif provider == "openai":
        try:
            from openai import OpenAI

            if not API_KEY:
                logger.error("API_KEY enviroment variable is not set.")
                exit(1)
            if not BASE_URL:
                logger.error("BASE_URL enviroment variable is not set.")
                exit(1)
            client = OpenAI(
                api_key=API_KEY,
                base_url=BASE_URL,
            )
        except ImportError:
            raise ImportError("mistralai package not installed.")
    else:
        raise ValueError(f"Unsupported provider: {provider}")
    return client


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
    provider: str,
    question: str,
    options: List[str],
    temperature: float,
    client,
) -> str:
    """Ask the LLM to answer the MCQ and return its response."""
    prompt = prompt_template.replace("{question}", question)
    prompt = prompt.replace("{option_a}", options[0])
    prompt = prompt.replace("{option_b}", options[1])
    prompt = prompt.replace("{option_c}", options[2])
    prompt = prompt.replace("{option_d}", options[3])

    if provider == "mistral":
        resp = client.chat.complete(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=100,
        )
        return resp.choices[0].message.content.strip()
    else:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=100,
        )
        return resp.choices[0].message.content.strip()


def main():
    parser = argparse.ArgumentParser(description="LatamQA evaluation script")
    parser.add_argument("--model", required=True, help="Model name to evaluate")
    parser.add_argument("--provider", choices=["openai", "mistral"], default="openai", help="Model provider")
    parser.add_argument("--region", choices=["es-la", "es-es", "pt-br"], default="es-la", help="Dataset selection")
    parser.add_argument("--lang", choices=["o", "en"], default="o", help="Language: 'o' for original, 'en' for english")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of rows")
    parser.add_argument("--seed", type=int, default=42, help="Seed for shuffling options")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature")
    parser.add_argument("--prompt_template", type=str, default=None, help="File name of custom prompt template")
    args = parser.parse_args()

    print(f"\n{'=' * 60}")
    logger.info("LatamQA multiple choice question (MCQ) evaluation")

    table = [
        ["Model:", args.model],
        ["Provider:", args.provider],
        ["Region:", args.region],
        ["Language:", "english" if args.lang == "en" else "original"],
        ["Temperature:", args.temperature],
        ["Seed:", args.seed],
    ]
    if args.limit:
        table.append(["Limit:", args.limit])
    if args.prompt_template:
        table.append(["Custom prompt template:", args.prompt_template])

    logger.info(tabulate(table, tablefmt="rounded_grid", colalign=["left", "right"]))

    ds = load_dataset(f"inria-chile/latamqa_mcq_{args.region}")

    if args.lang == "en":
        q, a, d1, d2, d3 = "question_en", "answer_en", "distractor1_en", "distractor2_en", "distractor3_en"
    else:
        q, a, d1, d2, d3 = "question", "answer", "distractor1", "distractor2", "distractor3"

    client = select_provider(args.provider)

    data = ds["train"]
    if args.limit:
        data = data.select(range(min(args.limit, len(data))))

    results = []
    correct = 0
    total = 0
    total_err = 0

    if args.prompt_template:
        with open("prompt_eval.txt", "r", encoding="utf-8") as f:
            prompt_template = f.read()
    else:
        prompt_template = DEFAULT_PROPMT_TEMPLATE

    for item in tqdm(data, desc="Evaluating"):
        question = item[q]
        row_seed = args.seed + hash(str(item["article_id"])) % 10000
        options, correct_letter = shuffle_options(item[a], item[d1], item[d2], item[d3], row_seed)
        try:
            response = evaluate_mcq(
                args.model,
                prompt_template,
                args.provider,
                question,
                options,
                args.temperature,
                client,
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
            print(f"\n  [ERROR] article_id={item.get('article_id')}: {e}")
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
        ["Model:", args.model],
        ["Region:", args.region],
        ["Language:", "english" if args.lang == "en" else "original"],
        SEPARATING_LINE,
        ["Total:", total],
        ["Correct:", correct],
        ["Errors:", total_err],
        ["Accuracy:", f"{accuracy:.2%}"],
    ]

    logger.info("Evaluation results:")
    logger.info(tabulate(results_table, tablefmt="rounded_grid", colalign=["left", "right"]))

    df_results = pd.DataFrame(results)
    model_tag = sanitize(args.model)
    out_name = f"mcq_eval_results_{args.region}_{args.lang}_{model_tag}.csv"
    out_path = Path("results") / out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(out_path, index=False, encoding="utf-8")
    print(f"\n  Results saved to: {out_path}")

    summary = {
        "model": args.model,
        "region": args.region,
        "lang": args.lang,
        "total": total,
        "correct": correct,
        "errors": total_err,
        "accuracy": accuracy,
    }
    summary_path = Path("results") / f"mcq_eval_summary_{args.region}_{args.lang}_{model_tag}.txt"
    with open(summary_path, "w") as f:
        lines = [f"{k}: {v}" for k, v in summary.items()]
        f.writelines(lines)
    logger.info(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
