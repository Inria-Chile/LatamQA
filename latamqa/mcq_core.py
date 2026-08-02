#!/usr/bin/env python3
"""Leaf module holding the MCQ evaluation core: prompt template, letter parser,
prompt builder, deterministic option shuffle, and the single-question LLM call.

These primitives were previously defined in :mod:`latamqa.eval_mcq`, which also
pulls in ``pandas``, ``datasets``, ``structlog``, ``rich`` and ``tqdm`` at import
time (for the dataset-driven batch path). Downstream consumers that only need to
score one question — a programmatic caller, a lightweight service, a test — should
not have to drag that whole closure in. Extracting the core here keeps
``import latamqa.mcq_core`` dependency-free (stdlib only): LiteLLM is imported
lazily inside :func:`evaluate_mcq`, at the point of the actual network call, so
merely importing the parser/template/prompt-builder costs nothing.

:mod:`latamqa.eval_mcq` re-exports every public name defined here, so existing
imports (``from latamqa.eval_mcq import evaluate_mcq, extract_answer, ...``) keep
working unchanged.
"""

import random
import re
from typing import Any, List, Mapping, Tuple

DEFAULT_PROPMT_TEMPLATE: str = """"Answer the following multiple-choice question by selecting ONLY the letter (A, B, C, or D) of the correct answer.

Question: {question}

A) {option_a}
B) {option_b}
C) {option_c}
D) {option_d}

Answer:
"""  # noqa: E501

# LiteLLM-side retries for transient failures (rate limits, timeouts, blips).
DEFAULT_NUM_RETRIES = 3
# Completion-token cap per question. Only a letter (A/B/C/D) is needed, but this
# budget must also cover any hidden reasoning a "thinking" model emits before its
# answer — too low a cap truncates those models to an empty response. Plain models
# are unaffected: they stop right after the letter, so the higher ceiling is free.
DEFAULT_MAX_TOKENS = 2048


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

    LiteLLM is imported here rather than at module top so that importing the
    parser/prompt helpers stays free of the (heavy) LiteLLM dependency closure.
    """
    from litellm import completion

    prompt = build_prompt(prompt_template, question, options)

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": DEFAULT_MAX_TOKENS,  # a letter, plus room for a thinking model's reasoning
        "num_retries": num_retries,
    }

    if llm_api_key:
        kwargs["api_key"] = llm_api_key
    if llm_uri:
        kwargs["api_base"] = llm_uri

    resp = completion(**kwargs)
    return resp.choices[0].message.content.strip()  # type: ignore


def evaluate_mcq_dict(
    model: str,
    question: Mapping[str, Any],
    temperature: float = 0.0,
    prompt_template: str | None = None,
    llm_api_key: str | None = None,
    llm_uri: str | None = None,
    num_retries: int = DEFAULT_NUM_RETRIES,
) -> str:
    """Answer a single MCQ supplied as a question dict and return the raw response.

    This is the programmatic entry point for callers that hold a question as a
    mapping rather than as separate positional arguments. ``question`` must
    provide:

    - ``"question"``: the question stem (``str``).
    - ``"options"``: the four answer options, already ordered A, B, C, D
      (a sequence of exactly four ``str``).

    ``prompt_template`` defaults to :data:`DEFAULT_PROPMT_TEMPLATE` (the D44
    English instruction template); the remaining keyword arguments mirror
    :func:`evaluate_mcq`. The model's raw text is returned unscored — pass it to
    :func:`extract_answer` to recover the predicted letter.

    Raises ``KeyError`` if a required field is missing and ``ValueError`` if
    ``options`` does not hold exactly four entries, so a malformed question fails
    loudly at the boundary instead of producing a silently wrong prompt.
    """
    template = prompt_template if prompt_template is not None else DEFAULT_PROPMT_TEMPLATE
    stem = question["question"]
    options = list(question["options"])
    if len(options) != 4:
        raise ValueError(
            f"evaluate_mcq_dict expects exactly 4 options (A-D), got {len(options)}: {options!r}"
        )
    return evaluate_mcq(
        model=model,
        prompt_template=template,
        question=stem,
        options=options,
        temperature=temperature,
        llm_api_key=llm_api_key,
        llm_uri=llm_uri,
        num_retries=num_retries,
    )
