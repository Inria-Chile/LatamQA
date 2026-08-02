"""Tests for the dependency-light MCQ core (`latamqa.mcq_core`).

These cover the letter parser, prompt builder, prompt template, deterministic
option shuffle, and the new question-dict entry point. The single LLM call is
exercised with a fake `litellm` module injected into `sys.modules`, so no
network access (and no LiteLLM install) is required.
"""

import sys
import types

import pytest

from latamqa.mcq_core import (
    DEFAULT_PROPMT_TEMPLATE,
    build_prompt,
    evaluate_mcq_dict,
    extract_answer,
    shuffle_options,
)


def test_extract_answer():
    # Mirrors latamqa.eval_mcq's original test_extract_answer so the extracted
    # parser is provably identical (D44 fidelity by construction).
    assert extract_answer("A") == "A"
    assert extract_answer("The answer is (B)") == "B"
    assert extract_answer("C. Brasília") == "C"
    assert extract_answer("Option D: Salvador") == "D"
    assert extract_answer("None of the above") is None
    assert extract_answer("a") == "A"
    assert extract_answer("the answer is a") is None
    assert extract_answer("the answer is A") == "A"


def test_extract_answer_negatives():
    assert extract_answer("") is None
    assert extract_answer("   ") is None
    assert extract_answer("I am not sure") is None


def test_shuffle_options_is_deterministic():
    answer, d1, d2, d3, seed = "Rio de Janeiro", "São Paulo", "Brasília", "Salvador", 42
    options, correct_letter = shuffle_options(answer, d1, d2, d3, seed)

    assert len(options) == 4
    assert {answer, d1, d2, d3} == set(options)
    assert options[ord(correct_letter) - ord("A")] == answer
    # Same seed -> same shuffle.
    assert shuffle_options(answer, d1, d2, d3, seed) == (options, correct_letter)


def test_build_prompt_substitutes_all_fields():
    prompt = build_prompt(DEFAULT_PROPMT_TEMPLATE, "What is 2+2?", ["3", "4", "5", "6"])
    assert "What is 2+2?" in prompt
    assert "A) 3" in prompt
    assert "B) 4" in prompt
    assert "C) 5" in prompt
    assert "D) 6" in prompt
    # No unfilled placeholders remain.
    for placeholder in ("{question}", "{option_a}", "{option_b}", "{option_c}", "{option_d}"):
        assert placeholder not in prompt


def test_default_template_is_the_english_letter_only_instruction():
    # The D44 template: English, letter-only (A/B/C/D) instruction.
    assert "multiple-choice question" in DEFAULT_PROPMT_TEMPLATE
    assert "ONLY the letter (A, B, C, or D)" in DEFAULT_PROPMT_TEMPLATE
    assert "Answer:" in DEFAULT_PROPMT_TEMPLATE


def _install_fake_litellm(monkeypatch, content: str, sink: dict | None = None):
    """Inject a fake `litellm` module whose `completion` records kwargs and
    returns an OpenAI-shaped response carrying ``content``."""
    fake = types.ModuleType("litellm")

    def completion(**kwargs):
        if sink is not None:
            sink.update(kwargs)
        message = types.SimpleNamespace(content=content)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])

    fake.completion = completion
    monkeypatch.setitem(sys.modules, "litellm", fake)


def test_evaluate_mcq_dict_builds_prompt_and_returns_stripped_response(monkeypatch):
    sink: dict = {}
    _install_fake_litellm(monkeypatch, content="  B) São Paulo  ", sink=sink)

    question = {"question": "Capital of Brazil?", "options": ["Rio", "Brasília", "SP", "Salvador"]}
    out = evaluate_mcq_dict(
        "openai/some-model",
        question,
        temperature=0.0,
        llm_uri="http://127.0.0.1:9/v1",
        llm_api_key="dummy",
    )

    assert out == "B) São Paulo"  # stripped
    assert extract_answer(out) == "B"
    # The dict entry point routed through evaluate_mcq -> litellm.completion with
    # the expected OpenAI-compatible kwargs.
    assert sink["model"] == "openai/some-model"
    assert sink["api_base"] == "http://127.0.0.1:9/v1"
    assert sink["api_key"] == "dummy"
    assert sink["messages"][0]["role"] == "user"
    assert "Capital of Brazil?" in sink["messages"][0]["content"]
    assert "A) Rio" in sink["messages"][0]["content"]


def test_evaluate_mcq_dict_defaults_to_the_d44_template(monkeypatch):
    sink: dict = {}
    _install_fake_litellm(monkeypatch, content="A", sink=sink)
    evaluate_mcq_dict("openai/m", {"question": "Q?", "options": ["a", "b", "c", "d"]})
    prompt = sink["messages"][0]["content"]
    assert "ONLY the letter (A, B, C, or D)" in prompt


def test_evaluate_mcq_dict_rejects_wrong_option_count(monkeypatch):
    _install_fake_litellm(monkeypatch, content="A")
    with pytest.raises(ValueError):
        evaluate_mcq_dict("openai/m", {"question": "Q?", "options": ["a", "b", "c"]})


def test_evaluate_mcq_dict_requires_question_and_options(monkeypatch):
    _install_fake_litellm(monkeypatch, content="A")
    with pytest.raises(KeyError):
        evaluate_mcq_dict("openai/m", {"options": ["a", "b", "c", "d"]})
    with pytest.raises(KeyError):
        evaluate_mcq_dict("openai/m", {"question": "Q?"})
