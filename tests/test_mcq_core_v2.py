"""Tests for the protocol v2 helpers in `latamqa.mcq_core` (stable option orders, prompt pieces, parser).

The golden values were produced by the pilot launcher that validated protocol v2 on Hugging Face Inference
Providers (Sep 2026), so these tests pin the port to the tested behaviour.
"""

import pytest

from latamqa.mcq_core import (
    DEFAULT_PROPMT_TEMPLATE,
    V2_MAX_TOKENS,
    V2_OPTION_ORDERS,
    V2_PROMPT_TEMPLATE,
    V2_SYSTEM_MESSAGE,
    extract_answer,
    has_think_tag,
    parse_answer_v2,
    permuted_options,
    question_id,
    shuffle_options,
)

OPTS = ("Santiago", "Lima", "Quito", "Bogotá")
GOLDEN = {  # permuted_options("12345:0a1b2c3d", *OPTS, order, 42), from the pilot launcher
    1: (["Lima", "Quito", "Santiago", "Bogotá"], "C"),
    2: (["Quito", "Santiago", "Bogotá", "Lima"], "B"),
    3: (["Santiago", "Bogotá", "Lima", "Quito"], "A"),
    4: (["Bogotá", "Lima", "Quito", "Santiago"], "D"),
    5: (["Quito", "Lima", "Bogotá", "Santiago"], "D"),
}


def test_v2_constants():
    assert V2_PROMPT_TEMPLATE == DEFAULT_PROPMT_TEMPLATE[1:]  # only the stray leading quote is dropped
    assert not V2_PROMPT_TEMPLATE.startswith('"')
    assert V2_SYSTEM_MESSAGE == "Answer with a single letter: A, B, C, or D."
    assert V2_MAX_TOKENS == 16
    assert V2_OPTION_ORDERS == 5


def test_question_id_is_stable():
    assert question_id(12345, "¿Cuál es la capital de Chile?") == "12345:1fba4ab0"
    assert question_id("12345", "¿Cuál es la capital de Chile?") == "12345:1fba4ab0"
    assert question_id(12345, "Another question") != "12345:1fba4ab0"


@pytest.mark.parametrize("order", sorted(GOLDEN))
def test_permuted_options_matches_the_pilot(order):
    assert permuted_options("12345:0a1b2c3d", *OPTS, order, 42) == GOLDEN[order]


def test_orders_1_to_4_are_cyclic_shifts_and_place_the_answer_once_per_slot():
    for qid in ("a:1", "b:2", "c:3", "d:4"):
        base, _ = permuted_options(qid, *OPTS, 1)
        letters = set()
        for order in range(1, 5):
            options, correct = permuted_options(qid, *OPTS, order)
            k = order - 1
            assert options == base[k:] + base[:k]
            assert options["ABCD".index(correct)] == "Santiago"
            letters.add(correct)
        assert letters == set("ABCD")  # averaging runs 1-4 cancels position bias


def test_permuted_options_does_not_touch_the_global_rng():
    import random

    random.seed(7)
    expected = random.random()
    random.seed(7)
    permuted_options("q", *OPTS, 5)
    assert random.random() == expected


def test_permuted_options_rejects_bad_orders():
    for order in (0, 6):
        with pytest.raises(ValueError):
            permuted_options("q", *OPTS, order)


def test_v1_shuffle_is_unchanged():
    # Protocol v1 (the leaderboard) must keep its exact behaviour (value taken from main before v2 was added).
    assert shuffle_options(*OPTS, 42) == (["Quito", "Lima", "Bogotá", "Santiago"], "D")


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("B", ("B", "only")),
        (" B.", ("B", "only")),
        ("(C)", ("C", "only")),
        ("**D**", ("D", "only")),
        ("A)", ("A", "only")),
        ("C) Brasília", ("C", "lead")),
        ("B) São Paulo, capital do estado", ("B", "lead")),
        ("La respuesta correcta es D.", ("D", "cue")),
        ("A resposta correta é B", ("B", "cue")),
        ("Creo que la respuesta es A", ("A", "cue")),
        ("The answer is (C)", ("C", "cue")),
        ("", (None, "none")),
        (None, (None, "none")),
        ("I am not sure", (None, "none")),
        ("A resposta não está clara", (None, "none")),
    ],
)
def test_parse_answer_v2(reply, expected):
    assert parse_answer_v2(reply) == expected


def test_parse_answer_v2_fixes_the_v1_first_character_rule():
    # v1 reads the first character of a sentence as the answer; v2 does not.
    assert extract_answer("Creo que la respuesta es A") == "C"
    assert parse_answer_v2("Creo que la respuesta es A") == ("A", "cue")
    assert extract_answer("A resposta correta é D") == "A"
    assert parse_answer_v2("A resposta correta é D") == ("D", "cue")


def test_has_think_tag():
    assert has_think_tag("<think>hmm</think>B")
    assert has_think_tag("hmm</think> B")
    assert not has_think_tag("B")
    assert not has_think_tag(None)
