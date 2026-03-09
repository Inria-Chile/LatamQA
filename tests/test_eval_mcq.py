from latamqa.eval_mcq import extract_answer, sanitize, shuffle_options


def test_sanitize():
    assert sanitize("gpt-4o") == "gpt-4o"
    assert sanitize("openai/gpt-4o") == "openai-gpt-4o"
    assert sanitize("http://localhost:11434") == "http---localhost-11434"


def test_shuffle_options():
    answer = "Rio de Janeiro"
    d1 = "São Paulo"
    d2 = "Brasília"
    d3 = "Salvador"
    seed = 42

    options, correct_letter = shuffle_options(answer, d1, d2, d3, seed)

    assert len(options) == 4
    assert answer in options
    assert d1 in options
    assert d2 in options
    assert d3 in options
    assert options[ord(correct_letter) - ord("A")] == answer


def test_extract_answer():
    assert extract_answer("A") == "A"
    assert extract_answer("The answer is (B)") == "B"
    assert extract_answer("C. Brasília") == "C"
    assert extract_answer("Option D: Salvador") == "D"
    assert extract_answer("None of the above") is None
    # In eval_mcq.py, extract_answer("a") returns "A" because it does .upper()
    # UNLESS it's a standalone "a" matched by \b([ABCD])\b.
    # Looking at the code:
    # response = response.upper().strip()
    # if response and response[0] in "ABCD": return response[0]
    # So "a" -> "A" -> response[0] is "A" -> returns "A".
    assert extract_answer("a") == "A"

    # Let's test the lowercase 'a' exception logic:
    # if match.group(1) == "A":
    #     start, end = match.span()
    #     if response_ini[start:end] == "a":
    #         return None
    # This only triggers if the FIRST check fails.
    # For "the answer is a", response[0] is 'T', not in 'ABCD'.
    # re.search(r"\b([ABCD])\b", "THE ANSWER IS A") matches 'A'.
    # response_ini["the answer is a"[14:15]] is 'a'.
    # So it should return None.
    assert extract_answer("the answer is a") is None
    assert extract_answer("the answer is A") == "A"
