import pytest

from latamqa.model_eval import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_NUM_RETRIES,
    resolve_model_options,
)

# No option set on the command line (every flag omitted).
NO_CLI = {"batch_size": None, "num_retries": None, "llm_uri": None}


def test_defaults_when_neither_source_sets_the_option():
    opts = resolve_model_options("m", {}, dict(NO_CLI))
    assert opts == {"batch_size": DEFAULT_BATCH_SIZE, "num_retries": DEFAULT_NUM_RETRIES, "llm_uri": None}


def test_cli_only():
    opts = resolve_model_options("m", {}, {"batch_size": 32, "num_retries": 5, "llm_uri": "http://localhost:8000/v1"})
    assert opts == {"batch_size": 32, "num_retries": 5, "llm_uri": "http://localhost:8000/v1"}


def test_yaml_only():
    model = {"Batch size": 8, "Number of retries": 1, "LLM URI": "https://api.example.com"}
    opts = resolve_model_options("m", model, dict(NO_CLI))
    assert opts == {"batch_size": 8, "num_retries": 1, "llm_uri": "https://api.example.com"}


def test_cli_and_yaml_mix_across_different_options():
    # batch_size from the YAML, num_retries from the CLI -- no clash since each
    # option is set in exactly one place.
    model = {"Batch size": 8}
    opts = resolve_model_options("m", model, {"batch_size": None, "num_retries": 7, "llm_uri": None})
    assert opts == {"batch_size": 8, "num_retries": 7, "llm_uri": None}


def test_clash_stops_the_run():
    model = {"Batch size": 8}
    with pytest.raises(SystemExit):
        resolve_model_options("m", model, {"batch_size": 32, "num_retries": None, "llm_uri": None})


def test_clash_stops_even_when_values_are_equal():
    # Ambiguous source of truth is still an error, even if the values agree.
    model = {"Batch size": 32}
    with pytest.raises(SystemExit):
        resolve_model_options("m", model, {"batch_size": 32, "num_retries": None, "llm_uri": None})


def test_invalid_yaml_type_stops_the_run():
    model = {"Batch size": "lots"}
    with pytest.raises(SystemExit):
        resolve_model_options("m", model, dict(NO_CLI))


def test_yaml_bool_is_rejected():
    # `bool` is an `int` subclass, but "Batch size: true" must not become 1.
    model = {"Batch size": True}
    with pytest.raises(SystemExit):
        resolve_model_options("m", model, dict(NO_CLI))


def test_batch_size_below_minimum_stops_the_run():
    model = {"Batch size": 0}
    with pytest.raises(SystemExit):
        resolve_model_options("m", model, dict(NO_CLI))


def test_negative_cli_value_stops_the_run():
    with pytest.raises(SystemExit):
        resolve_model_options("m", {}, {"batch_size": -4, "num_retries": None, "llm_uri": None})


def test_num_retries_zero_is_valid():
    model = {"Number of retries": 0}
    opts = resolve_model_options("m", model, dict(NO_CLI))
    assert opts["num_retries"] == 0


def test_integer_valued_float_is_accepted_and_coerced():
    # `Batch size: 8.0` passes the JSON Schema "integer" type, so the resolver
    # must agree (accept it, coerced to int) instead of aborting mid-run.
    model = {"Batch size": 8.0, "Number of retries": 3.0}
    opts = resolve_model_options("m", model, dict(NO_CLI))
    assert opts["batch_size"] == 8 and isinstance(opts["batch_size"], int)
    assert opts["num_retries"] == 3 and isinstance(opts["num_retries"], int)


def test_non_integer_float_still_stops_the_run():
    model = {"Batch size": 8.5}
    with pytest.raises(SystemExit):
        resolve_model_options("m", model, dict(NO_CLI))


# --- LLM URI follows the same CLI-xor-YAML clash rule as the numeric options ---


def test_llm_uri_cli_only():
    opts = resolve_model_options("m", {}, {"batch_size": None, "num_retries": None, "llm_uri": "http://cli"})
    assert opts["llm_uri"] == "http://cli"


def test_llm_uri_yaml_only():
    model = {"LLM URI": "https://api.example.com"}
    opts = resolve_model_options("m", model, dict(NO_CLI))
    assert opts["llm_uri"] == "https://api.example.com"


def test_llm_uri_clash_stops_the_run():
    model = {"LLM URI": "https://api.example.com"}
    with pytest.raises(SystemExit):
        resolve_model_options("m", model, {"batch_size": None, "num_retries": None, "llm_uri": "http://cli"})


def test_llm_uri_clash_stops_even_when_values_are_equal():
    model = {"LLM URI": "http://same"}
    with pytest.raises(SystemExit):
        resolve_model_options("m", model, {"batch_size": None, "num_retries": None, "llm_uri": "http://same"})


def test_llm_uri_empty_yaml_value_stops_the_run():
    # A blank/whitespace-only "LLM URI" is not a usable endpoint.
    model = {"LLM URI": "   "}
    with pytest.raises(SystemExit):
        resolve_model_options("m", model, dict(NO_CLI))


def test_llm_uri_non_string_yaml_value_stops_the_run():
    model = {"LLM URI": 8000}
    with pytest.raises(SystemExit):
        resolve_model_options("m", model, dict(NO_CLI))


def test_llm_uri_defaults_to_none():
    opts = resolve_model_options("m", {}, dict(NO_CLI))
    assert opts["llm_uri"] is None
