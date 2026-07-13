import pytest
import yaml

from latamqa.model_eval import MODELS_DIR, load_models
from latamqa.model_schema import MODEL_TYPES, validate_model_config

# A minimal config that satisfies every required field; individual tests mutate a copy.
VALID = {
    "LiteLLM model name": "openai/gpt-4o",
    "Model name": "GPT-4o",
    "Model URL": "https://openai.com/gpt-4o",
    "Model size": "large",
    "Model type": "large",
}


def _with(**overrides):
    return {**VALID, **overrides}


def test_minimal_valid_config_has_no_errors():
    assert validate_model_config(VALID) == []


def test_all_optional_fields_valid():
    cfg = _with(
        **{
            "Paper URL": "https://arxiv.org/abs/2407.21783",
            "LLM URI": "http://localhost:8000/v1",
            "Comments": "note",
            "Batch size": 8,
            "Number of retries": 0,
        }
    )
    assert validate_model_config(cfg) == []


@pytest.mark.parametrize("missing", list(VALID))
def test_each_required_field_is_enforced(missing):
    cfg = {k: v for k, v in VALID.items() if k != missing}
    errors = validate_model_config(cfg)
    assert any("required property" in e and missing in e for e in errors), errors


def test_model_type_must_be_in_enum():
    errors = validate_model_config(_with(**{"Model type": "huge"}))
    assert errors and "is not one of" in errors[0]


def test_model_types_constant_matches_schema_enum():
    # The leaderboard sorts by this list; it must stay in lockstep with the enum.
    assert MODEL_TYPES == ["small", "medium", "large"]


def test_unknown_key_is_rejected():
    # Guards against typos like "Model typ" silently doing nothing.
    errors = validate_model_config(_with(**{"Model typ": "large"}))
    assert errors and "Additional properties" in errors[0]


def test_url_without_scheme_is_rejected():
    assert validate_model_config(_with(**{"Model URL": "example.com"}))


@pytest.mark.parametrize("value", ["lots", True, 0, -1, 1.5])
def test_batch_size_rejects_non_positive_int(value):
    # Strings, bools, floats and values < 1 are all invalid.
    assert validate_model_config(_with(**{"Batch size": value}))


def test_batch_size_accepts_positive_int():
    assert validate_model_config(_with(**{"Batch size": 32})) == []


@pytest.mark.parametrize("value", ["lots", True, -1, 1.5])
def test_num_retries_rejects_bad_values(value):
    assert validate_model_config(_with(**{"Number of retries": value}))


def test_num_retries_accepts_zero():
    assert validate_model_config(_with(**{"Number of retries": 0})) == []


def test_empty_document_is_rejected():
    # yaml.safe_load of an empty file yields None.
    errors = validate_model_config(None)
    assert errors and "is not of type 'object'" in errors[0]


def test_show_in_leaderboard_boolean_is_allowed():
    # leaderboard.py filters rows on this flag, so the schema must permit it.
    assert validate_model_config(_with(**{"show_in_leaderboard": False})) == []


def test_show_in_leaderboard_must_be_boolean():
    assert validate_model_config(_with(**{"show_in_leaderboard": "no"}))


def test_all_shipped_model_configs_are_valid():
    # The real configs in latamqa/models/ must always satisfy the schema.
    for model_file in sorted(MODELS_DIR.glob("*.yaml")):
        assert validate_model_config(yaml.safe_load(model_file.read_text())) == [], model_file.name


def _write(dir_path, name, config):
    (dir_path / name).write_text(yaml.safe_dump(config), encoding="utf-8")


def test_load_models_returns_valid_configs(tmp_path):
    _write(tmp_path, "good.yaml", VALID)
    _write(tmp_path, "good2.yaml", _with(**{"Model name": "Other", "Batch size": 4}))
    models = load_models(tmp_path)
    assert set(models) == {"good", "good2"}
    assert models["good2"]["Batch size"] == 4


def test_load_models_stops_on_invalid_config(tmp_path):
    _write(tmp_path, "good.yaml", VALID)
    _write(tmp_path, "bad.yaml", {k: v for k, v in VALID.items() if k != "Model type"})
    with pytest.raises(SystemExit):
        load_models(tmp_path)


def test_load_models_reports_all_bad_files(tmp_path, capsys):
    _write(tmp_path, "bad1.yaml", {k: v for k, v in VALID.items() if k != "Model type"})
    _write(tmp_path, "bad2.yaml", _with(**{"Model type": "huge"}))
    with pytest.raises(SystemExit):
        load_models(tmp_path)
    # Both offending files should be named in the single aggregated fatal message.
    out = capsys.readouterr().out
    assert "bad1.yaml" in out and "bad2.yaml" in out


def test_load_models_handles_unparseable_yaml(tmp_path, capsys):
    # A syntactically broken YAML must produce a clean, file-pointing error
    # rather than a raw PyYAML traceback.
    _write(tmp_path, "good.yaml", VALID)
    (tmp_path / "broken.yaml").write_text("key: [unclosed\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        load_models(tmp_path)
    out = capsys.readouterr().out
    assert "broken.yaml" in out and "could not parse YAML" in out
