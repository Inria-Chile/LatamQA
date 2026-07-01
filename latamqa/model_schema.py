"""JSON Schema for the per-model configuration files in ``latamqa/models/``.

Each ``*.yaml`` in that directory describes one evaluable model. The schema
below is the single source of truth for which fields are allowed, which are
required, and what their types/ranges are. Both ``model_eval`` and
``leaderboard`` validate configs against it (via
:func:`latamqa.model_eval.load_models`), so a malformed YAML is caught up front
with a clear message instead of surfacing as a confusing error deep in a run.
"""

from jsonschema import Draft202012Validator

# Coarse model-size categories, ordered small -> large. Kept here so the allowed
# set (schema enum) and the leaderboard sort order share a single definition.
MODEL_TYPES = ["small", "medium", "large"]

# URL fields must be absolute http(s) URLs. Deliberately loose: this guards
# against obvious mistakes (a missing scheme, a stray value) without pulling in a
# full RFC 3986 validator.
_URL_PATTERN = r"^https?://"

MODEL_CONFIG_SCHEMA: dict = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "LatamQA model configuration",
    "description": "One evaluable model, as stored in latamqa/models/<model-id>.yaml.",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "LiteLLM model name",
        "Model name",
        "Model URL",
        "Model size",
        "Model type",
    ],
    "properties": {
        "LiteLLM model name": {
            "type": "string",
            "minLength": 1,
            "description": "Identifier passed to litellm.completion(), e.g. 'gpt-4o' or 'ollama/llama3'.",
        },
        "Model name": {
            "type": "string",
            "minLength": 1,
            "description": "Human-readable display name; keys the leaderboard.",
        },
        "Model URL": {
            "type": "string",
            "pattern": _URL_PATTERN,
            "description": "Link to the model's page.",
        },
        "Model size": {
            "type": "string",
            "minLength": 1,
            "description": "Parameter count or qualitative size, e.g. '70B', '1T (32B active)', 'undisclosed'.",
        },
        "Model type": {
            "type": "string",
            "enum": MODEL_TYPES,
            "description": "Coarse size category used to order the leaderboard.",
        },
        "Paper URL": {
            "type": "string",
            "pattern": _URL_PATTERN,
            "description": "Optional link to the model's paper.",
        },
        "LLM URI": {
            "type": "string",
            "pattern": _URL_PATTERN,
            "description": "Optional OpenAI-compatible base URL for a self-hosted/custom endpoint (same as --llm_uri).",
        },
        "Comments": {
            "type": "string",
            "description": "Optional free-text note shown in the leaderboard.",
        },
        "show_in_leaderboard": {
            "type": "boolean",
            "description": "Optional; set false to hide the model from the public leaderboard "
            "(the leaderboard filters rows on this flag).",
        },
        "Batch size": {
            "type": "integer",
            "minimum": 1,
            "description": "Optional per-model request concurrency (same as --batch_size).",
        },
        "Number of retries": {
            "type": "integer",
            "minimum": 0,
            "description": "Optional per-model LiteLLM retry count for transient failures (same as --num_retries).",
        },
    },
}

# Fail fast at import time if the schema itself is malformed, then reuse a single
# compiled validator for every file.
Draft202012Validator.check_schema(MODEL_CONFIG_SCHEMA)
_VALIDATOR = Draft202012Validator(MODEL_CONFIG_SCHEMA)


def validate_model_config(config: object) -> list[str]:
    """Return a list of human-readable schema violations for one model config.

    An empty list means ``config`` is a valid model configuration. ``config`` is
    whatever ``yaml.safe_load`` produced for the file (usually a ``dict``; an
    empty file yields ``None``, which is itself reported as a violation).
    """
    errors: list[str] = []
    for error in sorted(_VALIDATOR.iter_errors(config), key=lambda e: list(e.path)):
        location = ".".join(str(part) for part in error.path) or "<document>"
        errors.append(f"{location}: {error.message}")
    return errors
