import argparse
from datetime import datetime
from itertools import product
from pathlib import Path

import pandas as pd
import yaml
from ollama import list as ollama_list
from rich.console import Console
from rich.markdown import Markdown
from structlog import get_logger
from tqdm.auto import tqdm

from latamqa.eval_mcq import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_NUM_RETRIES,
    DEFAULT_RESULTS_DIR,
    REGIONAL_DATASETS,
    TARGET_LANGUAGES,
    run_evaluation,
)
from latamqa.model_schema import validate_model_config

logger = get_logger(__name__)

MODELS_DIR = Path(__file__).parent / "models"


def load_models(path: Path) -> dict[str, dict]:
    """Load and validate model configurations from a directory of YAML files.

    Every ``*.yaml`` is validated against the model configuration schema
    (:data:`latamqa.model_schema.MODEL_CONFIG_SCHEMA`). If any file is invalid,
    all problems across all files are reported and the process stops, so a
    malformed config is caught here rather than surfacing later mid-evaluation.
    """
    models: dict[str, dict] = {}
    errors: list[str] = []
    for model_file in sorted(path.glob("*.yaml")):
        try:
            with open(model_file, "r") as f:
                config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            # A syntactically broken file never reaches schema validation; report
            # it the same way so the user still gets a clean, file-pointing error.
            errors.append(f"{model_file.name} -> could not parse YAML: {' '.join(str(e).split())}")
            continue
        file_errors = validate_model_config(config)
        if file_errors:
            errors.extend(f"{model_file.name} -> {msg}" for msg in file_errors)
            continue
        models[model_file.stem] = config

    if errors:
        logger.fatal("Invalid model configuration file(s):\n" + "\n".join(f"  • {e}" for e in errors))
        exit(-1)

    return models


def check_ollama_model(model_name, model: dict) -> bool:
    """Check if the Ollama model is available."""

    litellm_model = model["LiteLLM model name"]
    if litellm_model.startswith("ollama/") and not litellm_model.endswith("-cloud"):
        ollama_model_name = litellm_model.split("/", 1)[1]
        try:
            matching_models = [m for m in ollama_list().models if m.model == ollama_model_name]
        except ConnectionError as e:
            logger.fatal(
                f"Model definition «{model_name}» uses a LiteLLM Ollama-based model "
                f"'{model['LiteLLM model name']}' but Ollama is not available: {e}."
            )
            exit(-1)
        if not matching_models:
            logger.fatal(
                f"Model definition «{model_name}» uses a LiteLLM Ollama-based model "
                f"'{model['LiteLLM model name']}' not found locally by ollama. "
                f"Ensure it is available by running `ollama pull {ollama_model_name}`."
            )
            exit(-1)
    return True


def _validate_int(minimum: int):
    """Build a validator that accepts an int >= ``minimum`` and stops the run otherwise."""

    def _validator(model_name: str, source: str, option: str, value: object) -> int:
        # Accept ints and integer-valued floats (e.g. YAML `8.0`), matching the
        # JSON Schema "integer" type so the model-config schema and this resolver
        # agree. `bool` is an `int` subclass, so reject it explicitly (e.g. so
        # `Batch size: true` is not read as 1); `8.5` and strings also fail.
        if isinstance(value, bool):
            number = None
        elif isinstance(value, int):
            number = value
        elif isinstance(value, float) and value.is_integer():
            number = int(value)
        else:
            number = None

        if number is None or number < minimum:
            logger.fatal(
                f"Invalid «{option}» for model «{model_name}» ({source}): "
                f"expected an integer >= {minimum}, got {value!r}."
            )
            exit(-1)
        return number

    return _validator


def _validate_str(model_name: str, source: str, option: str, value: object) -> str:
    """Validate that a resolved option is a non-empty string, or stop the run."""
    if not isinstance(value, str) or not value.strip():
        logger.fatal(
            f"Invalid «{option}» for model «{model_name}» ({source}): "
            f"expected a non-empty string, got {value!r}."
        )
        exit(-1)
    return value


# Options that may be set per-model in the YAML *or* on the command line.
# Maps the CLI/parameter name -> (YAML field name, built-in default, validator).
# Setting the same option in both places is treated as a clash: the run stops
# rather than silently guessing which value should win.
YAML_OVERRIDABLE_OPTIONS: dict[str, tuple[str, object, object]] = {
    "batch_size": ("Batch size", DEFAULT_BATCH_SIZE, _validate_int(1)),
    "num_retries": ("Number of retries", DEFAULT_NUM_RETRIES, _validate_int(0)),
    "llm_uri": ("LLM URI", None, _validate_str),
}


def resolve_model_options(
    model_name: str,
    model: dict,
    cli_overrides: dict[str, object],
) -> dict[str, object]:
    """Merge per-model YAML options with command-line overrides.

    ``cli_overrides`` maps each option name to the value passed on the command
    line, or ``None`` when the flag was omitted. When only one source sets a
    value, that source wins; when neither does, the built-in default is used.
    If BOTH the command line and the model YAML set the same option, the run is
    stopped so the ambiguity is surfaced instead of being silently resolved.
    """
    resolved: dict[str, object] = {}
    for option, (yaml_key, default, validate) in YAML_OVERRIDABLE_OPTIONS.items():
        cli_value = cli_overrides.get(option)
        yaml_value = model.get(yaml_key)

        if cli_value is not None and yaml_value is not None:
            logger.fatal(
                f"Option clash for model «{model_name}»: «{option}» is set both on the command "
                f"line (--{option}={cli_value}) and in the model YAML («{yaml_key}: {yaml_value}»). "
                f"Set it in only one place."
            )
            exit(-1)

        if cli_value is not None:
            resolved[option] = validate(model_name, "command line", option, cli_value)
        elif yaml_value is not None:
            resolved[option] = validate(model_name, f"YAML field «{yaml_key}»", option, yaml_value)
        else:
            resolved[option] = default

    return resolved


def compute_results(
    model_name: str,
    max_results: int | None = None,
    seed: int = 42,
    temperature: float = 0.0,
    prompt_template: str | None = None,
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
    llm_api_key: str | None = None,
    llm_uri: str | None = None,
    batch_size: int | None = None,
    num_retries: int | None = None,
) -> dict[str, float | str | int]:
    model = load_models(MODELS_DIR)[model_name]

    # `batch_size`, `num_retries` and `llm_uri` may each come from the CLI or the
    # model YAML (e.g. a YAML may pin its own OpenAI-compatible endpoint via the
    # "LLM URI" field), but not both — setting one in both places is reported as
    # a clash and stops the run.
    options = resolve_model_options(
        model_name,
        model,
        {"batch_size": batch_size, "num_retries": num_retries, "llm_uri": llm_uri},
    )
    batch_size = options["batch_size"]
    num_retries = options["num_retries"]
    llm_uri = options["llm_uri"]

    instances = product(REGIONAL_DATASETS, TARGET_LANGUAGES)

    results = {}

    current_results_dir = Path(results_dir) / f"{model_name}__{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger.info(f"Storing results for model '{model_name}' in directory: {current_results_dir}")

    for inst in tqdm(list(instances), desc=f"Computing for «{model_name}»", leave=False):
        region, lang = inst
        result = run_evaluation(
            model["LiteLLM model name"],
            region=region,
            lang=lang,
            seed=seed,
            temperature=temperature,
            prompt_template=prompt_template,
            results_dir=current_results_dir,
            llm_api_key=llm_api_key,
            llm_uri=llm_uri,
            max_results=max_results,
            batch_size=batch_size,
            num_retries=num_retries,
        )
        results[f"{region} ({lang})"] = result["accuracy"]
    return results


def list_models():
    """Lists all available models in the `models/` directory."""
    console = Console()
    models = load_models(MODELS_DIR)

    if not models:
        logger.info("No model configuration files found in `models/` directory.")
        return

    df = pd.DataFrame.from_dict(models, orient="index")

    # Create the linked model name column
    df["Model name link"] = df.apply(
        lambda row: (
            f"[{row['Model name']}]({row['Model URL']})"
            if pd.notna(row.get("Model URL")) and pd.notna(row.get("Model name"))
            else row.get("Model name", "N/A")
        ),
        axis=1,
    )

    display_df = df.reset_index().rename(columns={"index": "Model ID", "Model name link": "Model name"})

    display_columns = ["Model ID", "Model name", "LiteLLM model name", "Model size", "Model type"]
    display_df = display_df[[col for col in display_columns if col in display_df.columns]]
    console.print(Markdown(display_df.to_markdown(index=False)))


def main():
    parser = argparse.ArgumentParser(description="Manage the LatamQA leaderboard.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="List available models.")
    update_parser = subparsers.add_parser("evaluate", help="Evaluate a model and store results locally.")
    update_parser.add_argument(
        "--model",
        required=True,
        help="Model to evaluate (models should be defined in the models directory)",
    )
    update_parser.add_argument("--max_results", type=int, default=None, help="Maximum number of rows to process")
    update_parser.add_argument("--seed", type=int, default=42, help="Seed for shuffling options")
    update_parser.add_argument("--temperature", type=float, default=0.0, help="Temperature")
    update_parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help=(
            f"Number of concurrent requests kept in flight (1 = sequential). Overrides the model YAML's "
            f"'Batch size' field; setting it in both places is an error. Defaults to {DEFAULT_BATCH_SIZE}."
        ),
    )
    update_parser.add_argument(
        "--num_retries",
        type=int,
        default=None,
        help=(
            f"LiteLLM retries for transient failures (rate limits, timeouts). Overrides the model YAML's "
            f"'Number of retries' field; setting it in both places is an error. Defaults to {DEFAULT_NUM_RETRIES}."
        ),
    )
    update_parser.add_argument("--prompt_template", type=str, default=None, help="File name of custom prompt template")
    update_parser.add_argument("--results_dir", type=str, default=DEFAULT_RESULTS_DIR, help="Folder for storing results")
    update_parser.add_argument(
        "--llm_api_key",
        type=str,
        default=None,
        help="API key for LLM (if needed)",
    )
    update_parser.add_argument(
        "--llm_uri",
        type=str,
        default=None,
        help="URL for local/custom LLM provider. Set it here OR in the model YAML's 'LLM URI' field, "
        "not both (setting it in both places is an error).",
    )

    args = parser.parse_args()

    if args.command == "list":
        list_models()
    elif args.command == "evaluate":
        compute_results(
            model_name=args.model,
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


if __name__ == "__main__":
    main()
