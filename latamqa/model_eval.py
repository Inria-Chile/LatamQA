import argparse
import os
import re
from itertools import product
from pathlib import Path

import pandas as pd
import yaml
from datasets import Dataset, load_dataset
from ollama import list as ollama_list
from rich.console import Console
from rich.markdown import Markdown
from structlog import get_logger
from tqdm.auto import tqdm
from datetime import datetime, timezone

from latamqa.eval_mcq import REGIONAL_DATASETS, TARGET_LANGUAGES, run_evaluation, DEFAULT_RESULTS_DIR

logger = get_logger(__name__)

MODELS_DIR = Path(__file__).parent / "models"


def load_models(path: Path) -> dict[str, dict[str, str]]:
    """Load model configurations from a directory of YAML files."""
    models = {}
    for model_file in path.glob("*.yaml"):
        with open(model_file, "r") as f:
            models[model_file.stem] = yaml.safe_load(f)
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
                f"Model definition «{model_name}» uses a LiteLLM Ollama-based model '{model['LiteLLM model name']}' but Ollama is not available: {e}."
            )
            exit(-1)
        if not matching_models:
            logger.fatal(
                f"Model definition «{model_name}» uses a LiteLLM Ollama-based model '{model['LiteLLM model name']}' not found locally by ollama. Ensure it is available by running `ollama pull {ollama_model_name}`."
            )
            exit(-1)
    return True


def compute_results(
    model_name: str,
    max_results: int | None = None,
    seed: int = 42,
    temperature: float = 0.0,
    prompt_template: str | None = None,
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
    llm_api_key: str | None = None,
    llm_uri: str | None = None,
) -> dict[str, float | str | int]:
    model = load_models(MODELS_DIR)[model_name]

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
    update_parser.add_argument("--prompt_template", type=str, default=None, help="File name of custom prompt template")
    update_parser.add_argument("--results_dir", type=str, default=DEFAULT_RESULTS_DIR, help="Folder for storing results")
    update_parser.add_argument(
        "--llm_api_key",
        type=str,
        default=None,
        help="API key for LLM (if needed)",
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
        )


if __name__ == "__main__":
    main()
