import argparse
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

from latamqa.eval_mcq import REGIONAL_DATASETS, TARGET_LANGUAGES, run_evaluation

logger = get_logger(__name__)

MODELS_DIR = Path(__file__).parent / "models"

LEADERBOARD_DATASET = "inria-chile/latamqa-leaderboard"
README_FILE = Path(__file__).parent.parent / "README.md"


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


def compute_results(
    model_name: str,
    max_results: int | None = None,
    seed: int = 42,
    temperature: float = 0.0,
    prompt_template: str | None = None,
    results_dir: str | Path = Path().cwd() / "results",
    llm_api_key: str | None = None,
    llm_uri: str | None = None,
) -> dict[str, float | str | int]:
    model = load_models(MODELS_DIR)[model_name]

    check_ollama_model(model_name, model)

    instances = product(REGIONAL_DATASETS, TARGET_LANGUAGES)

    results = {}

    for inst in tqdm(list(instances), desc=f"Computing for '{model_name}'"):
        region, lang = inst
        result = run_evaluation(
            model["LiteLLM model name"],
            region=region,
            lang=lang,
            seed=seed,
            temperature=temperature,
            prompt_template=prompt_template,
            results_dir=results_dir,
            llm_api_key=llm_api_key,
            llm_uri=llm_uri,
            max_results=max_results,
        )
        results[f"{region} ({lang})"] = result["accuracy"]

    return results


def list_models():
    """Lists all available models from the models directory."""
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


def update_leaderboard(
    model_name: str,
    max_results: int | None = None,
    seed: int = 42,
    temperature: float = 0.0,
    prompt_template: str | None = None,
    results_dir: str | Path = Path().cwd() / "results",
    llm_api_key: str | None = None,
    llm_uri: str | None = None,
):
    """Runs evaluation for a model and updates the leaderboard."""
    models = load_models(MODELS_DIR)

    if model_name not in models:
        logger.error(f"Model '{model_name}' not found. Available models: {', '.join(models.keys())}")
        return

    try:
        dataset = load_dataset(LEADERBOARD_DATASET)
        df = dataset["train"].to_pandas()
        logger.info(f"Loaded existing results from Hugging Face dataset '{LEADERBOARD_DATASET}'.")
    except Exception as e:
        logger.warning(f"Could not load dataset '{LEADERBOARD_DATASET}'. Creating a new one. Error: {e}")
        df = pd.DataFrame()

    logger.info(f"Computing leaderboard results for model: {model_name}")
    results = compute_results(
        model_name,
        max_results=max_results,
        seed=seed,
        temperature=temperature,
        prompt_template=prompt_template,
        results_dir=results_dir,
        llm_api_key=llm_api_key,
        llm_uri=llm_uri,
    )
    entry = {**results, **models[model_name], "timestamp": pd.Timestamp.now()}

    df = pd.concat([df, pd.DataFrame(data=[entry])], ignore_index=True)  # Convert timestamp to string for serialization
    if "timestamp" in df.columns:
        df["timestamp"] = df["timestamp"].astype(str)

    new_dataset = Dataset.from_pandas(df)
    try:
        new_dataset.push_to_hub(LEADERBOARD_DATASET, private=False)
        logger.info(f"Leaderboard updated on Hugging Face Hub for '{model_name}'.")
    except Exception as e:
        logger.error(
            f"Failed to push to Hugging Face Hub. Please ensure you are logged in (`huggingface-cli login`). Error: {e}"
        )


def get_leaderboard_markdown() -> str:
    """Generates the leaderboard as a Markdown table."""
    try:
        dataset = load_dataset(LEADERBOARD_DATASET)
        df = dataset["train"].to_pandas()
    except Exception as e:
        logger.warning(f"Could not load dataset '{LEADERBOARD_DATASET}'. Run an update to create it. Error: {e}")
        return ""

    latest_results = df.sort_values("timestamp").groupby("Model name").tail(1)

    if "Model type" in latest_results.columns:
        model_type_order = ["small", "medium", "large"]
        latest_results["Model type"] = pd.Categorical(
            latest_results["Model type"].str.lower(), categories=model_type_order, ordered=True
        )
        latest_results = latest_results.sort_values("Model type")

    display_df = latest_results.copy()

    # Create the linked model name column
    if "Model URL" in display_df.columns and "Model name" in display_df.columns:
        display_df["Model name"] = display_df.apply(
            lambda row: (
                f"[{row['Model name']}]({row['Model URL']})" if pd.notna(row.get("Model URL")) else row.get("Model name", "N/A")
            ),
            axis=1,
        )

    # Fill missing Model type and Model size with ?
    if "Model type" in display_df.columns:
        display_df["Size"] = display_df["Model type"].astype(object).fillna("??")
    if "Model size" in display_df.columns:
        display_df["# params"] = display_df["Model size"].fillna("??")

    # Replace NaN with empty string in Comments
    if "Comments" in display_df.columns:
        display_df["Comments"] = display_df["Comments"].fillna("")

    # Create the Paper link column
    if "Paper URL" in display_df.columns:
        display_df["Ref."] = display_df["Paper URL"].apply(lambda url: f"[🔗]({url})" if pd.notna(url) and url else "")

    accuracy_columns = [f"{region} ({lang})" for region, lang in product(REGIONAL_DATASETS, TARGET_LANGUAGES)]
    existing_accuracy_cols = [col for col in accuracy_columns if col in display_df.columns]

    # Calculate average accuracy if accuracy columns exist
    if existing_accuracy_cols:
        display_df["Average"] = display_df[existing_accuracy_cols].mean(axis=1)

    cols_to_format = [col for col in ["Average"] + accuracy_columns if col in display_df.columns]

    for col in cols_to_format:
        numeric_col = pd.to_numeric(display_df[col], errors="coerce")
        if not numeric_col.isnull().all():
            max_val = numeric_col.max()

            def highlight_max(val):
                if pd.isna(val):
                    return ""
                return f"**{val:.4f}**" if abs(val - max_val) < 1e-9 else f"{val:.4f}"

            display_df[col] = numeric_col.apply(highlight_max)

    display_columns = ["Model name", "Ref.", "Size", "# params", "Comments", "Average"] + accuracy_columns
    display_df = display_df[[col for col in display_columns if col in display_df.columns]]

    return display_df.to_markdown(index=False)


def update_readme_leaderboard():
    """Updates the leaderboard in README.md."""
    leaderboard_md = get_leaderboard_markdown()
    if not leaderboard_md:
        logger.warning("Leaderboard is empty, not updating README.md.")
        return

    try:
        with open(README_FILE, "r") as f:
            readme_content = f.read()
    except FileNotFoundError:
        logger.error(f"{README_FILE} not found.")
        return

    new_readme_content, num_subs = re.subn(
        r'(<span name="leaderboard">)(.*?)(</span>)',
        lambda m: f"{m.group(1)}\n\n{leaderboard_md}\n\n{m.group(3)}",
        readme_content,
        flags=re.DOTALL,
    )

    if num_subs == 0:
        logger.warning('Could not find `<span name="leaderboard">...</span>` in README.md. Not updating.')
        return

    with open(README_FILE, "w") as f:
        f.write(new_readme_content)

    logger.info("README.md updated with the latest leaderboard.")


def show_leaderboard():
    """Displays the latest leaderboard results for each model."""
    leaderboard_md = get_leaderboard_markdown()
    if not leaderboard_md:
        return

    console = Console()
    console.print(Markdown(leaderboard_md, style="table.center"))


def plot_leaderboard(results_dir: str | Path = Path().cwd() / "results"):
    """Generates and saves a combined radar and line plot of the leaderboard accuracies."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        import seaborn as sns
    except ImportError:
        logger.error(
            "matplotlib, seaborn, and numpy are required for plotting. Please install them: pip install matplotlib seaborn numpy"
        )
        return

    try:
        dataset = load_dataset(LEADERBOARD_DATASET)
        df = dataset["train"].to_pandas()
    except Exception as e:
        logger.warning(f"Could not load dataset '{LEADERBOARD_DATASET}'. Run an update to create it. Error: {e}")
        return

    latest_results = df.sort_values("timestamp").groupby("Model name").tail(1)

    accuracy_columns = [f"{region} ({lang})" for region, lang in product(REGIONAL_DATASETS, TARGET_LANGUAGES)]
    existing_accuracy_cols = [col for col in accuracy_columns if col in latest_results.columns]

    if not existing_accuracy_cols:
        logger.warning("No accuracy columns found in leaderboard data. Cannot generate plot.")
        return

    # Ensure accuracy columns are numeric.
    for col in existing_accuracy_cols:
        latest_results[col] = pd.to_numeric(latest_results[col], errors="coerce")

    # Calculate average for sorting and sort
    latest_results["Average"] = latest_results[existing_accuracy_cols].mean(axis=1)
    latest_results = latest_results.sort_values("Average", ascending=False).reset_index(drop=True)

    sns.set_context("paper", font_scale=1.1)
    sns.set_style("whitegrid")

    # Define styles and create mappings for consistent plotting
    model_names = latest_results["Model name"].tolist()
    palette = sns.color_palette("colorblind", n_colors=len(model_names))
    linestyles = ["-", "--", "-.", ":"]
    markers = ["o", "s", "v", "^", "<", ">", "p", "*", "X", "D"]

    color_map = {model: color for model, color in zip(model_names, palette)}
    style_map = {model: linestyles[i % len(linestyles)] for i, model in enumerate(model_names)}
    marker_map = {model: markers[i % len(markers)] for i, model in enumerate(model_names)}

    # Create figure and subplots
    fig = plt.figure(figsize=(22, 8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.2])
    ax_radar = fig.add_subplot(gs[0], polar=True)
    ax_line = fig.add_subplot(gs[1])

    # --- Radar Plot ---
    labels = np.array(existing_accuracy_cols)
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Close the plot

    for i, row in latest_results.iterrows():
        model_name = row["Model name"]
        stats = row[existing_accuracy_cols].values.flatten().tolist()
        stats += stats[:1]  # Close the plot

        ax_radar.plot(
            angles,
            stats,
            color=color_map[model_name],
            linewidth=2,
            linestyle=style_map[model_name],
            marker=marker_map[model_name],
            label=model_name,
        )
        ax_radar.fill(angles, stats, color=color_map[model_name], alpha=0.01)

    # Formatting radar plot
    ax_radar.set_yticklabels([])
    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(labels, size=12)
    ax_radar.set_rlabel_position(30)
    yticks = [0.2, 0.4, 0.6, 0.8, 1.0]
    ax_radar.set_yticks(yticks)
    ax_radar.set_yticklabels([f"{y:.1f}" for y in yticks], color="grey", size=10)
    ax_radar.set_ylim(0, 1.05)
    ax_radar.set_title("LatamQA MCQ Leaderboard - Accuracy Radar", size=18, color="black", y=1.15)

    # --- Line Plot ---
    long_df = latest_results.melt(
        id_vars=["Model name"], value_vars=existing_accuracy_cols, var_name="Metric", value_name="Accuracy"
    )
    long_df.dropna(subset=["Accuracy"], inplace=True)

    lineplot_linestyles  = [style_map[model] for model in style_map]

    sns.lineplot(
        data=long_df,
        x="Metric",
        y="Accuracy",
        hue="Model name",
        style="Model name",
        markers=marker_map,
        dashes=lineplot_linestyles,
        palette=color_map,
        ax=ax_line,
        legend=False,  # Handled by fig.legend
    )

    # Formatting line plot
    ax_line.set_title("LatamQA MCQ Leaderboard - Accuracy by Region and Language", fontsize=18, pad=20)
    ax_line.set_xlabel("Regions and languages", fontsize=14, labelpad=15)
    ax_line.set_ylabel("Accuracy", fontsize=14, labelpad=15)
    ax_line.set_ylim(0, 1.0)
    ax_line.tick_params(axis="both", which="major", labelsize=12)
    plt.setp(ax_line.get_xticklabels(), rotation=29, ha="right")
    sns.despine(ax=ax_line)

    # --- Shared Legend and final adjustments ---
    handles, labels = ax_radar.get_legend_handles_labels()
    fig.legend(handles, labels, title="Model", loc="center left", bbox_to_anchor=(0.9, 0.5), title_fontsize="14", fontsize="12")

    plt.tight_layout(rect=[0, 0, 0.9, 1])  # Adjust layout to make room for the legend

    plot_path = Path(results_dir) / "leaderboard_combined_plot.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, bbox_inches="tight", dpi=300)
    logger.info(f"Leaderboard plot saved to {plot_path}")


def main():
    parser = argparse.ArgumentParser(description="Manage the LatamQA leaderboard.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="List available models.")
    update_parser = subparsers.add_parser("update", help="Update leaderboard with a model's results.")
    update_parser.add_argument(
        "--model",
        required=True,
        help="Model name to evaluate (e.g., 'gpt-4o', 'ollama/llama3.1', 'anthropic/claude-3') see <https://docs.litellm.ai/docs/providers> for details.",  # noqa: E501
    )
    update_parser.add_argument("--max_results", type=int, default=None, help="Maximum number of rows to process")
    update_parser.add_argument("--seed", type=int, default=42, help="Seed for shuffling options")
    update_parser.add_argument("--temperature", type=float, default=0.0, help="Temperature")
    update_parser.add_argument("--prompt_template", type=str, default=None, help="File name of custom prompt template")
    update_parser.add_argument("--results_dir", type=str, default=Path().cwd() / "results", help="Folder for storing results")
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
        help="URL for local/custom LLM provider (if needed)",
    )
    subparsers.add_parser("show", help="Show the current leaderboard with the latest results.")
    subparsers.add_parser("update-readme", help="Updates the leaderboard in README.md.")
    plot_parser = subparsers.add_parser("plot", help="Generate a plot of the leaderboard.")
    plot_parser.add_argument("--results_dir", type=str, default=Path().cwd() / "results", help="Folder for storing results")

    args = parser.parse_args()

    if args.command == "list":
        list_models()
    elif args.command == "update":
        update_leaderboard(
            model_name=args.model,
            max_results=args.max_results,
            seed=args.seed,
            temperature=args.temperature,
            prompt_template=args.prompt_template,
            results_dir=args.results_dir,
            llm_api_key=args.llm_api_key,
            llm_uri=args.llm_uri,
        )
    elif args.command == "show":
        show_leaderboard()
    elif args.command == "update-readme":
        update_readme_leaderboard()
    elif args.command == "plot":
        plot_leaderboard(results_dir=args.results_dir)



if __name__ == "__main__":
    main()
