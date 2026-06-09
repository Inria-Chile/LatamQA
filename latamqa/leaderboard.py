import argparse
import re
from itertools import product
from pathlib import Path

import pandas as pd
from datasets import Dataset, load_dataset
from rich.console import Console
from rich.markdown import Markdown
from structlog import get_logger

from latamqa.eval_mcq import REGIONAL_DATASETS, TARGET_LANGUAGES
from latamqa.model_eval import MODELS_DIR, list_models, load_models

logger = get_logger(__name__)

LEADERBOARD_DATASET = "inria-chile/latamqa-leaderboard"
README_FILE = Path(__file__).parent.parent / "README.md"

# Every region/language combination that a complete evaluation must cover.
EXPECTED_SLICES = [f"{region} ({lang})" for region, lang in product(REGIONAL_DATASETS, TARGET_LANGUAGES)]


def load_results(model: dict, results_dir: str | Path) -> dict:
    """Loads the results for a given model from the specified results directory.

    Parses the per-slice ``mcq_eval_summary_*.txt`` files produced by
    `run_evaluation` and returns a mapping of ``"{region} ({lang})"`` to accuracy,
    matching the columns expected by the leaderboard.
    """
    results_dir = Path(results_dir)
    if not results_dir.is_dir():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    expected_model = model.get("LiteLLM model name")
    results: dict[str, float] = {}

    for summary_path in sorted(results_dir.glob("mcq_eval_summary_*.txt")):
        summary = {}
        for line in summary_path.read_text(encoding="utf-8").splitlines():
            key, sep, value = line.partition(":")
            if sep:
                summary[key.strip()] = value.strip()

        # Skip summaries that belong to a different model when sharing a directory.
        if expected_model and summary.get("model") not in (None, expected_model):
            continue

        region, lang = summary.get("region"), summary.get("lang")
        if region is None or lang is None or "accuracy" not in summary:
            logger.warning(f"Skipping malformed summary file: {summary_path}")
            continue

        results[f"{region} ({lang})"] = float(summary["accuracy"])

    if not results:
        logger.warning(f"No evaluation summaries found in: {results_dir}")

    return results


def validate_results(results: dict, results_dir: str | Path) -> list[str]:
    """Validates that ``results`` covers every expected region/language slice.

    Returns the list of missing ``"{region} ({lang})"`` slices. An empty list
    means the results directory contains all the information the leaderboard
    needs. Slices with a non-numeric accuracy are also reported as missing.
    """
    missing: list[str] = []
    for slice_name in EXPECTED_SLICES:
        value = results.get(slice_name)
        if value is None:
            missing.append(slice_name)
            continue
        try:
            float(value)
        except (TypeError, ValueError):
            logger.warning(f"Non-numeric accuracy for slice '{slice_name}' in {results_dir}: {value!r}")
            missing.append(slice_name)
    return missing


def update_leaderboard(
    model_name: str,
    results_dir: str | Path,
    dry_run: bool = False,
    allow_incomplete: bool = False,
):
    """Reads a folder with model results and updates the leaderboard.

    When ``dry_run`` is True, every step runs as usual (loading the existing
    dataset, parsing results, building the new entry) but the result is logged
    instead of being pushed to the Hugging Face Hub.
    """
    models = load_models(MODELS_DIR)

    if model_name not in models:
        logger.error(f"Model '{model_name}' not found. Available models: {', '.join(models.keys())}")
        return

    model = models[model_name]

    try:
        dataset = load_dataset(LEADERBOARD_DATASET)
        df = dataset["train"].to_pandas()
        logger.info(f"Loaded existing results from Hugging Face dataset '{LEADERBOARD_DATASET}'.")
    except Exception as e:
        logger.warning(f"Could not load dataset '{LEADERBOARD_DATASET}'. Creating a new one. Error: {e}")
        df = pd.DataFrame()

    logger.info(f"Updating leaderboard results for model: {model_name} from results directory: {results_dir}")

    results = load_results(model, results_dir=results_dir)

    missing = validate_results(results, results_dir)
    if missing:
        message = (
            f"Results directory '{results_dir}' is missing {len(missing)} of {len(EXPECTED_SLICES)} "
            f"required slices for '{model_name}': {', '.join(missing)}"
        )
        if not allow_incomplete:
            logger.error(f"{message}. Aborting (pass allow_incomplete=True / --allow_incomplete to override).")
            return
        logger.warning(f"{message}. Proceeding anyway because allow_incomplete is set.")

    entry = {**results, **models[model_name], "timestamp": pd.Timestamp.now()}

    df = pd.concat([df, pd.DataFrame(data=[entry])], ignore_index=True)  # Convert timestamp to string for serialization
    if "timestamp" in df.columns:
        df["timestamp"] = df["timestamp"].astype(str)

    new_dataset = Dataset.from_pandas(df)

    if dry_run:
        logger.warn(f"dry-run==True; skipping push to '{LEADERBOARD_DATASET}'. Would add entry for '{model_name}': {results}")
        return

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

    # Keep only the latest entry for each model based on the timestamp, and sort by Model type if available.
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


def update_readme_with_leaderboard():
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

    # Optionally refresh the Mermaid radar chart if a `radar` span is present.
    radar_md = get_leaderboard_mermaid_radar()
    if radar_md:
        new_readme_content, radar_subs = re.subn(
            r'(<span name="radar">)(.*?)(</span>)',
            lambda m: f"{m.group(1)}\n\n{radar_md}\n\n{m.group(3)}",
            new_readme_content,
            flags=re.DOTALL,
        )
        if radar_subs == 0:
            logger.warning('Could not find `<span name="radar">...</span>` in README.md. Skipping radar chart.')

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

    lineplot_linestyles = [style_map[model] for model in style_map]

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


def get_leaderboard_mermaid_radar() -> str:
    """Generates the leaderboard as a Mermaid radar chart.

    Returns a fenced ```mermaid block using the ``radar-beta`` chart type
    (Mermaid >= 11.6), with one axis per region/language slice and one curve
    per model. Returns an empty string when no leaderboard data is available.
    """
    try:
        dataset = load_dataset(LEADERBOARD_DATASET)
        df = dataset["train"].to_pandas()
    except Exception as e:
        logger.warning(f"Could not load dataset '{LEADERBOARD_DATASET}'. Run an update to create it. Error: {e}")
        return ""

    latest_results = df.sort_values("timestamp").groupby("Model name").tail(1)

    accuracy_columns = [f"{region} ({lang})" for region, lang in product(REGIONAL_DATASETS, TARGET_LANGUAGES)]
    existing_accuracy_cols = [col for col in accuracy_columns if col in latest_results.columns]

    if not existing_accuracy_cols:
        logger.warning("No accuracy columns found in leaderboard data. Cannot generate radar chart.")
        return ""

    # Ensure accuracy columns are numeric and sort models by overall average.
    for col in existing_accuracy_cols:
        latest_results[col] = pd.to_numeric(latest_results[col], errors="coerce")
    latest_results["Average"] = latest_results[existing_accuracy_cols].mean(axis=1)
    latest_results = latest_results.sort_values("Average", ascending=False).reset_index(drop=True)

    # Axis keys must be simple identifiers; map them to the human-readable slice labels.
    axis_keys = [f"a{i}" for i in range(len(existing_accuracy_cols))]
    axis_defs = ", ".join(f'{key}["{label}"]' for key, label in zip(axis_keys, existing_accuracy_cols))

    lines = ["```mermaid", "---", "title: LatamQA MCQ Leaderboard", "---", "radar-beta", f"  axis {axis_defs}"]

    for i, row in latest_results.iterrows():
        values = ", ".join(f"{v:.3f}" if pd.notna(v) else "0" for v in row[existing_accuracy_cols])
        lines.append(f'  curve c{i}["{row["Model name"]}"]{{{values}}}')

    lines += ["", "  max 1", "  min 0", "```"]

    return "\n".join(lines)


def show_leaderboard_radar():
    """Prints the leaderboard as a Mermaid radar chart block."""
    radar = get_leaderboard_mermaid_radar()
    if radar:
        print(radar)


def main():
    parser = argparse.ArgumentParser(description="Manage the LatamQA leaderboard.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="List available models.")
    update_parser = subparsers.add_parser("update", help="Update leaderboard from a model's results directory.")
    update_parser.add_argument(
        "--model",
        required=True,
        help="Model ID to update (must match a configuration file in the models directory, e.g. 'gemma-2-2b').",
    )
    update_parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Folder containing the model's evaluation results (mcq_eval_summary_*.txt files).",
    )
    update_parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Run the full update process but do not push changes to the leaderboard dataset.",
    )
    update_parser.add_argument(
        "--allow_incomplete",
        action="store_true",
        help="Push the entry even if the results directory is missing some region/language slices.",
    )
    subparsers.add_parser("show", help="Show the current leaderboard with the latest results.")
    subparsers.add_parser("radar", help="Print the leaderboard as a Mermaid radar chart.")
    subparsers.add_parser("update-readme", help="Updates the leaderboard in README.md.")
    plot_parser = subparsers.add_parser("plot", help="Generate a plot of the leaderboard.")
    plot_parser.add_argument("--results_dir", type=str, default=Path().cwd() / "results", help="Folder for storing results")

    args = parser.parse_args()

    if args.command == "list":
        list_models()
    elif args.command == "update":
        update_leaderboard(
            model_name=args.model,
            results_dir=args.results_dir,
            dry_run=args.dry_run,
            allow_incomplete=args.allow_incomplete,
        )
    elif args.command == "show":
        show_leaderboard()
    elif args.command == "radar":
        show_leaderboard_radar()
    elif args.command == "update-readme":
        update_readme_with_leaderboard()
    elif args.command == "plot":
        plot_leaderboard(results_dir=args.results_dir)


if __name__ == "__main__":
    main()
