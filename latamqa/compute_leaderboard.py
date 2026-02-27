

from latamqa.eval_mcq_any import run_evaluation, TARGET_LANGUAGES, REGIONAL_DATASETS

from itertools import product

import pandas as pd



MODELS = {
    "llama-3.1-8b": {
        "LiteLLM model name": "ollama/llama3.1:8b",
        "Model name": "meta-llama/Llama-3.1-8B-Instruct",
        "Model URL": "https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct",
        "Model size": "8B",
        "Model type": "small",
    },
    "llama-3.3-70b": {
        "LiteLLM model name": "huggingface/meta-llama/Llama-3.3-70B-Instruct",
        "Model name": "meta-llama/Llama-3.3-70B-Instruct",
        "Model URL": "https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct",
        "Model size": "70B",
        "Model type": "large",
    },
}

hyper_params = dict(max_results=2, seed=42, temperature=0.0)

RESULTS_FILE = "leaderboard.csv"


def compute_results(model_name):
    model = MODELS[model_name]
    instances = product(REGIONAL_DATASETS, TARGET_LANGUAGES)

    results = {}

    for inst in instances:
        region, lang = inst
        result = run_evaluation(model["LiteLLM model name"], region, lang, **hyper_params)
        results[f"{region} ({lang})"] = result["accuracy"]

    return results


def main():
    try:
        df = pd.read_csv(RESULTS_FILE)
    except IOError:
        df = pd.DataFrame()

    model = "llama-3.1-8b"

    results = compute_results(model)

    entry = {**results, **MODELS[model]}
    print(entry)

    df = pd.concat(df, pd.DataFrame(data=entry))

    df.to_csv(RESULTS_FILE)

    print(entry)


if __name__ == "__main__":
    main()
