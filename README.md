# `LatamQA`: Leveraging Wikidata for Geographically Informed Sociocultural Bias Dataset Creation: Application to Latin America 

<table><tr><td width="74%">
<a href="https://users.dcc.uchile.cl/~vbarrier/paper/MME_ACL_SC_Biases.pdf">LatamQA</a> is a cultural knowledge benchmark designed to evaluate Large Language Models on Latin American contexts. The dataset addresses the critical gap in bias detection resources for non-English languages and underrepresented cultures. Built from 26,000+ Wikipedia articles and structured using Wikidata's knowledge graph with expert guidance from social scientists, LatamQA contains over 26,000 multiple-choice questions covering the diverse popular and social cultures of Latin American countries. Questions are available in Spanish and Portuguese (the region's primary languages) as well as English translations, enabling evaluation of both multilingual capabilities and cultural representation. This resource helps researchers assess whether LLMs—predominantly trained on Global North data—exhibit prejudicial behavior or knowledge gaps when handling Latin American cultural contexts.</td><td width="26%"><img src="latam_questions_map.png" alt="MCQs per Latam country" width="100%"/></td></tr></table>

## Dataset composition

* MCQ in Latam Spanish, Iberian Spanish, Brasilian Portuguese. Every file has the English version
* Metadata and content of the Wikipedia articles

## LatamQA online datasets

We have made available datasets and matching metadata as the Hugging Face `dataset` collection: <https://huggingface.co/collections/inria-chile/latamqa>.

| Dataset name | HF hub identifier |
| :--- | :--- |
| Latam Spanish MCQ dataset | [`inria-chile/latamqa_mcq_es-la`](https://huggingface.co/datasets/inria-chile/latamqa_mcq_es-la) |
| Latam Spanish metadata | [`inria-chile/latamqa_articles_es-la`](https://huggingface.co/datasets/inria-chile/latamqa_articles_es-la) |
| Iberian Spanish MCQ dataset | [`inria-chile/latamqa_mcq_es-es`](https://huggingface.co/datasets/inria-chile/latamqa_mcq_es-es) |
| Iberian Spanish metadata | [`inria-chile/latamqa_articles_es-es`](https://huggingface.co/datasets/inria-chile/latamqa_articles_es-es) |
| Brazilian Portuguese MCQ dataset | [`inria-chile/latamqa_mcq_pt-br`](https://huggingface.co/datasets/inria-chile/latamqa_mcq_pt-br) |
| Brazilian Portuguese metadata | [`inria-chile/latamqa_articles_pt-br`](https://huggingface.co/datasets/inria-chile/latamqa_articles_pt-br) |

Usage example:

```python
from datasets import load_dataset

dataset = load_dataset("inria-chile/latamqa_mcq_es-la")
dataset_as_df = dataset["train"].to_pandas()
```

## `eval_mcq` Evaluation script

[`latamqa/eval_mcq.py`](latamqa/eval_mcq.py) evaluates a model on the LatamQA MCQ benchmark via OpenAI-compatible or Mistral APIs.

### Requirements

1. Install the `uv` dependencies handling tool <https://docs.astral.sh/uv/getting-started/installation/>. For instance, by running:

  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

2. In the `LatamQA` directory, install project dependencies by running `uv sync`. This will create a Python virtual environment in the folder `.venv/`.
3. Setup `API_LLM` and `URL_LLM` environment variables:

```bash
export API_LLM="your-api-key"
export URL_LLM="https://your-api-endpoint"  # OpenAI-compatible base URL
```

### Usage

```bash
uv run eval_mcq --model MODEL [--provider {openai,mistral}] [--region {es-la,es-es,pt-br}] [--lang {o,en}] [--limit LIMIT] [--seed SEED] [--temperature TEMPERATURE] [--prompt_template PROMPT_TEMPLATE]
```

| Argument  | Default | Description |
|-----------|---------|-------------|
| `--model` | (required) | Model name to evaluate |
| `--provider` | `openai` | `openai` or `mistral` |
| `--region` | `es-la` | `es-la`, `es-es`, or `pt-br` |
| `--lang` | `o` | `o` (original language) or `en` (English) |
| `--limit` | None | Limit number of questions evaluated |
| `--seed` | `42` | Seed for answer option shuffling |
| `--temperature` | `0.0` | Sampling temperature |
| `--prompt_template` | None | File name of custom prompt template |

**Note:** `eval_mcq.py` can also be run by activating the Python virtual environment and running the script directly, for instance:

```bash
source .venv/bin/activate
python latamqa/eval_mcq.py --model <your-model>
```

### Examples

```bash
# Evaluate on Latam Spanish (default)
uv run eval_mcq --model meta-llama/Llama-3.1-8B-Instruct

# Evaluate on Brazilian Portuguese, english questions, first 100 items
uv run eval_mcq --model meta-llama/Llama-3.1-8B-Instruct --region pt-br --lang en --limit 100

# Evaluate with Mistral provider
uv run eval_mcq --model mistral-large-latest --provider mistral --region es-es
```

## `eval_mcq_any` Universal Evaluation script

[`latamqa/eval_mcq_any.py`](latamqa/eval_mcq_any.py) is a universal version of the evaluation script that supports 100+ LLM providers (OpenAI, Anthropic, Mistral, Ollama, vLLM, etc.) via [LiteLLM](https://github.com/BerriAI/litellm).

### Usage

```bash
uv run eval_mcq_any --model PROVIDER/MODEL_NAME [--region {es-la,es-es,pt-br}] [--lang {o,en}] [--api_key API_KEY] [--base_url BASE_URL] [OTHER_OPTIONS]
```

| Argument  | Default | Description |
|-----------|---------|-------------|
| `--model` | (required) | Model name (e.g., `gpt-4o`, `anthropic/claude-3-5-sonnet-20240620`, `ollama/llama3`) |
| `--api_key` | `API_LLM` env | API key for the provider |
| `--base_url` | `URL_LLM` env | Base URL for local or custom providers (e.g., `http://localhost:11434` for Ollama) |

### Examples for different providers

```bash
# OpenAI (standard)
export OPENAI_API_KEY="sk-..."
uv run eval_mcq_any --model gpt-4o

# Anthropic
export ANTHROPIC_API_KEY="sk-ant-..."
uv run eval_mcq_any --model anthropic/claude-3-5-sonnet-20240620

# Local model via Ollama
uv run eval_mcq_any --model ollama/llama3 --base_url http://localhost:11434

# Local model via vLLM (OpenAI-compatible)
uv run eval_mcq_any --model openai/your-model --base_url http://localhost:8000/v1 --api_key "dummy"
```

### Custom prompt template

If you want to use a custom evaluation prompt you can pass the file name as argument to `eval_mcq.py`. File `prompt_eval.txt` contains an example of prompt.

### Output

Results are saved in the `results/` directory:

* `mcq_eval_results_<region>_<lang>_<model>.csv` -- per-question details
* `mcq_eval_summary_<region>_<lang>_<model>.txt` -- accuracy summary

## Leaderboard

The project includes a Gradio-based leaderboard to visualize model performance.

### 1. Update leaderboard data
After running evaluations, aggregate the results into the leaderboard format:
```bash
uv run update_leaderboard
```

### 2. Launch the leaderboard app
```bash
uv run python latamqa/leaderboard/app.py
```
The leaderboard will be available at `http://127.0.0.1:7860`.

## Citation

If this work was useful please cite it as:

> Karmim, Y., Pino, R., Contreras, H., Lira, H., Cifuentes, S., Escoffier, S., Martí, L., Seddah, D., & Barriere, V. (2026). **Leveraging Wikidata for Geographically Informed Sociocultural Bias Dataset Creation: Application to Latin America**. In Proceedings of the Workshop on Multilingual Multicultural Evaluation of the 19th Conference of the European Chapter of the Association for Computational Linguistics (EACL'2026). Rabbat, Morocco. [⟨hal-05510068⟩](https:/.hal.science/hal-05510068).

BibTeX:

```bibtex
@inproceedings{karmimleveraging2026,
  title      = {
    Leveraging Wikidata for Geographically Informed Sociocultural Bias Dataset Creation: {A}pplication to
    {L}atin {A}merica
  },
  author     = {
    Karmim, Yannis and Pino, Renato and Contreras, Hernan and Lira, Hernan and Cifuentes, Sebastien and
    Escoffier, Simon and Mart\'{i}, Luis and Seddah, Djam{\'e} and Barri{\`e}re, Valentin
  },
  year       = 2026,
  month      = {Mar},
  booktitle  = {
    Proceedings of the Workshop on Multilingual Multicultural Evaluation of the 19th Conference of the
    European Chapter of the Association for Computational Linguistics (EACL'2026)
  },
  address    = {Rabbat, Morocco},
  url        = {https://inria.hal.science/hal-05510068},
  editor     = {
    Pinzhen Chen and Vil\'{e}m Zouhar and Hanxu Hu and Simran Khanuja and Wenhao Zhu and Barry Haddow and
    Alexandra Birch and Alham Fikri Aji and Rico Sennrich and Sara Hooker
  },
  hal_id     = {hal-05510068},
  hal_version = {v1},
  eprint     = {hal-05510068},
  eprinttype = {hal}
}
```

