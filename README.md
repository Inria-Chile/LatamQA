# LatamQA 

<!--
Source - https://stackoverflow.com/a/38274615
Posted by user3638471, modified by community. See post 'Timeline' for change history
Retrieved 2026-02-11, License - CC BY-SA 4.0
-->

<img src="latam_questions_map.png" alt="Alt Text" width="300"/>


[LatamQA](https://users.dcc.uchile.cl/~vbarrier/paper/MME_ACL_SC_Biases.pdf) is a cultural knowledge benchmark designed to evaluate Large Language Models on Latin American contexts. The dataset addresses the critical gap in bias detection resources for non-English languages and underrepresented cultures. Built from 26,000+ Wikipedia articles and structured using Wikidata's knowledge graph with expert guidance from social scientists, LatamQA contains over 26,000 multiple-choice questions covering the diverse popular and social cultures of Latin American countries. Questions are available in Spanish and Portuguese (the region's primary languages) as well as English translations, enabling evaluation of both multilingual capabilities and cultural representation. This resource helps researchers assess whether LLMs—predominantly trained on Global North data—exhibit prejudicial behavior or knowledge gaps when handling Latin American cultural contexts.

## Composition 

* MCQ in Latam Spanish, Iberian Spanish, Brasilian Portuguese. Every file has the English version
* Metadata and content of the Wikipedia articles

## HuggingFace's `dataset`

The collection of datasets is available online: https://huggingface.co/collections/inria-chile/latamqa

## Evaluation script

`eval_mcq.py` evaluates a model on the LatamQA MCQ benchmark via OpenAI-compatible or Mistral APIs.

### Requirements

```bash
pip install pandas datasets openai tqdm
# For Mistral models:
pip install mistralai
```

### Environment variables

```bash
export API_LLM="your-api-key"
export URL_LLM="https://your-api-endpoint"  # OpenAI-compatible base URL
```

### Usage

```bash
python eval_mcq.py --model <model_name> [options]
```

| Argument | Default | Description |
|---|---|---|
| `--model` | (required) | Model name to evaluate |
| `--provider` | `openai` | `openai` or `mistral` |
| `--region` | `es-la` | `es-la`, `es-es`, or `pt-br` |
| `--lang` | `o` | `o` (original language) or `en` (english) |
| `--limit` | None | Limit number of questions evaluated |
| `--seed` | `42` | Seed for answer option shuffling |
| `--temperature` | `0.0` | Sampling temperature |

### Examples

```bash
# Evaluate on Latam Spanish (default)
python eval_mcq.py --model meta-llama/Llama-3.1-8B-Instruct

# Evaluate on Brazilian Portuguese, english questions, first 100 items
python eval_mcq.py --model meta-llama/Llama-3.1-8B-Instruct --region pt-br --lang en --limit 100

# Evaluate with Mistral provider
python eval_mcq.py --model mistral-large-latest --provider mistral --region es-es
```

### Prompt template

The evaluation prompt is defined in `prompt_eval.txt`.

### Output

Results are saved in the `results/` directory:
- `mcq_eval_results_<region>_<lang>_<model>.csv` -- per-question details
- `mcq_eval_summary_<region>_<lang>_<model>.txt` -- accuracy summary

## Cite 

If this work was useful please cite the following: 
```
@inproceedings{karmimleveraging2026,
  title={Leveraging Wikidata for Geographically Informed Sociocultural Bias Dataset Creation: Application to Latin America},
  author={Karmim, Yannis and Pino, Renato and Contreras, Hernan and Lira, Hernan and Cifuentes, Sebastien and Escoffier, Simon and Marti, Luis and Seddah, Djamé and Barriere, Valentin},
  booktitle={Proceedings of the 1sh Workshop on Multilingual Multicultural Evaluation @ EACL26},
  url={https://users.dcc.uchile.cl/~vbarrier/paper/MME_ACL_SC_Biases.pdf},
  year={2026}
}
```

