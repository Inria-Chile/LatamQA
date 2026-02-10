# LatamQA 

LatamQA is a cultural knowledge benchmark designed to evaluate Large Language Models on Latin American contexts. The dataset addresses the critical gap in bias detection resources for non-English languages and underrepresented cultures. Built from 26,000+ Wikipedia articles and structured using Wikidata's knowledge graph with expert guidance from social scientists, LatamQA contains over 26,000 multiple-choice questions covering the diverse popular and social cultures of Latin American countries. Questions are available in Spanish and Portuguese (the region's primary languages) as well as English translations, enabling evaluation of both multilingual capabilities and cultural representation. This resource helps researchers assess whether LLMs—predominantly trained on Global North data—exhibit prejudicial behavior or knowledge gaps when handling Latin American cultural contexts.

## Composition 

* MCQ in Latam Spanish, Iberian Spanish, Brasilian Portuguese. Every file has the English version
* Metadata and content of the Wikipedia articles

## HuggingFace's `dataset`

The collection of datasets is available online: https://huggingface.co/collections/inria-chile/latamqa

## Evaluation scripts 

The evaluation scripts to audit your model will be available soon! 

## Cite 

```
@inproceedings{karmimleveraging2026,
  title={Leveraging Wikidata for Geographically Informed Sociocultural Bias Dataset Creation: Application to Latin America},
  author={Karmim, Yannis and Pino, Renato and Contreras, Hernan and Lira, Hernan and Cifuentes, Sebastien and Escoffier, Simon and Marti, Luis and Seddah, Djamé and Barriere, Valentin},
  booktitle={Proceedings of the 1sh Workshop on Multilingual Multicultural Evaluation @ EACL26},
  year={2026}
}
```

