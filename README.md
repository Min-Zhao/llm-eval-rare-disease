# LLM Evaluation for Free-Text Rare Disease QA  
Code and full prompt templates for automating LLM-based evaluation of free-text answers to rare disease patient questions.

## Overview
This repository provides the full **code, prompt templates, NLP metrics, and analysis scripts** used in the manuscript evaluating large language models (LLMs) for patient-facing free-text QA in **Complex Lymphatic Anomalies (CLAs)**.

Included components:

- Answer generation for open-weight LLMs (Ollama)
- LLM-based evaluators (reference-free, reference-guided, and rubric-based)
- Pairwise comparison evaluator
- NLP metrics: BLEU, ROUGE-L, METEOR, BERTScore
- Correlation, ICC, and binary agreement tools
- A small synthetic example dataset for quick testing

The **full generated answers and physician-annotated accuracy scores** used in the manuscript are publicly available at: 
CLA-QA: An Expert-Annotated Benchmark of Patient Questions and LLM Responses for Complex Lymphatic Anomalies.

DOI: 🔗 https://data.library.wustl.edu/record/108301?v=tab

## Installation
```bash
git clone https://github.com/<your-username>/llm-eval-free-text-rare-disease-qa
cd llm-eval-free-text-rare-disease-qa
pip install -r requirements.txt
python -c "import nltk; nltk.download('wordnet')"
```
**Note:** This repository uses **Ollama** to run local LLMs for evaluation.

## Citation
If you use this repository or the evaluation pipeline, please cite:

Zhao M, et al. Automating Evaluation of LLM-Generated Responses to Patient Questions about Rare Diseases.
medRxiv, 2025.  https://doi.org/10.1101/2025.10.06.25337181
