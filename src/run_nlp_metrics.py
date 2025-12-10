#!/usr/bin/env python
"""
run_nlp_metrics.py

Compute NLP overlap metrics between a reference answer column and a model
answer column in a CSV file.

Metrics:
- BLEU          (from the 'bleu' package)
- ROUGE-L F1    (from 'rouge-score')
- METEOR        (from 'nltk')
- BERTScore F1  (from 'bert-score')

Example:
    python src/run_nlp_metrics.py \
        --input results/cla_qa_answers_gpt4.csv \
        --output results/cla_qa_nlp_metrics_gpt4.csv \
        --reference-col reference_answer \
        --answer-col gpt4_answer
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List

import pandas as pd


# ---------------------------------------------------------------------
# Metric imports
# ---------------------------------------------------------------------


def import_bleu():
    try:
        from bleu import BLEU  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "The 'bleu' package is required for BLEU.\n"
            "Install it with:\n\n"
            "    pip install bleu\n"
        ) from exc
    return BLEU


def import_rouge_scorer():
    try:
        from rouge_score import rouge_scorer  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "The 'rouge-score' package is required for ROUGE-L.\n"
            "Install it with:\n\n"
            "    pip install rouge-score\n"
        ) from exc
    return rouge_scorer


def import_meteor():
    try:
        from nltk.translate.meteor_score import single_meteor_score  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "The 'nltk' package is required for METEOR.\n"
            "Install it with:\n\n"
            "    pip install nltk\n"
            "    python -c \"import nltk; nltk.download('wordnet')\"\n"
        ) from exc
    return single_meteor_score


def import_bertscore():
    try:
        from bert_score import score as bert_score  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "The 'bert-score' package is required for BERTScore.\n"
            "Install it with:\n\n"
            "    pip install bert-score\n"
        ) from exc
    return bert_score


# ---------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------


def compute_bleu_scores(refs: List[str], hyps: List[str]) -> List[float]:
    """
    Compute BLEU for each (ref, hyp) pair using the 'bleu' package.

    BLEU(...) typically returns a score in [0, 1]. We keep it as-is.
    """
    BLEU = import_bleu()
    scores: List[float] = []

    for ref, hyp in zip(refs, hyps):
        if not ref or not hyp:
            scores.append(float("nan"))
            continue
        score = BLEU(hyp, ref)  # hypothesis, reference
        scores.append(float(score))

    return scores


def compute_rouge_l_scores(refs: List[str], hyps: List[str]) -> List[float]:
    """
    Compute ROUGE-L F1 for each (ref, hyp) pair.
    """
    rouge_scorer = import_rouge_scorer()
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)  # type: ignore[arg-type]
    scores: List[float] = []

    for ref, hyp in zip(refs, hyps):
        if not ref or not hyp:
            scores.append(float("nan"))
            continue
        r = scorer.score(ref, hyp)["rougeL"]
        scores.append(float(r.fmeasure))

    return scores


def compute_meteor_scores(refs: List[str], hyps: List[str]) -> List[float]:
    """
    Compute METEOR for each (ref, hyp) pair.
    """
    single_meteor_score = import_meteor()
    scores: List[float] = []

    for ref, hyp in zip(refs, hyps):
        if not ref or not hyp:
            scores.append(float("nan"))
            continue
        s = single_meteor_score(ref, hyp)
        scores.append(float(s))

    return scores


def compute_bertscore_f1(refs: List[str], hyps: List[str], lang: str = "en") -> List[float]:
    """
    Compute BERTScore F1 for each (ref, hyp) pair.

    We call bert-score once over the full list.
    """
    bert_score = import_bertscore()
    P, R, F1 = bert_score(
        hyps,
        refs,
        lang=lang,
        rescale_with_baseline=True,
    )
    scores: List[float] = [float(f.item()) for f in F1]
    return scores


# ---------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------


def run_nlp_metrics(
    input_path: Path,
    output_path: Path,
    reference_col: str,
    answer_col: str,
    skip_bert: bool = False,
) -> None:
    """
    Load CSV, compute BLEU/ROUGE-L/METEOR/(BERTScore), and save results.

    Adds columns:
        bleu, rouge_l, meteor, bert_f1 (optional)
    """
    logging.info("Loading input CSV from %s", input_path)
    df = pd.read_csv(input_path)

    if reference_col not in df.columns:
        raise ValueError(f"Reference column '{reference_col}' not found in input CSV.")
    if answer_col not in df.columns:
        raise ValueError(f"Answer column '{answer_col}' not found in input CSV.")

    refs = df[reference_col].fillna("").astype(str).tolist()
    hyps = df[answer_col].fillna("").astype(str).tolist()

    logging.info("Computing BLEU (bleu package)...")
    df["bleu"] = compute_bleu_scores(refs, hyps)

    logging.info("Computing ROUGE-L F1 (rouge-score)...")
    df["rouge_l"] = compute_rouge_l_scores(refs, hyps)

    logging.info("Computing METEOR (nltk)...")
    df["meteor"] = compute_meteor_scores(refs, hyps)

    if not skip_bert:
        logging.info("Computing BERTScore F1 (bert-score)...")
        df["bert_f1"] = compute_bertscore_f1(refs, hyps)
    else:
        logging.info("Skipping BERTScore computation (--skip-bert set).")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    logging.info("Saving metrics to %s", output_path)
    df.to_csv(output_path, index=False)
    logging.info("Done.")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute BLEU, ROUGE-L, METEOR, and BERTScore for a single answer column."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input CSV with reference and answer columns.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output CSV with metric columns.",
    )
    parser.add_argument(
        "--reference-col",
        type=str,
        default="reference_answer",
        help="Name of the reference answer column.",
    )
    parser.add_argument(
        "--answer-col",
        type=str,
        default="answer",
        help="Name of the model answer column to evaluate.",
    )
    parser.add_argument(
        "--skip-bert",
        action="store_true",
        help="Skip BERTScore computation (faster, fewer dependencies).",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    args = parse_args()

    logging.info("Input:         %s", args.input)
    logging.info("Output:        %s", args.output)
    logging.info("Reference col: %s", args.reference_col)
    logging.info("Answer col:    %s", args.answer_col)
    logging.info("Skip BERT:     %s", args.skip_bert)

    run_nlp_metrics(
        input_path=args.input,
        output_path=args.output,
        reference_col=args.reference_col,
        answer_col=args.answer_col,
        skip_bert=args.skip_bert,
    )


if __name__ == "__main__":
    main()
