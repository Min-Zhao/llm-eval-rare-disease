#!/usr/bin/env python
"""
results_analysis.py

Summarize relationships between expert-graded metrics and automatic metrics.

This script loads a single CSV (e.g., per-answer results) and produces:

1) Item-level correlations:
   - Pearson & Spearman (no p-values)
   - Pearson + p-value
   - Spearman + p-value

2) Optional binarized metrics (1–3 vs 4–5):
   - Phi (Pearson on 0/1) + p-value
   - Agreement percentage

3) Optional ICC (Intraclass Correlation, ICC2) between expert and automatic metrics.
   (Supports rescaling NLP metrics from [0,1] to [0,5].)

4) Optional model-level correlations:
   - Pearson, Spearman, Kendall between model-level aggregated (e.g., mean) scores.

Outputs:
- A set of CSV tables in the output directory.

Example usage:

python src/results_analysis.py \
  --input results/cla_qa_eval_and_metrics.csv \
  --output-dir results/analysis \
  --expert-metrics accu_bs_mean,compl_bs_mean,relev_bs_mean \
  --auto-metrics bleu,rouge_l,meteor,bert_f1,llm_eval_score \
  --nlp-metrics bleu,rouge_l,meteor,bert_f1 \
  --model-col model
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd

from correlation_tests import (
    calculate_metric_correlations,
    calculate_pearson_and_pvalue_table,
    calculate_spearman_and_pvalue_table,
    convert_metrics_to_binary,
    calculate_pearson_and_pvalue_table_binary,
    calculate_agreement_proportion_table_binary,
    calculate_icc_and_pvalue_table,
    correlate_at_model_level_table,
)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _parse_list(arg: Optional[str]) -> List[str]:
    """Parse a comma-separated CLI argument into a list of strings."""
    if not arg:
        return []
    return [x.strip() for x in arg.split(",") if x.strip()]


def _save_table(df: pd.DataFrame, out_dir: Path, stem: str) -> Path:
    """Save a DataFrame as CSV with a given stem under out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stem}.csv"
    df.to_csv(path, index=True)
    logging.info("Saved: %s", path)
    return path


# ---------------------------------------------------------------------
# Main analysis routine
# ---------------------------------------------------------------------


def run_results_analysis(
    input_path: Path,
    output_dir: Path,
    expert_metrics: List[str],
    auto_metrics: List[str],
    nlp_metrics: List[str],
    model_col: Optional[str] = None,
    do_binary: bool = True,
    do_icc: bool = True,
    agg_func: str = "mean",
    round_digits: int = 3,
) -> None:
    """
    Run correlation and agreement analyses between expert and automatic metrics.
    """
    logging.info("Loading input from: %s", input_path)
    df = pd.read_csv(input_path)

    # Sanity checks
    missing_expert = [m for m in expert_metrics if m not in df.columns]
    missing_auto = [m for m in auto_metrics if m not in df.columns]
    if missing_expert:
        logging.warning("Expert metrics not found in DataFrame: %s", missing_expert)
    if missing_auto:
        logging.warning("Auto metrics not found in DataFrame: %s", missing_auto)

    expert_metrics = [m for m in expert_metrics if m in df.columns]
    auto_metrics = [m for m in auto_metrics if m in df.columns]

    if not expert_metrics or not auto_metrics:
        raise ValueError(
            "No overlapping expert/auto metrics found in DataFrame. "
            "Please check column names.",
        )

    logging.info("Expert metrics: %s", expert_metrics)
    logging.info("Auto metrics:   %s", auto_metrics)

    # ---------------------------------------------
    # 1. Item-level correlations (Pearson & Spearman)
    # ---------------------------------------------
    logging.info("Computing item-level Pearson & Spearman correlations...")
    metric_corr = calculate_metric_correlations(df, expert_metrics, auto_metrics)
    _save_table(metric_corr, output_dir, "item_level_correlations")

    logging.info("Computing item-level Pearson + p-value table...")
    pearson_table = calculate_pearson_and_pvalue_table(
        df,
        expert_metrics,
        auto_metrics,
    )
    _save_table(pearson_table, output_dir, "item_level_pearson_pvalues")

    logging.info("Computing item-level Spearman + p-value table...")
    spearman_table = calculate_spearman_and_pvalue_table(
        df,
        expert_metrics,
        auto_metrics,
    )
    _save_table(spearman_table, output_dir, "item_level_spearman_pvalues")

    # ---------------------------------------------
    # 2. Binary metrics: Phi + agreement
    # ---------------------------------------------
    if do_binary:
        logging.info("Converting metrics to binary (1–3 = 0, 4–5 = 1)...")
        df_binary = convert_metrics_to_binary(df, expert_metrics + auto_metrics)

        logging.info("Computing binary Phi correlations (Pearson on 0/1)...")
        phi_table = calculate_pearson_and_pvalue_table_binary(
            df_binary,
            expert_metrics,
            auto_metrics,
        )
        _save_table(phi_table, output_dir, "binary_phi_pvalues")

        logging.info("Computing binary agreement proportions...")
        agreement_table = calculate_agreement_proportion_table_binary(
            df_binary,
            expert_metrics,
            auto_metrics,
        )
        _save_table(agreement_table, output_dir, "binary_agreement_percent")
    else:
        logging.info("Skipping binary analyses (do_binary = False).")

    # ---------------------------------------------
    # 3. ICC between expert and automatic metrics
    # ---------------------------------------------
    if do_icc:
        logging.info("Computing ICC2 and p-values between expert and auto metrics...")
        icc_table = calculate_icc_and_pvalue_table(
            df,
            expert_graded_metrics=expert_metrics,
            auto_metrics=auto_metrics,
            nlp_metrics=nlp_metrics or None,
        )
        _save_table(icc_table, output_dir, "icc_item_level")
    else:
        logging.info("Skipping ICC analyses (do_icc = False).")

    # ---------------------------------------------
    # 4. Model-level correlations (aggregated per model)
    # ---------------------------------------------
    if model_col and model_col in df.columns:
        logging.info("Computing model-level correlations (agg=%s) using column '%s'...",
                     agg_func, model_col)
        model_corr_table = correlate_at_model_level_table(
            df=df,
            model_col=model_col,
            expert_graded_metrics=expert_metrics,
            auto_metrics=auto_metrics,
            agg_func=agg_func,
            round_digits=round_digits,
        )
        _save_table(model_corr_table, output_dir, "model_level_correlations")
    else:
        if model_col:
            logging.warning(
                "Model column '%s' not found in DataFrame; "
                "skipping model-level correlations.",
                model_col,
            )
        else:
            logging.info("No model_col specified; skipping model-level correlations.")

    logging.info("Results analysis completed.")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze correlations and agreement between expert-graded "
            "and automatic metrics."
        ),
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input CSV with expert and auto metric columns.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save analysis tables (CSV).",
    )
    parser.add_argument(
        "--expert-metrics",
        type=str,
        required=True,
        help=(
            "Comma-separated list of expert-graded metric columns "
            "(e.g., 'accu_bs_mean,compl_bs_mean,relev_bs_mean')."
        ),
    )
    parser.add_argument(
        "--auto-metrics",
        type=str,
        required=True,
        help=(
            "Comma-separated list of automatic metric columns "
            "(e.g., 'bleu,rouge_l,meteor,bert_f1,llm_eval_score')."
        ),
    )
    parser.add_argument(
        "--nlp-metrics",
        type=str,
        default="",
        help=(
            "Comma-separated list of automatic metrics to rescale from [0,1] "
            "to [0,5] before ICC (e.g., 'bleu,rouge_l,meteor,bert_f1'). "
            "If empty, no rescaling is performed."
        ),
    )
    parser.add_argument(
        "--model-col",
        type=str,
        default="",
        help=(
            "Optional column name for model identifiers (e.g., 'model'). "
            "If provided and present, model-level correlations are computed."
        ),
    )
    parser.add_argument(
        "--no-binary",
        action="store_true",
        help="Disable binary (1–3 vs 4–5) Phi and agreement analyses.",
    )
    parser.add_argument(
        "--no-icc",
        action="store_true",
        help="Disable ICC analyses.",
    )
    parser.add_argument(
        "--agg-func",
        type=str,
        default="mean",
        help="Aggregation function for model-level correlations (default: 'mean').",
    )
    parser.add_argument(
        "--round-digits",
        type=int,
        default=3,
        help="Decimal places for correlation coefficients (default: 3).",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    args = parse_args()

    expert_metrics = _parse_list(args.expert_metrics)
    auto_metrics = _parse_list(args.auto_metrics)
    nlp_metrics = _parse_list(args.nlp_metrics)
    model_col = args.model_col or None

    logging.info("Input:        %s", args.input)
    logging.info("Output dir:   %s", args.output_dir)
    logging.info("Expert:       %s", expert_metrics)
    logging.info("Auto:         %s", auto_metrics)
    logging.info("NLP (ICC):    %s", nlp_metrics)
    logging.info("Model col:    %s", model_col)
    logging.info("do_binary:    %s", not args.no_binary)
    logging.info("do_icc:       %s", not args.no_icc)

    run_results_analysis(
        input_path=args.input,
        output_dir=args.output_dir,
        expert_metrics=expert_metrics,
        auto_metrics=auto_metrics,
        nlp_metrics=nlp_metrics,
        model_col=model_col,
        do_binary=not args.no_binary,
        do_icc=not args.no_icc,
        agg_func=args.agg_func,
        round_digits=args.round_digits,
    )


if __name__ == "__main__":
    main()
