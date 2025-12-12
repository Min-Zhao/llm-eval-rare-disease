#!/usr/bin/env python
"""
correlation_tests.py

Utility functions for computing correlations and agreement between
expert-graded metrics and automatic metrics.

Includes:
- Pearson & Spearman correlations (per item)
- Pearson/P-value and Spearman/P-value tables
- Binary recoding (1–5 → low vs high)
- Phi correlations and agreement proportions for binary metrics
- ICC (Intraclass Correlation Coefficient) using pingouin
- Model-level correlations (aggregated per model)
"""

from __future__ import annotations

from typing import Iterable, List

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr, spearmanr
import pingouin as pg


# ---------------------------------------------------------------------
# Basic metric ↔ metric correlations
# ---------------------------------------------------------------------


def _drop_na_pairwise(
    x: pd.Series,
    y: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """Drop rows where x or y are NaN, returning aligned Series."""
    valid = ~(x.isna() | y.isna())
    return x[valid], y[valid]


def calculate_metric_correlations(
    df: pd.DataFrame,
    human_metrics: Iterable[str],
    auto_metrics: Iterable[str],
) -> pd.DataFrame:
    """
    Calculate Pearson and Spearman correlations between expert-graded
    and automatic metrics at the item level.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the metric columns.
    human_metrics : iterable of str
        Column names for expert-graded metrics.
    auto_metrics : iterable of str
        Column names for automatic metrics.

    Returns
    -------
    pd.DataFrame
        Rows = auto_metrics
        Columns = MultiIndex (Expert Metric, Correlation Type)
        where Correlation Type ∈ {'Pearson', 'Spearman'}.
    """
    human_metrics = list(human_metrics)
    auto_metrics = list(auto_metrics)

    columns = pd.MultiIndex.from_product(
        [human_metrics, ["Pearson", "Spearman"]],
        names=["Expert Metric", "Correlation Type"],
    )
    combined_df = pd.DataFrame(index=auto_metrics, columns=columns)

    for auto in auto_metrics:
        for expert in human_metrics:
            try:
                x, y = _drop_na_pairwise(df[auto], df[expert])
                if len(x) < 2:
                    raise ValueError("Not enough valid pairs for correlation.")
                pearson_corr, _ = pearsonr(x, y)
                spearman_corr, _ = spearmanr(x, y)

                combined_df.loc[auto, (expert, "Pearson")] = round(pearson_corr, 3)
                combined_df.loc[auto, (expert, "Spearman")] = round(spearman_corr, 3)

            except Exception:  # noqa: BLE001
                combined_df.loc[auto, (expert, "Pearson")] = None
                combined_df.loc[auto, (expert, "Spearman")] = None

    return combined_df


# ---------------------------------------------------------------------
# Pearson / Spearman + p-values
# ---------------------------------------------------------------------


def calculate_pearson_and_pvalue_table(
    df: pd.DataFrame,
    expert_graded_metrics: Iterable[str],
    auto_metrics: Iterable[str],
) -> pd.DataFrame:
    """
    Calculate Pearson correlation coefficient and p-value between
    automatic and expert metrics at the item level.

    Returns
    -------
    pd.DataFrame
        Rows = auto_metrics
        Columns = MultiIndex (Expert Metric, ['Pearson', 'P-value'])
    """
    expert_graded_metrics = list(expert_graded_metrics)
    auto_metrics = list(auto_metrics)

    columns = pd.MultiIndex.from_product(
        [expert_graded_metrics, ["Pearson", "P-value"]],
        names=["Expert Metric", "Statistic"],
    )
    combined_df = pd.DataFrame(index=auto_metrics, columns=columns)

    for auto in auto_metrics:
        for expert in expert_graded_metrics:
            try:
                x, y = _drop_na_pairwise(df[auto], df[expert])
                if len(x) < 2:
                    raise ValueError("Not enough valid pairs for correlation.")
                pearson_corr, p_value = pearsonr(x, y)

                combined_df.loc[auto, (expert, "Pearson")] = round(pearson_corr, 3)
                combined_df.loc[auto, (expert, "P-value")] = format(p_value, ".3g")
            except Exception:  # noqa: BLE001
                combined_df.loc[auto, (expert, "Pearson")] = None
                combined_df.loc[auto, (expert, "P-value")] = None

    return combined_df


def calculate_spearman_and_pvalue_table(
    df: pd.DataFrame,
    expert_graded_metrics: Iterable[str],
    auto_metrics: Iterable[str],
) -> pd.DataFrame:
    """
    Calculate Spearman correlation coefficient and p-value between
    automatic and expert metrics at the item level.

    Returns
    -------
    pd.DataFrame
        Rows = auto_metrics
        Columns = MultiIndex (Expert Metric, ['Spearman', 'P-value'])
    """
    expert_graded_metrics = list(expert_graded_metrics)
    auto_metrics = list(auto_metrics)

    columns = pd.MultiIndex.from_product(
        [expert_graded_metrics, ["Spearman", "P-value"]],
        names=["Expert Metric", "Statistic"],
    )
    combined_df = pd.DataFrame(index=auto_metrics, columns=columns)

    for auto in auto_metrics:
        for expert in expert_graded_metrics:
            try:
                x, y = _drop_na_pairwise(df[auto], df[expert])
                if len(x) < 2:
                    raise ValueError("Not enough valid pairs for correlation.")
                spearman_corr, p_value = spearmanr(x, y)

                combined_df.loc[auto, (expert, "Spearman")] = round(
                    spearman_corr,
                    3,
                )
                combined_df.loc[auto, (expert, "P-value")] = format(p_value, ".3g")
            except Exception:  # noqa: BLE001
                combined_df.loc[auto, (expert, "Spearman")] = None
                combined_df.loc[auto, (expert, "P-value")] = None

    return combined_df


# ---------------------------------------------------------------------
# Binary recoding + Phi correlations
# ---------------------------------------------------------------------


def convert_metrics_to_binary(
    df: pd.DataFrame,
    metric_list: Iterable[str],
    suffix: str = "_bi",
) -> pd.DataFrame:
    """
    Convert 1–5 scale metrics to binary: 0 (1–3), 1 (4–5).

    Parameters
    ----------
    df : pd.DataFrame
        Original DataFrame.
    metric_list : iterable of str
        List of metric column names to convert.
    suffix : str
        Suffix to add to new binary columns.

    Returns
    -------
    pd.DataFrame
        Copy of df with new binary columns added.
    """
    df_copy = df.copy()
    metric_list = list(metric_list)

    for col in metric_list:
        def _to_binary(x):
            if pd.isna(x):
                return np.nan
            return 1 if x >= 4 else 0

        df_copy[col + suffix] = df_copy[col].apply(_to_binary)

    return df_copy


def calculate_pearson_and_pvalue_table_binary(
    df: pd.DataFrame,
    expert_graded_metrics: Iterable[str],
    auto_metrics: Iterable[str],
    suffix: str = "_bi",
) -> pd.DataFrame:
    """
    Calculate Pearson correlation (Phi coefficient) and p-value between
    binarized auto and expert metrics (0/1).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with binary columns (0/1).
    expert_graded_metrics : iterable of str
        Original expert metric names (pre-suffix).
    auto_metrics : iterable of str
        Original auto metric names (pre-suffix).
    suffix : str
        Suffix used for binarized columns (default: '_bi').

    Returns
    -------
    pd.DataFrame
        Rows = auto_metrics
        Columns = MultiIndex (Expert Metric, ['Phi', 'P-value'])
    """
    expert_graded_metrics = list(expert_graded_metrics)
    auto_metrics = list(auto_metrics)

    expert_bi = [m + suffix for m in expert_graded_metrics]
    auto_bi = [m + suffix for m in auto_metrics]

    columns = pd.MultiIndex.from_product(
        [expert_graded_metrics, ["Phi", "P-value"]],
        names=["Expert Metric", "Statistic"],
    )
    combined_df = pd.DataFrame(index=auto_metrics, columns=columns)

    for auto, auto_col in zip(auto_metrics, auto_bi):
        for expert, expert_col in zip(expert_graded_metrics, expert_bi):
            try:
                x, y = _drop_na_pairwise(df[auto_col], df[expert_col])
                if len(x) < 2:
                    raise ValueError("Not enough valid pairs for Phi.")
                phi_corr, p_value = pearsonr(x, y)
                combined_df.loc[auto, (expert, "Phi")] = round(phi_corr, 3)
                combined_df.loc[auto, (expert, "P-value")] = format(p_value, ".3g")
            except Exception:  # noqa: BLE001
                combined_df.loc[auto, (expert, "Phi")] = None
                combined_df.loc[auto, (expert, "P-value")] = None

    return combined_df


def calculate_agreement_proportion_table_binary(
    df: pd.DataFrame,
    expert_graded_metrics: Iterable[str],
    auto_metrics: Iterable[str],
    suffix: str = "_bi",
) -> pd.DataFrame:
    """
    Calculate agreement proportion (%) between binarized expert and auto metrics.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with binary columns (0/1).
    expert_graded_metrics : iterable of str
        Original expert metric names (pre-suffix).
    auto_metrics : iterable of str
        Original auto metric names (pre-suffix).
    suffix : str
        Suffix used for binarized columns.

    Returns
    -------
    pd.DataFrame
        Rows = auto_metrics
        Columns = MultiIndex (Expert Metric, ['Agreement %'])
    """
    expert_graded_metrics = list(expert_graded_metrics)
    auto_metrics = list(auto_metrics)

    expert_bi = [m + suffix for m in expert_graded_metrics]
    auto_bi = [m + suffix for m in auto_metrics]

    columns = pd.MultiIndex.from_product(
        [expert_graded_metrics, ["Agreement %"]],
        names=["Expert Metric", "Statistic"],
    )
    combined_df = pd.DataFrame(index=auto_metrics, columns=columns)

    for auto, auto_col in zip(auto_metrics, auto_bi):
        for expert, expert_col in zip(expert_graded_metrics, expert_bi):
            try:
                subset = df[[auto_col, expert_col]].dropna()
                if subset.empty:
                    raise ValueError("No valid rows for agreement.")
                agreement = (subset[auto_col] == subset[expert_col]).mean()
                combined_df.loc[auto, (expert, "Agreement %")] = round(
                    agreement * 100,
                    1,
                )
            except Exception:  # noqa: BLE001
                combined_df.loc[auto, (expert, "Agreement %")] = None

    return combined_df


# ---------------------------------------------------------------------
# ICC (Intraclass Correlation) between expert and auto metrics
# ---------------------------------------------------------------------


def calculate_icc_and_pvalue_table(
    df: pd.DataFrame,
    expert_graded_metrics: Iterable[str],
    auto_metrics: Iterable[str],
    nlp_metrics: Iterable[str] | None = None,
) -> pd.DataFrame:
    """
    Calculate ICC (Intraclass Correlation Coefficient, ICC2) and p-value
    between automatic and expert metrics.

    Optionally rescales some NLP metrics from [0,1] to [0,5] before ICC.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with metric columns.
    expert_graded_metrics : iterable of str
        Expert metric column names (typically 1–5 scale).
    auto_metrics : iterable of str
        Automatic metric column names.
    nlp_metrics : iterable of str or None
        Names of automatic metrics to rescale from [0,1] to [0,5].
        If None, no rescaling is performed.

    Returns
    -------
    pd.DataFrame
        Rows = auto_metrics
        Columns = MultiIndex (Expert Metric, ['ICC', 'P-value'])
    """
    expert_graded_metrics = list(expert_graded_metrics)
    auto_metrics = list(auto_metrics)
    nlp_metrics = list(nlp_metrics) if nlp_metrics is not None else []

    columns = pd.MultiIndex.from_product(
        [expert_graded_metrics, ["ICC", "P-value"]],
        names=["Expert Metric", "Statistic"],
    )
    combined_df = pd.DataFrame(index=auto_metrics, columns=columns)

    # Work on a copy so we don't mutate df in-place
    df_work = df.copy()

    # Rescale specified NLP metrics (e.g., BLEU/ROUGE/METEOR/BERTScore 0–1 → 0–5)
    for metric in nlp_metrics:
        if metric in df_work.columns:
            df_work[metric] = df_work[metric] * 5

    for auto in auto_metrics:
        for expert in expert_graded_metrics:
            try:
                # Long format: 2 raters (auto vs expert), same targets
                ratings_auto = df_work[auto]
                ratings_expert = df_work[expert]
                mask = ~(ratings_auto.isna() | ratings_expert.isna())

                if mask.sum() < 2:
                    raise ValueError("Not enough valid pairs for ICC.")

                ratings_auto = ratings_auto[mask]
                ratings_expert = ratings_expert[mask]

                df_long = pd.DataFrame(
                    {
                        "targets": np.repeat(
                            ratings_auto.index.astype(str),
                            2,
                        ),
                        "raters": ["auto", "expert"] * len(ratings_auto),
                        "ratings": pd.concat(
                            [ratings_auto, ratings_expert],
                            ignore_index=True,
                        ),
                    },
                )

                icc_result = pg.intraclass_corr(
                    data=df_long,
                    targets="targets",
                    raters="raters",
                    ratings="ratings",
                )
                icc2_row = icc_result[icc_result["Type"] == "ICC2"].iloc[0]
                icc_value = round(icc2_row["ICC"], 3)
                p_value = format(icc2_row["pval"], ".3g")

                combined_df.loc[auto, (expert, "ICC")] = icc_value
                combined_df.loc[auto, (expert, "P-value")] = p_value

            except Exception:  # noqa: BLE001
                combined_df.loc[auto, (expert, "ICC")] = None
                combined_df.loc[auto, (expert, "P-value")] = None

    return combined_df


# ---------------------------------------------------------------------
# Model-level correlations (aggregating per model)
# ---------------------------------------------------------------------


def correlate_at_model_level_table(
    df: pd.DataFrame,
    model_col: str,
    expert_graded_metrics: Iterable[str],
    auto_metrics: Iterable[str],
    agg_func: str | callable = "mean",
    round_digits: int = 3,
) -> pd.DataFrame:
    """
    Compute Pearson, Spearman, and Kendall correlations at model-level
    between expert and auto metrics.

    Parameters
    ----------
    df : pd.DataFrame
        Input data with one row per (model, question).
    model_col : str
        Column name for model identifiers.
    expert_graded_metrics : iterable of str
        Expert metric column names.
    auto_metrics : iterable of str
        Automatic metric column names.
    agg_func : str or callable
        Aggregation per model (e.g., 'mean', 'median').
    round_digits : int
        Decimal places to round correlation coefficients.

    Returns
    -------
    pd.DataFrame
        Rows = auto_metrics
        Columns = MultiIndex (Expert Metric, Statistic), where Statistic is:
        ['Pearson r', 'Pearson p', 'Spearman ρ', 'Spearman p',
         'Kendall τ', 'Kendall p']
    """
    expert_graded_metrics = list(expert_graded_metrics)
    auto_metrics = list(auto_metrics)

    stat_labels = [
        "Pearson r",
        "Pearson p",
        "Spearman ρ",
        "Spearman p",
        "Kendall τ",
        "Kendall p",
    ]
    columns = pd.MultiIndex.from_product(
        [expert_graded_metrics, stat_labels],
        names=["Expert Metric", "Statistic"],
    )
    result_df = pd.DataFrame(index=auto_metrics, columns=columns)

    for auto in auto_metrics:
        for expert in expert_graded_metrics:
            try:
                df_grouped = (
                    df.groupby(model_col)[[expert, auto]]
                    .agg(agg_func)
                    .dropna()
                )
                if len(df_grouped) < 2:
                    raise ValueError("Not enough models to compute correlation.")

                pearson_r, pearson_p = pearsonr(
                    df_grouped[expert],
                    df_grouped[auto],
                )
                spearman_r, spearman_p = spearmanr(
                    df_grouped[expert],
                    df_grouped[auto],
                )
                kendall_r, kendall_p = kendalltau(
                    df_grouped[expert],
                    df_grouped[auto],
                )

                result_df.loc[auto, (expert, "Pearson r")] = round(
                    pearson_r,
                    round_digits,
                )
                result_df.loc[auto, (expert, "Pearson p")] = format(
                    pearson_p,
                    ".3g",
                )
                result_df.loc[auto, (expert, "Spearman ρ")] = round(
                    spearman_r,
                    round_digits,
                )
                result_df.loc[auto, (expert, "Spearman p")] = format(
                    spearman_p,
                    ".3g",
                )
                result_df.loc[auto, (expert, "Kendall τ")] = round(
                    kendall_r,
                    round_digits,
                )
                result_df.loc[auto, (expert, "Kendall p")] = format(
                    kendall_p,
                    ".3g",
                )

            except Exception:  # noqa: BLE001
                for stat in stat_labels:
                    result_df.loc[auto, (expert, stat)] = None

    return result_df
