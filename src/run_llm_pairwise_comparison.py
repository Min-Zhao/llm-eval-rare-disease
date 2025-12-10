#!/usr/bin/env python
"""
run_llm_pairwise_comparison.py

Pairwise comparison pipeline for model answers using an LLM evaluator.

Features:
- Convert expert-graded scores into pairwise win/tie records.
- Run pairwise LLM evaluation across all model pairs for each question.
- Compute win-rate per model from pairwise results.
- Compare LLM win-rates against human win-rates.
- Compute agreement with human pairwise decisions and Kendall's tau
  between human and LLM win-rates.

This script expects:
- A CSV with human-reviewed model scores per question.
- A CSV with model-generated answers for the same questions.
- Prompts in prompts/evaluator_prompts.json under "pairwise_optional".

Example usage:

python src/run_llm_pairwise_comparison.py \
  --human-scores data/human_scores.csv \
  --answers data/model_answers.csv \
  --output-prefix results/pairwise_eval \
  --rater-model llama3.3:70b \
  --pairwise-mode ref_free_pairwise
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import time
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from ollama import chat
from scipy.stats import kendalltau
from tqdm import tqdm

# ---------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------

DEFAULT_HUMAN_SCORES = Path("data/human_scores.csv")
DEFAULT_ANSWERS = Path("data/model_answers.csv")
DEFAULT_OUTPUT_PREFIX = Path("results/pairwise_eval")

DEFAULT_PROMPTS_PATH = Path("prompts/evaluator_prompts.json")
DEFAULT_PAIRWISE_MODE = "ref_free_pairwise"  # or "ref_guided_pairwise"

DEFAULT_RATER_MODELS: List[str] = ["llama3.3:70b"]
MAX_RETRIES = 5
RETRY_BACKOFF_SECONDS = 1.0


# ---------------------------------------------------------------------
# Prompt utilities
# ---------------------------------------------------------------------


def load_prompts(path: Path) -> Dict[str, Any]:
    if not path.exists():
        msg = f"Prompt file not found: {path}"
        raise FileNotFoundError(msg)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_pairwise_prompt(prompts: Dict[str, Any], prompt_key: str) -> str:
    try:
        return prompts["pairwise_optional"][prompt_key]
    except KeyError as exc:
        msg = f"Pairwise prompt not found for key='{prompt_key}' in evaluator_prompts.json"
        raise KeyError(msg) from exc


# ---------------------------------------------------------------------
# Human scores → pairwise winners and win-rates
# ---------------------------------------------------------------------


def scores_to_pairwise(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    """
    Convert per-model scores into pairwise comparison records.

    Returns columns:
        question_id, model_1, model_2, score_1, score_2,
        model_pairs (sorted 'modelA,modelB'), winner_model ('model' or 'tie')
    """
    comparisons: List[Dict[str, Any]] = []

    for question_id, group in df.groupby("question_id"):
        models = group["model"].tolist()
        scores = group[score_col].tolist()
        model_scores = dict(zip(models, scores))

        for m1, m2 in combinations(models, 2):
            s1 = model_scores[m1]
            s2 = model_scores[m2]

            if s1 > s2:
                winner = m1
            elif s2 > s1:
                winner = m2
            else:
                winner = "tie"

            comparisons.append(
                {
                    "question_id": question_id,
                    "model_1": m1,
                    "model_2": m2,
                    "score_1": s1,
                    "score_2": s2,
                    "model_pairs": ",".join(sorted([m1, m2])),
                    "winner_model": winner,
                },
            )

    return pd.DataFrame(comparisons)


def compute_winrate_from_pairwise(df_pairwise: pd.DataFrame) -> pd.DataFrame:
    """
    Compute win-rate per model from pairwise results.

    Assumes columns:
        model_pairs (modelA,modelB), winner_model ('modelA','modelB','tie')
    """
    records: List[Dict[str, Any]] = []

    for _, row in df_pairwise.iterrows():
        model1, model2 = row["model_pairs"].split(",")
        winner_model = row["winner_model"]

        if winner_model == model1:
            records.append({"model": model1, "win": 1.0, "total": 1.0})
            records.append({"model": model2, "win": 0.0, "total": 1.0})
        elif winner_model == model2:
            records.append({"model": model1, "win": 0.0, "total": 1.0})
            records.append({"model": model2, "win": 1.0, "total": 1.0})
        elif winner_model == "tie":
            records.append({"model": model1, "win": 0.5, "total": 1.0})
            records.append({"model": model2, "win": 0.5, "total": 1.0})
        else:
            # Unknown verdict
            continue

    df_records = pd.DataFrame(records)
    if df_records.empty:
        return pd.DataFrame(columns=["model", "win", "total", "win_rate"])

    summary = (
        df_records.groupby("model")
        .agg({"win": "sum", "total": "sum"})
        .reset_index()
    )
    summary["win_rate"] = summary["win"] / summary["total"]
    return summary


# ---------------------------------------------------------------------
# Pairwise LLM evaluator
# ---------------------------------------------------------------------


def clean_llm_output(raw: str, rater_model: str) -> str:
    """
    Clean raw LLM output to increase chance of valid JSON parsing.

    - Strip chain-of-thought for some models.
    - Remove control chars and markdown fences.
    """
    cot_models = {
        "deepseek-r1:14b",
        "qwen3:32b",
        "qwen3_32b_temp_0",
        "deepseek_r1_14b_temp_0",
    }

    text = raw

    if rater_model.lower() in cot_models:
        # Heuristic: drop everything before </think>
        if "</think>" in text:
            text = text.split("</think>", maxsplit=1)[-1]

    # Remove control chars
    text = re.sub(r"[\x00-\x1F]+", " ", text)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Fix escaped quotes
    text = text.replace('\\"', '"')

    # Strip markdown fences (```json ... ```)
    text = re.sub(r"^```?json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)

    return text.strip()


def parse_pairwise_json(raw: str) -> Tuple[str, str]:
    """
    Parse pairwise verdict JSON.

    Expected:
    {
      "winner": "A" | "B" | "tie",
      "explanation": "..."
    }

    Returns (winner, explanation).
    On failure, returns ("Error", error_message).
    """
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        return "Error", f"JSONDecodeError: {e}"

    winner = data.get("winner")
    explanation = data.get("explanation", "")

    if winner not in {"A", "B", "tie"}:
        return "Error", f"Invalid 'winner' field: {winner!r}"

    return winner, str(explanation)


def call_pairwise_llm(
    rater_model: str,
    system_prompt: str,
    question: str,
    answer_1: str,
    answer_2: str,
    max_retries: int = MAX_RETRIES,
) -> Tuple[str, str, float, str]:
    """
    Call LLM for pairwise comparison.

    Returns:
        final_winner: "Answer 1" | "Answer 2" | "tie" | "Error"
        explanation: text (or error)
        elapsed_time: seconds (NaN on failure)
        raw_output: original LLM response text
    """
    user_prompt = (
        "Question:\n"
        f"{question}\n\n"
        "Answer A:\n"
        f"{answer_1}\n\n"
        "Answer B:\n"
        f"{answer_2}\n\n"
        "Please follow the instructions and return only the JSON verdict."
    )

    last_error: Optional[str] = None

    for attempt in range(1, max_retries + 1):
        try:
            time.sleep(RETRY_BACKOFF_SECONDS)
            start_time = time.time()

            response = chat(
                model=rater_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )

            elapsed = time.time() - start_time
            raw_output = (
                response["message"]["content"].strip()
                if isinstance(response, dict)
                else response.message["content"].strip()
            )

            cleaned = clean_llm_output(raw_output, rater_model)
            winner, explanation = parse_pairwise_json(cleaned)

            # Map JSON "winner" to more readable labels
            if winner == "A":
                final_winner = "Answer 1"
            elif winner == "B":
                final_winner = "Answer 2"
            elif winner == "tie":
                final_winner = "tie"
            else:
                final_winner = "Error"

            return final_winner, explanation, elapsed, raw_output

        except Exception as e:  # noqa: BLE001
            last_error = str(e)
            logging.warning(
                "Pairwise LLM call failed (attempt %d/%d): %s",
                attempt,
                max_retries,
                last_error,
            )

    logging.error(
        "Pairwise LLM call failed after %d attempts. Last error: %s",
        max_retries,
        last_error,
    )
    return "Error", f"Error: {last_error}", math.nan, ""


def pairwise_evaluator(
    df: pd.DataFrame,
    answer_1_col: str,
    answer_2_col: str,
    rater_model: str,
    system_prompt: str,
) -> pd.DataFrame:
    """
    Evaluate which of two answers is better for each question using an LLM.

    Adds columns:
        output, final_verdict, explanation, response_time,
        answer_1, answer_2, winner_model
    """
    df = df.copy()

    outputs: List[str] = []
    verdicts: List[str] = []
    explanations: List[str] = []
    response_times: List[float] = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating answer pairs"):
        question = str(row["question"])
        ans1 = str(row[answer_1_col])
        ans2 = str(row[answer_2_col])

        verdict, explanation, elapsed, raw = call_pairwise_llm(
            rater_model=rater_model,
            system_prompt=system_prompt,
            question=question,
            answer_1=ans1,
            answer_2=ans2,
        )

        outputs.append(raw)
        verdicts.append(verdict)
        explanations.append(explanation)
        response_times.append(elapsed)

    df["output"] = outputs
    df["final_verdict"] = verdicts
    df["explanation"] = explanations
    df["response_time"] = response_times

    # Track which columns were used as answer_1 / answer_2
    df["answer_1"] = answer_1_col
    df["answer_2"] = answer_2_col

    def get_winner_model(row: pd.Series) -> str:
        if row["final_verdict"] == "Answer 1":
            # if the answers come from wide-format columns (e.g., "<model>_answer")
            return row["answer_1"].replace("_answer", "")
        if row["final_verdict"] == "Answer 2":
            return row["answer_2"].replace("_answer", "")
        if row["final_verdict"] == "tie":
            return "tie"
        return "Error"

    df["winner_model"] = df.apply(get_winner_model, axis=1)
    return df


def pairwise_evaluator_swap(
    df: pd.DataFrame,
    answer_1_col: str,
    answer_2_col: str,
    rater_model: str,
    system_prompt: str,
) -> pd.DataFrame:
    """
    Run pairwise evaluator twice: default and swapped answer order,
    to reduce positional bias.
    """
    df_default = pairwise_evaluator(
        df,
        answer_1_col,
        answer_2_col,
        rater_model=rater_model,
        system_prompt=system_prompt,
    )
    df_swap = pairwise_evaluator(
        df,
        answer_2_col,
        answer_1_col,
        rater_model=rater_model,
        system_prompt=system_prompt,
    )
    df_default["position"] = "default"
    df_swap["position"] = "swap"

    return pd.concat([df_default, df_swap], ignore_index=True)


def pairwise_evaluator_all_pairs(
    df: pd.DataFrame,
    rater_model: str,
    system_prompt: str,
) -> pd.DataFrame:
    """
    Given a long-format DataFrame with columns:
        question_id, question, model, answer
    run pairwise LLM evaluation for all model pairs per question.
    """
    models = df["model"].unique()
    model_pairs = list(combinations(models, 2))

    all_results: List[pd.DataFrame] = []

    for model1, model2 in model_pairs:
        df1 = (
            df[df["model"] == model1][["question_id", "question", "answer"]]
            .rename(columns={"answer": f"{model1}_answer"})
        )
        df2 = (
            df[df["model"] == model2][["question_id", "answer"]]
            .rename(columns={"answer": f"{model2}_answer"})
        )

        df_pair = pd.merge(df1, df2, on="question_id")
        df_pair["model_pairs"] = f"{model1},{model2}"

        evaluated = pairwise_evaluator_swap(
            df_pair,
            answer_1_col=f"{model1}_answer",
            answer_2_col=f"{model2}_answer",
            rater_model=rater_model,
            system_prompt=system_prompt,
        )

        # Drop the wide answer columns to avoid clutter
        evaluated = evaluated.drop(columns=[f"{model1}_answer", f"{model2}_answer"])
        all_results.append(evaluated)

    if not all_results:
        return pd.DataFrame()

    return pd.concat(all_results, ignore_index=True)


def normalize_model_pair(pair: str) -> str:
    return ",".join(sorted(pair.split(",")))


# ---------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------


def run_pipeline(
    human_scores_path: Path,
    answers_path: Path,
    output_prefix: Path,
    prompts_path: Path,
    pairwise_prompt_key: str,
    rater_models: List[str],
    human_score_col: str,
    model_order: Optional[List[str]] = None,
) -> None:
    """
    Full pipeline:
      - load data
      - compute human pairwise & win-rate
      - run pairwise LLM eval for each rater model
      - compute agreement + Kendall's tau
      - save outputs with timestamps
    """
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    logging.info("Loading human scores from: %s", human_scores_path)
    df_human = pd.read_csv(human_scores_path)

    logging.info("Loading model answers from: %s", answers_path)
    df_answers = pd.read_csv(answers_path)

    # Merge to get a long-format df with question_id, question, model, answer, etc.
    df = df_human.merge(
        df_answers,
        on=["question_id", "question"],
        how="left",
    )

    # Optionally enforce model order as a Categorical
    if model_order is not None:
        df["model"] = pd.Categorical(
            df["model"],
            categories=model_order,
            ordered=True,
        )

    # Human pairwise & win-rates
    logging.info("Computing human pairwise outcomes...")
    df_human_pairwise = scores_to_pairwise(df, score_col=human_score_col)
    df_human_pairwise["model_pairs"] = df_human_pairwise["model_pairs"].apply(
        normalize_model_pair,
    )

    df_human_winrate = compute_winrate_from_pairwise(df_human_pairwise)
    logging.info("Human win-rates:\n%s", df_human_winrate)

    # Save human winrates
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    human_winrate_path = output_prefix.with_name(
        f"{output_prefix.name}_human_winrate_{ts}.csv",
    )
    df_human_winrate.to_csv(human_winrate_path, index=False)

    prompts = load_prompts(prompts_path)
    system_prompt = get_pairwise_prompt(prompts, pairwise_prompt_key)

    log_file = output_prefix.with_name(
        f"{output_prefix.name}_elapsed_times_{ts}.txt",
    )

    for rater_model in rater_models:
        logging.info("=== Rater model: %s ===", rater_model)
        start_time = time.time()

        df_all_pairwise = pairwise_evaluator_all_pairs(
            df=df,
            rater_model=rater_model,
            system_prompt=system_prompt,
        )

        elapsed = time.time() - start_time
        logging.info(
            "Elapsed time for %s: %.2f seconds",
            rater_model,
            elapsed,
        )
        with log_file.open("a", encoding="utf-8") as f:
            f.write(f"{rater_model}: {elapsed:.2f} seconds\n")

        # Normalize model_pairs to match human
        df_all_pairwise["model_pairs"] = df_all_pairwise["model_pairs"].apply(
            normalize_model_pair,
        )
        df_human_pairwise["model_pairs"] = df_human_pairwise["model_pairs"].apply(
            normalize_model_pair,
        )

        # Save raw pairwise decisions
        pairwise_out_path = output_prefix.with_name(
            f"{output_prefix.name}_pairwise_raw_{rater_model}_{ts}.csv",
        )
        df_all_pairwise.to_csv(pairwise_out_path, index=False)

        # Compute LLM win-rates
        df_llm_winrate = compute_winrate_from_pairwise(df_all_pairwise)
        if model_order is not None:
            df_llm_winrate["model"] = pd.Categorical(
                df_llm_winrate["model"],
                categories=model_order,
                ordered=True,
            )
            df_llm_winrate = df_llm_winrate.sort_values("model")

        winrate_out_path = output_prefix.with_name(
            f"{output_prefix.name}_winrate_{rater_model}_{ts}.csv",
        )
        df_llm_winrate.to_csv(winrate_out_path, index=False)

        # Merge with human win-rates
        df_winrate_all = df_llm_winrate[["model", "win_rate"]].rename(
            columns={"win_rate": "win_rate_llm"},
        ).merge(
            df_human_winrate[["model", "win_rate"]].rename(
                columns={"win_rate": "win_rate_human"},
            ),
            on="model",
            how="left",
        )
        winrate_all_out_path = output_prefix.with_name(
            f"{output_prefix.name}_winrate_vs_human_{rater_model}_{ts}.csv",
        )
        df_winrate_all.to_csv(winrate_all_out_path, index=False)

        logging.info("LLM vs human win-rates:\n%s", df_winrate_all)

        # Agreement with human pairwise winners
        df_llm_votes = df_all_pairwise[["question_id", "model_pairs", "winner_model"]]
        df_llm_votes = df_llm_votes.rename(
            columns={"winner_model": f"{rater_model}_vote"},
        )

        df_pairwise_join = df_llm_votes.merge(
            df_human_pairwise[["question_id", "model_pairs", "winner_model"]],
            on=["question_id", "model_pairs"],
            how="left",
            suffixes=("", "_human"),
        )
        df_pairwise_join["agree"] = (
            df_pairwise_join[f"{rater_model}_vote"]
            == df_pairwise_join["winner_model_human"]
        )
        agreement_rate = df_pairwise_join["agree"].mean()
        logging.info("Agreement with human pairwise decisions: %.3f", agreement_rate)

        # Kendall's tau between win-rate vectors
        valid = df_winrate_all.dropna(subset=["win_rate_human", "win_rate_llm"])
        if len(valid) >= 2:
            tau, _ = kendalltau(
                valid["win_rate_human"].to_numpy(dtype=float),
                valid["win_rate_llm"].to_numpy(dtype=float),
            )
            logging.info("Kendall's tau (LLM vs human win-rate): %.3f", tau)
        else:
            logging.info("Not enough models for Kendall's tau.")

    logging.info("Elapsed times logged to: %s", log_file)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run pairwise LLM evaluation and compare against human scores.",
    )
    parser.add_argument(
        "--human-scores",
        type=Path,
        default=DEFAULT_HUMAN_SCORES,
        help=f"Path to CSV with human scores (default: {DEFAULT_HUMAN_SCORES})",
    )
    parser.add_argument(
        "--answers",
        type=Path,
        default=DEFAULT_ANSWERS,
        help=f"Path to CSV with model answers (default: {DEFAULT_ANSWERS})",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=DEFAULT_OUTPUT_PREFIX,
        help=f"Prefix for output files (default: {DEFAULT_OUTPUT_PREFIX})",
    )
    parser.add_argument(
        "--prompts-path",
        type=Path,
        default=DEFAULT_PROMPTS_PATH,
        help=f"Path to evaluator_prompts.json (default: {DEFAULT_PROMPTS_PATH})",
    )
    parser.add_argument(
        "--pairwise-mode",
        type=str,
        default=DEFAULT_PAIRWISE_MODE,
        choices=["ref_free_pairwise", "ref_guided_pairwise"],
        help="Which pairwise prompt to use from evaluator_prompts.json.",
    )
    parser.add_argument(
        "--rater-models",
        type=str,
        default=",".join(DEFAULT_RATER_MODELS),
        help="Comma-separated list of evaluator (rater) models to run.",
    )
    parser.add_argument(
        "--human-score-col",
        type=str,
        default="accu_bs",
        help="Column name for human scores to convert into pairwise comparisons.",
    )
    parser.add_argument(
        "--model-order",
        type=str,
        default="",
        help=(
            "Optional comma-separated list of models to impose an order on "
            "for outputs (e.g., 'gpt4,llama3.2,phi4')."
        ),
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    args = parse_args()

    rater_models = [m.strip() for m in args.rater_models.split(",") if m.strip()]
    model_order = (
        [m.strip() for m in args.model_order.split(",") if m.strip()]
        if args.model_order
        else None
    )

    logging.info("Human scores:   %s", args.human_scores)
    logging.info("Answers:        %s", args.answers)
    logging.info("Output prefix:  %s", args.output_prefix)
    logging.info("Prompts path:   %s", args.prompts_path)
    logging.info("Pairwise mode:  %s", args.pairwise_mode)
    logging.info("Rater models:   %s", ", ".join(rater_models))
    logging.info("Human score col:%s", args.human_score_col)
    logging.info("Model order:    %s", model_order)

    run_pipeline(
        human_scores_path=args.human_scores,
        answers_path=args.answers,
        output_prefix=args.output_prefix,
        prompts_path=args.prompts_path,
        pairwise_prompt_key=args.pairwise_mode,
        rater_models=rater_models,
        human_score_col=args.human_score_col,
        model_order=model_order,
    )


if __name__ == "__main__":
    main()
