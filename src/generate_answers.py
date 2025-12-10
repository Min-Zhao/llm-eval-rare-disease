#!/usr/bin/env python
"""
generate_answers.py

Generate LLM answers for a list of questions using local Ollama models.
- Reads a CSV with a `question` column.
- Calls one or more Ollama models.
- Saves answers and response times.
- Is resume-friendly via a temporary output file.
"""

from __future__ import annotations

import argparse
import logging
import math
import time
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
from ollama import chat

# ---------------------------------------------------------------------
# Default configuration (can be overridden via CLI)
# ---------------------------------------------------------------------
DEFAULT_INPUT = Path("data/cla_qa_questions.csv")
DEFAULT_TEMP = Path("results/cla_qa_answers_temp.csv")
DEFAULT_FINAL = Path("results/cla_qa_answers.csv")

DEFAULT_MODELS: List[str] = [
    "phi4",
    "mistral",
    "gemma3:27b",
    "qwen3:32b",
    "llama3.3:70b",
    "deepseek-r1:70b"
]

MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2.0
CHECKPOINT_EVERY = 50  # save temp every N newly answered questions per model


# ---------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------
def generate_response(model: str, question: str) -> Tuple[str, float]:
    """
    Call the Ollama chat() API and return (answer, elapsed_time_in_seconds).

    On repeated failure, returns:
        answer = 'ERROR: <message>'
        elapsed_time = NaN
    """
    last_error: Optional[str] = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            start = time.time()
            response = chat(
                model=model,
                messages=[{"role": "user", "content": question}],
            )
            elapsed = time.time() - start

            # `response` can be dict-like or object-like depending on version
            if isinstance(response, dict):
                output = response["message"]["content"].strip()
            else:
                output = response.message["content"].strip()

            return output, elapsed

        except Exception as e:  # noqa: BLE001
            last_error = str(e)
            logging.warning(
                "Model %s failed on attempt %d/%d: %s",
                model,
                attempt,
                MAX_RETRIES,
                last_error,
            )
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_SECONDS)

    # If we get here, all attempts failed
    logging.error(
        "Model %s failed after %d attempts. Last error: %s",
        model,
        MAX_RETRIES,
        last_error,
    )
    return f"ERROR: {last_error}", math.nan


def load_questions(input_path: Path, temp_path: Path) -> pd.DataFrame:
    """
    Load questions DataFrame.

    If a temp file exists, resume from it.
    Otherwise, load from the input CSV.

    The CSV must contain a 'question' column.
    """
    if temp_path.exists():
        logging.info("Found temp file: %s. Resuming from it.", temp_path)
        df = pd.read_csv(temp_path)
    else:
        logging.info("Loading input questions from: %s", input_path)
        df = pd.read_csv(input_path)

    if "question" not in df.columns:
        raise ValueError("Input CSV must contain a 'question' column.")

    return df


def ensure_model_columns(df: pd.DataFrame, model: str) -> None:
    """
    Ensure the answer and response-time columns for a model exist in the DataFrame.
    """
    answer_col = f"{model}_answer"
    time_col = f"{model}_response_time"

    if answer_col not in df.columns:
        df[answer_col] = ""
    if time_col not in df.columns:
        df[time_col] = math.nan


# ---------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------
def run_generation(
    input_path: Path,
    temp_path: Path,
    final_path: Path,
    models: List[str],
) -> None:
    """
    Main generation loop across all models.
    """
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    final_path.parent.mkdir(parents=True, exist_ok=True)

    df = load_questions(input_path, temp_path)

    for model in models:
        logging.info("=== Starting model: %s ===", model)
        answer_col = f"{model}_answer"
        time_col = f"{model}_response_time"

        ensure_model_columns(df, model)

        since_ckpt = 0

        # Iterate by index so we can assign with .at[]
        for idx in range(len(df)):
            question = str(df.at[idx, "question"])

            # Skip if already filled (resume-friendly)
            existing = df.at[idx, answer_col]
            if isinstance(existing, str) and existing.strip():
                continue

            answer, elapsed = generate_response(model, question)
            df.at[idx, answer_col] = answer
            df.at[idx, time_col] = elapsed

            since_ckpt += 1
            if since_ckpt >= CHECKPOINT_EVERY:
                logging.info(
                    "Checkpoint: saving temp results after %d new answers for model %s",
                    since_ckpt,
                    model,
                )
                df.to_csv(temp_path, index=False)
                since_ckpt = 0

        # Save after each model finishes
        logging.info("Finished model %s. Saving temp results.", model)
        df.to_csv(temp_path, index=False)

    # Final save
    logging.info("All models completed. Writing final output to: %s", final_path)
    df.to_csv(final_path, index=False)
    logging.info("Done.")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate LLM answers for rare-disease QA using local Ollama models.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Path to input questions CSV (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--temp",
        type=Path,
        default=DEFAULT_TEMP,
        help=f"Path to temp CSV for resumable runs (default: {DEFAULT_TEMP})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_FINAL,
        help=f"Path to final output CSV (default: {DEFAULT_FINAL})",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=",".join(DEFAULT_MODELS),
        help="Comma-separated list of Ollama model names to run.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    args = parse_args()
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    logging.info("Input:  %s", args.input)
    logging.info("Temp:   %s", args.temp)
    logging.info("Output: %s", args.output)
    logging.info("Models: %s", ", ".join(models))

    run_generation(args.input, args.temp, args.output, models)


if __name__ == "__main__":
    main()
