#!/usr/bin/env python
"""
run_llm_evaluator.py

Run an LLM-based evaluator over a CSV of QA pairs.

- Supports reference-free and reference-guided evaluation modes.
- Loads instruction prompts from prompts/evaluator_prompts.json.
- Expects a CSV with at least:
    - question column
    - answer column (the answer to be evaluated)
    - optional reference_answer column (for reference-guided mode)
- Writes evaluator score, explanation, and raw response for each row.

Example usage:

python src/run_llm_evaluator.py \
  --input results/cla_qa_answers.csv \
  --output results/cla_qa_eval_scores_rg.csv \
  --mode ref_guided \
  --prompt-key accuracy_with_definition_long_instruction \
  --model gpt-oss:20b \
  --answer-col gpt4_answer
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from ollama import chat

# ---------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------

DEFAULT_INPUT = Path("results/cla_qa_answers.csv")
DEFAULT_OUTPUT = Path("results/cla_qa_eval_scores.csv")
DEFAULT_TEMP = Path("results/cla_qa_eval_scores_temp.csv")

DEFAULT_PROMPTS_PATH = Path("prompts/evaluator_prompts.json")

DEFAULT_MODE = "ref_free"  # or "ref_guided"
DEFAULT_PROMPT_KEY = "accuracy_with_definition_and_rubric"
DEFAULT_MODEL_NAME = "gpt-oss:20b"

MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2.0
CHECKPOINT_EVERY = 50  # save temp every N new evaluations


# ---------------------------------------------------------------------
# Prompt utilities
# ---------------------------------------------------------------------


def load_prompts(path: Path) -> Dict[str, Any]:
    """Load evaluator prompts from a JSON file."""
    if not path.exists():
        msg = f"Prompt file not found: {path}"
        raise FileNotFoundError(msg)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_instruction_prompt(
    prompts: Dict[str, Any],
    mode: str,
    prompt_key: str,
) -> str:
    """
    Retrieve the instruction/system prompt text from the JSON structure.

    mode:        "ref_free" or "ref_guided"
    prompt_key:  e.g., "simple", "accuracy_with_definition", etc.
    """
    try:
        return prompts[mode][prompt_key]
    except KeyError as exc:
        msg = f"Prompt not found for mode='{mode}', key='{prompt_key}' in evaluator_prompts.json"
        raise KeyError(msg) from exc


# ---------------------------------------------------------------------
# LLM call + JSON parsing
# ---------------------------------------------------------------------


def build_user_message(
    mode: str,
    question: str,
    answer: str,
    reference_answer: Optional[str] = None,
) -> str:
    """
    Build the user message content sent to the evaluator LLM.

    This keeps the JSON prompts in evaluator_prompts.json as the "instruction"
    and uses this as the 'user' content with concrete question/answers.
    """
    if mode == "ref_free":
        return (
            "[Question]\n"
            f"{question}\n\n"
            "[Model Answer]\n"
            f"{answer}\n\n"
            "Please evaluate this answer according to the instructions."
        )

    if mode == "ref_guided":
        ref = reference_answer or ""
        return (
            "[Question]\n"
            f"{question}\n\n"
            "[Reference Answer]\n"
            f"{ref}\n\n"
            "[Model Answer]\n"
            f"{answer}\n\n"
            "Please evaluate this answer according to the instructions."
        )

    msg = f"Unsupported mode: {mode}. Expected 'ref_free' or 'ref_guided'."
    raise ValueError(msg)


def call_evaluator_llm(
    model: str,
    system_prompt: str,
    user_prompt: str,
) -> Tuple[str, float]:
    """
    Call the evaluator LLM via Ollama and return (raw_text_response, elapsed_seconds).

    On repeated failure, returns:
        raw_text_response = 'ERROR: <message>'
        elapsed_seconds   = NaN
    """
    last_error: Optional[str] = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            start = time.time()
            response = chat(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            elapsed = time.time() - start

            if isinstance(response, dict):
                text = response["message"]["content"].strip()
            else:
                text = response.message["content"].strip()

            return text, elapsed

        except Exception as e:  # noqa: BLE001
            last_error = str(e)
            logging.warning(
                "Evaluator model %s failed on attempt %d/%d: %s",
                model,
                attempt,
                MAX_RETRIES,
                last_error,
            )
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_SECONDS)

    logging.error(
        "Evaluator model %s failed after %d attempts. Last error: %s",
        model,
        MAX_RETRIES,
        last_error,
    )
    return f"ERROR: {last_error}", math.nan


def parse_eval_json(raw: str) -> Tuple[Optional[float], Optional[str]]:
    """
    Parse the evaluator's JSON output.

    Expected schema:
    {
        "score": <int or float 1..5>,
        "explanation": "<short text>"
    }

    Returns (score, explanation). If parsing fails, returns (None, None).
    """
    try:
        # Try direct JSON loading first
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Heuristic: try to extract JSON substring between first '{' and last '}'
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1 or end <= start:
            logging.warning("Could not find JSON object in response.")
            return None, None
        try:
            data = json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            logging.warning("Failed to parse JSON after substring extraction.")
            return None, None

    score = data.get("score")
    explanation = data.get("explanation")

    # Basic validation
    try:
        if score is not None:
            score = float(score)
    except (TypeError, ValueError):
        logging.warning("Invalid 'score' field type in evaluator output: %r", score)
        score = None

    if explanation is not None:
        explanation = str(explanation)

    return score, explanation


# ---------------------------------------------------------------------
# DataFrame loop
# ---------------------------------------------------------------------


def load_answers(input_path: Path, temp_path: Path) -> pd.DataFrame:
    """
    Load the input data (answers for evaluation).

    If a temp file exists, resume from it; otherwise load from input CSV.
    """
    if temp_path.exists():
        logging.info("Found temp file: %s. Resuming from it.", temp_path)
        return pd.read_csv(temp_path)

    logging.info("Loading input answers from: %s", input_path)
    return pd.read_csv(input_path)


def run_evaluator(
    input_path: Path,
    output_path: Path,
    temp_path: Path,
    prompts_path: Path,
    mode: str,
    prompt_key: str,
    model_name: str,
    question_col: str,
    answer_col: str,
    reference_col: Optional[str],
) -> None:
    """
    Main evaluation loop.

    For each row, builds a user prompt and calls the evaluator model.
    Adds columns:
        <model>_eval_score
        <model>_eval_explanation
        <model>_eval_raw
        <model>_eval_time
    """
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    prompts = load_prompts(prompts_path)
    system_prompt = get_instruction_prompt(prompts, mode, prompt_key)

    df = load_answers(input_path, temp_path)

    # Column names for this evaluator/model
    score_col = f"{model_name}_eval_score"
    expl_col = f"{model_name}_eval_explanation"
    raw_col = f"{model_name}_eval_raw"
    time_col = f"{model_name}_eval_time"

    # Ensure columns exist
    for col in (score_col, expl_col, raw_col, time_col):
        if col not in df.columns:
            df[col] = pd.NA

    since_ckpt = 0

    for idx in range(len(df)):
        # Skip if this row already evaluated (resume-friendly)
        if pd.notna(df.at[idx, score_col]):
            continue

        question = str(df.at[idx, question_col])
        answer = str(df.at[idx, answer_col])

        ref_answer: Optional[str] = None
        if mode == "ref_guided":
            if reference_col is None:
                msg = "reference_col must be provided for ref_guided mode."
                raise ValueError(msg)
            ref_answer = str(df.at[idx, reference_col])

        user_prompt = build_user_message(
            mode=mode,
            question=question,
            answer=answer,
            reference_answer=ref_answer,
        )

        raw_response, elapsed = call_evaluator_llm(
            model=model_name,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        score, explanation = parse_eval_json(raw_response)

        df.at[idx, score_col] = score
        df.at[idx, expl_col] = explanation
        df.at[idx, raw_col] = raw_response
        df.at[idx, time_col] = elapsed

        since_ckpt += 1
        if since_ckpt >= CHECKPOINT_EVERY:
            logging.info(
                "Checkpoint: saving temp results after %d new evaluations",
                since_ckpt,
            )
            df.to_csv(temp_path, index=False)
            since_ckpt = 0

    logging.info("Finished all evaluations. Writing temp and final outputs.")
    df.to_csv(temp_path, index=False)
    df.to_csv(output_path, index=False)
    logging.info("Done. Final results saved to: %s", output_path)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LLM-based evaluator over QA answers.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Path to input CSV containing questions and answers (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Path to final output CSV (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--temp",
        type=Path,
        default=DEFAULT_TEMP,
        help=f"Path to temp CSV for resumable runs (default: {DEFAULT_TEMP})",
    )
    parser.add_argument(
        "--prompts-path",
        type=Path,
        default=DEFAULT_PROMPTS_PATH,
        help=f"Path to evaluator_prompts.json (default: {DEFAULT_PROMPTS_PATH})",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["ref_free", "ref_guided"],
        default=DEFAULT_MODE,
        help="Evaluation mode: 'ref_free' or 'ref_guided'.",
    )
    parser.add_argument(
        "--prompt-key",
        type=str,
        default=DEFAULT_PROMPT_KEY,
        help=(
            "Prompt key under the selected mode in evaluator_prompts.json "
            "(e.g., 'simple', 'accuracy', 'accuracy_with_definition_and_rubric')."
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help=f"Evaluator model name for Ollama (default: {DEFAULT_MODEL_NAME}).",
    )
    parser.add_argument(
        "--question-col",
        type=str,
        default="question",
        help="Name of the question column in the input CSV.",
    )
    parser.add_argument(
        "--answer-col",
        type=str,
        default="answer",
        help="Name of the answer column to be evaluated in the input CSV.",
    )
    parser.add_argument(
        "--reference-col",
        type=str,
        default="reference_answer",
        help=(
            "Name of the reference answer column (used only in ref_guided mode). "
            "Ignored in ref_free mode."
        ),
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    args = parse_args()

    logging.info("Input:        %s", args.input)
    logging.info("Output:       %s", args.output)
    logging.info("Temp:         %s", args.temp)
    logging.info("Prompts path: %s", args.prompts_path)
    logging.info("Mode:         %s", args.mode)
    logging.info("Prompt key:   %s", args.prompt_key)
    logging.info("Model:        %s", args.model)
    logging.info("Question col: %s", args.question_col)
    logging.info("Answer col:   %s", args.answer_col)
    logging.info("Reference col:%s", args.reference_col)

    run_evaluator(
        input_path=args.input,
        output_path=args.output,
        temp_path=args.temp,
        prompts_path=args.prompts_path,
        mode=args.mode,
        prompt_key=args.prompt_key,
        model_name=args.model,
        question_col=args.question_col,
        answer_col=args.answer_col,
        reference_col=args.reference_col if args.mode == "ref_guided" else None,
    )


if __name__ == "__main__":
    main()
