from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.metadata import load_question_metadata
from src.eval_utils import evaluate_optional_distribution
from src.io_utils import write_json
from src.logging_utils import setup_logging
from src.data_utils import attach_human_scores_from_full, resolve_data_path

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate saved benchmark/robustness outputs without rerunning the LLM")
    parser.add_argument("--run-dir", required=True, help="Path to outputs/runs/<run_id> OR plain outputs dir containing predictions_*.tsv")
    parser.add_argument("--question-metadata", required=True, help="Path to question_metadata.json")
    parser.add_argument("--full-value-qa", default=None, help="Optional path to human data file: full_value_qa.tsv OR raw WVS csv/zip")
    parser.add_argument("--write-per-file", action="store_true", help="Write separate posthoc summaries next to each prediction file")
    return parser.parse_args()


def _attach_human_from_subset(pred_df: pd.DataFrame, subset_path: Path) -> pd.DataFrame:
    df = pred_df.copy()
    if not subset_path.exists():
        logger.info("No subset.tsv found at %s", subset_path)
        return df
    subset_df = pd.read_csv(subset_path, sep="	")
    if "HUMAN_SCORE" not in subset_df.columns:
        logger.info("subset.tsv exists but has no HUMAN_SCORE column")
        return df

    join_cols = [c for c in ["QUESTION_ID", "PARTICIPANT_ID", "GROUP_KEY"] if c in df.columns and c in subset_df.columns]
    if not join_cols:
        logger.warning("Could not find join columns between predictions and subset")
        return df

    merged = df.merge(
        subset_df[join_cols + ["HUMAN_SCORE"]].drop_duplicates(),
        on=join_cols,
        how="left",
        suffixes=("", "_FROM_SUBSET"),
    )
    if "HUMAN_SCORE_x" in merged.columns and "HUMAN_SCORE_y" in merged.columns:
        merged["HUMAN_SCORE"] = merged["HUMAN_SCORE_x"].combine_first(merged["HUMAN_SCORE_y"])
        merged = merged.drop(columns=[c for c in ["HUMAN_SCORE_x", "HUMAN_SCORE_y"] if c in merged.columns])
    elif "HUMAN_SCORE" not in merged.columns and "HUMAN_SCORE_FROM_SUBSET" in merged.columns:
        merged["HUMAN_SCORE"] = merged["HUMAN_SCORE_FROM_SUBSET"]
    logger.info("Attached HUMAN_SCORE from subset where available")
    return merged


def _load_prediction_files(run_dir: Path) -> list[Path]:
    files = sorted(run_dir.glob("predictions_*.tsv"))
    robustness = run_dir / "robustness_predictions.tsv"
    if robustness.exists():
        files.append(robustness)
    return files


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    log_path = setup_logging(run_dir / "logs", "posthoc_eval")
    logger.info("Starting posthoc evaluation run_dir=%s", run_dir)

    pred_files = _load_prediction_files(run_dir)
    if not pred_files:
        raise FileNotFoundError(f"No prediction TSV files found in {run_dir}")

    meta = load_question_metadata(Path(args.question_metadata))
    human_data = resolve_data_path(args.full_value_qa, data_dir=run_dir, project_root=ROOT) if args.full_value_qa else None
    subset_path = run_dir / "subset.tsv"

    all_results: dict[str, dict] = {
        "run_dir": str(run_dir),
        "timestamp": datetime.now().isoformat(),
        "log_path": str(log_path),
        "human_data": str(human_data) if human_data else None,
        "files": {},
    }

    for pred_path in pred_files:
        logger.info("Evaluating saved predictions file=%s", pred_path)
        pred_df = pd.read_csv(pred_path, sep="	")
        if "HUMAN_SCORE" not in pred_df.columns:
            pred_df = _attach_human_from_subset(pred_df, subset_path)
        if human_data and ("HUMAN_SCORE" not in pred_df.columns or pred_df["HUMAN_SCORE"].isna().all()):
            pred_df = attach_human_scores_from_full(pred_df, human_data)

        summary = evaluate_optional_distribution(pred_df, meta)
        summary["input_file"] = str(pred_path)
        summary["rows"] = int(len(pred_df))
        summary["has_human_score"] = bool("HUMAN_SCORE" in pred_df.columns)
        if "HUMAN_SCORE" in pred_df.columns:
            summary["non_null_human_score_rows"] = int(pred_df["HUMAN_SCORE"].notna().sum())
        all_results["files"][pred_path.name] = summary

        if args.write_per_file:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            out_path = run_dir / f"posthoc_eval_{pred_path.stem}_{ts}.json"
            write_json(out_path, summary)
            logger.info("Wrote per-file posthoc summary to %s", out_path)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out_path = run_dir / f"posthoc_eval_all_{ts}.json"
    write_json(out_path, all_results)
    logger.info("Posthoc evaluation completed: %s", out_path)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.getLogger(__name__).exception("Fatal posthoc evaluation error: %s", repr(e))
        raise
