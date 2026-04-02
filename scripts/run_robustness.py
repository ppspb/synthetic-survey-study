from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.data_utils import load_samples, build_subset, attach_human_scores_from_full, resolve_data_path
from src.io_utils import ensure_dir, append_jsonl, append_tsv_row, load_jsonl, read_tsv_if_exists, write_json
from src.logging_utils import setup_logging
from src.robustness import (
    build_client,
    load_meta,
    robustness_execution_key,
    _run_row_with_recovery,
    summarize_robustness,
)
from src.prompt_variants import apply_question_perturbation
from src.metadata import get_question_scale, get_question_text
from src.run_utils import create_run_dir, write_latest_pointer, write_run_manifest, apply_test_overrides, apply_runtime_overrides

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--modes", nargs="*", default=None, help="Optional subset of prompt modes")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--disable-llm-judge", action="store_true")
    parser.add_argument("--test-run", action="store_true", help="Run a reduced robustness suite")
    parser.add_argument("--test-questions", type=int, default=2)
    parser.add_argument("--test-groups", type=int, default=1)
    parser.add_argument("--test-participants", type=int, default=2)
    parser.add_argument("--resume-run-dir", default=None, help="Continue writing into an existing robustness run dir")
    parser.add_argument("--stop-after-rows", type=int, default=None, help="Debug option: stop after N new rows to test resume")
    parser.add_argument("--model", default=None, help="Override api.model from YAML")
    parser.add_argument("--base-url", default=None, help="Override api.base_url from YAML")
    parser.add_argument("--api-key", default=None, help="Override api.api_key from YAML")
    parser.add_argument("--outputs-dir", default=None, help="Override paths.outputs_dir from YAML")
    return parser.parse_args()


def _prepare_run_dir(config, args):
    ensure_dir(config.paths.outputs_dir)
    if args.resume_run_dir:
        run_dir = Path(args.resume_run_dir)
        ensure_dir(run_dir)
        run_id = run_dir.name
        log_path = setup_logging(run_dir / "logs", "robustness_resume")
        logger.info("Resuming existing robustness run_dir=%s", run_dir)
    else:
        run_id, run_dir = create_run_dir(config.paths.outputs_dir, "robustness", run_name=args.run_name)
        log_path = setup_logging(run_dir / "logs", "robustness")
        write_latest_pointer(config.paths.outputs_dir, "robustness", run_dir)
    return run_id, run_dir, log_path


def _summary_from_files(run_dir: Path, compare_to: str) -> dict:
    pred_df = read_tsv_if_exists(run_dir / "robustness_predictions.tsv")
    raw_rows = load_jsonl(run_dir / "robustness_raw.jsonl")
    return summarize_robustness(pred_df, raw_rows, compare_to=compare_to)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config = apply_runtime_overrides(config, model=args.model, base_url=args.base_url, api_key=args.api_key, outputs_dir=args.outputs_dir)

    if args.test_run:
        selected_modes = args.modes if args.modes else config.benchmark.prompt_modes[:1]
        selected_perturbations = ["original", "question_paraphrase"]
        config = apply_test_overrides(
            config,
            max_questions=args.test_questions,
            max_groups=args.test_groups,
            participants_per_group_question=args.test_participants,
            modes=selected_modes,
            perturbations=selected_perturbations,
            disable_llm_judge=args.disable_llm_judge or True,
            mock=args.mock,
        )
        if not args.run_name and not args.resume_run_dir:
            args.run_name = "test"
    else:
        config = apply_test_overrides(
            config,
            modes=args.modes,
            disable_llm_judge=args.disable_llm_judge,
            mock=args.mock,
        )

    run_id, run_dir, log_path = _prepare_run_dir(config, args)
    modes = config.benchmark.prompt_modes
    logger.info("Starting robustness run_id=%s config=%s modes=%s run_dir=%s", run_id, args.config, modes, run_dir)

    samples_df = load_samples(config.paths.data_dir / config.paths.samples_file)
    subset_df = build_subset(
        samples_df=samples_df,
        question_ids=config.benchmark.questions,
        groups=config.benchmark.groups,
        participants_per_group_question=config.benchmark.participants_per_group_question,
        random_seed=config.benchmark.random_seed,
    )
    full_value_path = None
    if config.paths.full_value_qa_file:
        full_value_path = resolve_data_path(config.paths.full_value_qa_file, data_dir=config.paths.data_dir, project_root=ROOT)
    subset_df = attach_human_scores_from_full(subset_df, full_value_path)
    subset_path = run_dir / "subset.tsv"
    if not subset_path.exists():
        subset_df.to_csv(subset_path, sep="	", index=False)
        logger.info("Saved subset to %s", subset_path)

    write_run_manifest(run_dir, {
        "run_id": run_id,
        "script": "run_robustness.py",
        "config_path": str(args.config),
        "run_dir": str(run_dir),
        "log_path": str(log_path),
        "test_run": bool(args.test_run),
        "mock": bool(args.mock),
        "disable_llm_judge": bool(args.disable_llm_judge or args.test_run),
        "modes": modes,
        "questions": config.benchmark.questions,
        "groups": config.benchmark.groups,
        "participants_per_group_question": config.benchmark.participants_per_group_question,
        "perturbations": config.robustness.perturbations,
        "full_value_qa_file": str(full_value_path) if full_value_path else None,
        "resume_run_dir": str(args.resume_run_dir) if args.resume_run_dir else None,
        "stop_after_rows": args.stop_after_rows,
    })

    meta = load_meta(config)
    client = build_client(config)
    pred_path = run_dir / "robustness_predictions.tsv"
    raw_path = run_dir / "robustness_raw.jsonl"
    summary_path = run_dir / "robustness_summary.json"

    existing_pred = read_tsv_if_exists(pred_path)
    processed_keys = set(existing_pred["EXECUTION_KEY"].astype(str).tolist()) if not existing_pred.empty and "EXECUTION_KEY" in existing_pred.columns else set()
    logger.info("Resume status for robustness: already_completed=%d", len(processed_keys))

    tasks = []
    rows = subset_df.to_dict(orient="records")
    for mode in modes:
        for perturbation in config.robustness.perturbations:
            for row in rows:
                tasks.append((mode, perturbation, row))

    new_rows_written = 0
    for mode, perturbation, row in tqdm(tasks, desc="robustness"):
        exec_key = robustness_execution_key(row, mode, perturbation)
        if exec_key in processed_keys:
            continue
        qid = row["QUESTION_ID"]
        try:
            question_text = get_question_text(meta, qid)
            perturbed_question = apply_question_perturbation(question_text, perturbation)
            score_min, score_max = get_question_scale(meta, qid)
            pred_row, raw_row = _run_row_with_recovery(
                client=client,
                config=config,
                meta=meta,
                mode=mode,
                row=row,
                question_text=perturbed_question,
                score_min=score_min,
                score_max=score_max,
                perturbation=perturbation,
            )
            raw_row["perturbed_question"] = perturbed_question
            append_tsv_row(pred_path, pred_row)
            append_jsonl(raw_path, raw_row)
            processed_keys.add(exec_key)
            new_rows_written += 1
        except Exception as e:
            logger.exception(
                "Robustness row failed | mode=%s perturbation=%s qid=%s participant=%s error=%s",
                mode, perturbation, row.get("QUESTION_ID"), row.get("PARTICIPANT_ID"), repr(e)
            )
            error_row = {
                "status": "error",
                "ROW_KEY": f"{row.get('QUESTION_ID')}::{row.get('PARTICIPANT_ID')}::{mode}",
                "EXECUTION_KEY": exec_key,
                "QUESTION_ID": row.get("QUESTION_ID"),
                "PARTICIPANT_ID": row.get("PARTICIPANT_ID"),
                "GROUP_KEY": row.get("GROUP_KEY"),
                "HUMAN_SCORE": row.get("HUMAN_SCORE"),
                "prompt_mode": mode,
                "perturbation": perturbation,
                "error": repr(e),
                "generation_attempts": getattr(e, "attempts", []),
            }
            append_jsonl(raw_path, error_row)
            if not config.robustness.continue_on_error:
                raise
        if args.stop_after_rows is not None and new_rows_written >= args.stop_after_rows:
            logger.error("Stopping early because --stop-after-rows=%s was reached", args.stop_after_rows)
            raise RuntimeError(f"Stopped early after {new_rows_written} new rows by request")

    summary = _summary_from_files(run_dir, compare_to=config.robustness.compare_to)
    write_json(summary_path, summary)
    logger.info("Saved robustness outputs pred=%s raw=%s summary=%s", pred_path, raw_path, summary_path)
    logger.info("Robustness completed successfully run_dir=%s", run_dir)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.getLogger(__name__).exception("Fatal robustness error: %s", repr(e))
        raise
