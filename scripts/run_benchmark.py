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
from src.run_utils import create_run_dir, write_latest_pointer, write_run_manifest, apply_test_overrides, apply_runtime_overrides
from src.benchmark import (
    benchmark_row_key,
    build_client,
    load_question_metadata,
    get_question_text,
    get_question_scale,
    _run_row_with_recovery,
)
from src.eval_utils import aggregate_judge_summary, evaluate_optional_distribution

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--modes", nargs="*", default=None, help="Optional subset of prompt modes")
    parser.add_argument("--mock", action="store_true", help="Force MockClient regardless of config")
    parser.add_argument("--disable-llm-judge", action="store_true")
    parser.add_argument("--test-run", action="store_true", help="Run a small quick benchmark that is likely to finish")
    parser.add_argument("--test-questions", type=int, default=3)
    parser.add_argument("--test-groups", type=int, default=2)
    parser.add_argument("--test-participants", type=int, default=2)
    parser.add_argument("--resume-run-dir", default=None, help="Continue writing into an existing benchmark run dir")
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
        log_path = setup_logging(run_dir / "logs", "benchmark_resume")
        logger.info("Resuming existing benchmark run_dir=%s", run_dir)
    else:
        run_id, run_dir = create_run_dir(config.paths.outputs_dir, "benchmark", run_name=args.run_name)
        log_path = setup_logging(run_dir / "logs", "benchmark")
        write_latest_pointer(config.paths.outputs_dir, "benchmark", run_dir)
    return run_id, run_dir, log_path


def _summary_from_files(run_dir: Path, mode: str, meta: dict[str, dict]) -> dict:
    pred_path = run_dir / f"predictions_{mode}.tsv"
    raw_path = run_dir / f"raw_{mode}.jsonl"
    pred_df = read_tsv_if_exists(pred_path)
    raw_rows = load_jsonl(raw_path)
    return {
        "judge_summary": aggregate_judge_summary(raw_rows),
        "distribution_summary": evaluate_optional_distribution(pred_df, meta),
    }


def _compare_modes(all_summary: dict) -> dict:
    rows = []
    for mode, summary in all_summary.items():
        dist = summary.get("distribution_summary", {})
        rows.append({
            "mode": mode,
            "mean_w1": dist.get("mean_w1"),
            "top_option_hit_rate": dist.get("top_option_hit_rate"),
            "mean_variance_ratio": dist.get("mean_variance_ratio"),
            "self_correlation_distance": dist.get("self_correlation_distance"),
            "valid_rows": dist.get("valid_rows"),
            "failed_rows": summary.get("judge_summary", {}).get("failed_rows"),
        })
    ranked = [r for r in rows if r["mean_w1"] is not None]
    ranked.sort(key=lambda x: x["mean_w1"])
    return {
        "ranked_by_mean_w1": ranked,
        "has_distribution_metrics": bool(ranked),
        "note": "Lower mean_w1 and self_correlation_distance are better. Higher top_option_hit_rate is better. Variance ratio closer to 1 is better.",
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config = apply_runtime_overrides(config, model=args.model, base_url=args.base_url, api_key=args.api_key, outputs_dir=args.outputs_dir)

    if args.test_run:
        selected_modes = args.modes if args.modes else config.benchmark.prompt_modes
        config = apply_test_overrides(
            config,
            max_questions=args.test_questions,
            max_groups=args.test_groups,
            participants_per_group_question=args.test_participants,
            modes=selected_modes,
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
    logger.info("Starting benchmark run_id=%s config=%s run_dir=%s", run_id, args.config, run_dir)

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
        "script": "run_benchmark.py",
        "config_path": str(args.config),
        "run_dir": str(run_dir),
        "log_path": str(log_path),
        "test_run": bool(args.test_run),
        "mock": bool(args.mock),
        "disable_llm_judge": bool(args.disable_llm_judge or args.test_run),
        "modes": config.benchmark.prompt_modes,
        "questions": config.benchmark.questions,
        "groups": config.benchmark.groups,
        "participants_per_group_question": config.benchmark.participants_per_group_question,
        "full_value_qa_file": str(full_value_path) if full_value_path else None,
        "resume_run_dir": str(args.resume_run_dir) if args.resume_run_dir else None,
        "stop_after_rows": args.stop_after_rows,
    })

    meta = load_question_metadata(config.paths.data_dir / config.paths.question_metadata_file)
    client = build_client(config)
    new_rows_written = 0

    for mode in config.benchmark.prompt_modes:
        logger.info("Running prompt mode=%s", mode)
        pred_path = run_dir / f"predictions_{mode}.tsv"
        raw_path = run_dir / f"raw_{mode}.jsonl"
        summary_path = run_dir / f"summary_{mode}.json"

        existing_pred = read_tsv_if_exists(pred_path)
        processed_keys = set(existing_pred["ROW_KEY"].astype(str).tolist()) if not existing_pred.empty and "ROW_KEY" in existing_pred.columns else set()
        logger.info("Resume status for mode=%s: already_completed=%d total_rows=%d", mode, len(processed_keys), len(subset_df))

        iterator = tqdm(subset_df.to_dict(orient="records"), desc=f"benchmark:{mode}")
        for row in iterator:
            row_key = benchmark_row_key(row, mode)
            if row_key in processed_keys:
                continue
            qid = row["QUESTION_ID"]
            try:
                question_text = get_question_text(meta, qid)
                score_min, score_max = get_question_scale(meta, qid)
                pred_row, raw_row = _run_row_with_recovery(
                    client=client,
                    config=config,
                    meta=meta,
                    mode=mode,
                    row=row,
                    question_text=question_text,
                    score_min=score_min,
                    score_max=score_max,
                )
                append_tsv_row(pred_path, pred_row)
                append_jsonl(raw_path, raw_row)
                processed_keys.add(row_key)
                new_rows_written += 1
            except Exception as e:
                logger.exception(
                    "Benchmark row failed | mode=%s qid=%s participant=%s error=%s",
                    mode,
                    row.get("QUESTION_ID"),
                    row.get("PARTICIPANT_ID"),
                    repr(e),
                )
                error_row = {
                    "status": "error",
                    "ROW_KEY": row_key,
                    "QUESTION_ID": row.get("QUESTION_ID"),
                    "PARTICIPANT_ID": row.get("PARTICIPANT_ID"),
                    "GROUP_KEY": row.get("GROUP_KEY"),
                    "Continent": row.get("Continent"),
                    "Urban_Rural": row.get("Urban_Rural"),
                    "Education": row.get("Education"),
                    "HUMAN_SCORE": row.get("HUMAN_SCORE"),
                    "prompt_mode": mode,
                    "error": repr(e),
                    "recovery_attempted": True,
                    "generation_attempts": getattr(e, "attempts", []),
                }
                append_jsonl(raw_path, error_row)
                if not config.benchmark.continue_on_error:
                    raise
            if args.stop_after_rows is not None and new_rows_written >= args.stop_after_rows:
                logger.error("Stopping early because --stop-after-rows=%s was reached", args.stop_after_rows)
                raise RuntimeError(f"Stopped early after {new_rows_written} new rows by request")

        summary = _summary_from_files(run_dir, mode, meta)
        write_json(summary_path, summary)
        logger.info("Saved outputs for mode=%s pred=%s raw=%s summary=%s", mode, pred_path, raw_path, summary_path)

    all_summary = {mode: _summary_from_files(run_dir, mode, meta) for mode in config.benchmark.prompt_modes}
    all_summary["mode_comparison"] = _compare_modes({k: v for k, v in all_summary.items()})
    write_json(run_dir / "summary_all.json", all_summary)
    logger.info("Benchmark completed successfully run_dir=%s", run_dir)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.getLogger(__name__).exception("Fatal benchmark error: %s", repr(e))
        raise
