from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from .config import AppConfig
from .eval_utils import aggregate_judge_summary, clamp_score, safe_int
from .judge import run_optional_llm_judge, run_rule_judge
from .lm_client import create_client
from .metadata import get_question_scale, get_question_text, load_question_metadata
from .prompt_variants import apply_question_perturbation
from .prompts import build_prompt
from .question_state import build_response_format, select_state_dimensions

logger = logging.getLogger(__name__)


def robustness_row_key(row: dict[str, Any], mode: str) -> str:
    return f"{row['QUESTION_ID']}::{row['PARTICIPANT_ID']}::{mode}"


def robustness_execution_key(row: dict[str, Any], mode: str, perturbation: str) -> str:
    return f"{row['QUESTION_ID']}::{row['PARTICIPANT_ID']}::{mode}::{perturbation}"


def _frame_variant_for_perturbation(perturbation: str) -> str:
    if perturbation == "frame_minimal":
        return "minimal"
    return "default"


def _normalize_state_shape(parsed: dict[str, Any], mode: str, state_keys: list[str]) -> dict[str, Any]:
    if not mode.startswith("interview_state_guided"):
        return parsed
    if isinstance(parsed.get("state"), dict):
        return parsed
    flat_present = all(k in parsed for k in state_keys)
    if flat_present:
        state = {k: parsed.get(k) for k in state_keys}
        normalized = dict(parsed)
        for k in state_keys:
            normalized.pop(k, None)
        normalized["state"] = state
        return normalized
    return parsed


def _build_single_response(
    client,
    config: AppConfig,
    meta: dict[str, dict],
    mode: str,
    row: dict[str, Any],
    question_text: str,
    score_min: int,
    score_max: int,
    perturbation: str,
    frame_variant: str = "default",
    output_variant: str = "standard",
) -> tuple[dict[str, Any], dict[str, Any]]:
    state_dims = select_state_dimensions(meta, row["QUESTION_ID"]) if mode.startswith("interview_state_guided") else None
    prompt = build_prompt(
        mode=mode,
        question_text=question_text,
        continent=str(row["Continent"]),
        urban_rural=str(row["Urban_Rural"]),
        education=str(row["Education"]),
        score_min=score_min,
        score_max=score_max,
        explanation_max_words=config.benchmark.explanation_max_words,
        frame_variant=frame_variant,
        output_variant=output_variant,
        state_dims=state_dims,
    )
    schema_hint = build_response_format(state_dims, score_min, score_max, nested_state=(output_variant != "flat_state"))
    trace = client.generate_json_trace(
        prompt=prompt,
        temperature=config.api.temperature,
        max_retries=config.api.max_retries,
        schema_hint=schema_hint,
    )
    state_keys = [key for key, _ in (state_dims or [])]
    parsed = _normalize_state_shape(trace.parsed, mode=mode, state_keys=state_keys)

    raw_score = safe_int(parsed.get("answer_score"), fallback=score_min)
    model_score = clamp_score(raw_score, score_min, score_max)
    rule_judge = run_rule_judge(
        parsed=parsed,
        mode=mode,
        score_min=score_min,
        score_max=score_max,
        explanation_max_words=config.benchmark.explanation_max_words,
        expected_state_keys=set(state_keys) if state_keys else None,
    )
    llm_judge = None
    if config.judge.enabled and config.judge.use_llm:
        llm_judge = run_optional_llm_judge(
            client=client,
            enabled=True,
            model=config.judge.llm_model,
            max_retries=config.judge.max_retries,
            mode=mode,
            question_text=question_text,
            parsed=parsed,
        )

    row_key = robustness_row_key(row, mode)
    execution_key = robustness_execution_key(row, mode, perturbation)
    pred_row = {
        "ROW_KEY": row_key,
        "EXECUTION_KEY": execution_key,
        "QUESTION_ID": row["QUESTION_ID"],
        "PARTICIPANT_ID": row["PARTICIPANT_ID"],
        "GROUP_KEY": row["GROUP_KEY"],
        "MODEL_SCORE": model_score,
        "prompt_mode": mode,
        "perturbation": perturbation,
    }
    if "HUMAN_SCORE" in row and pd.notna(row.get("HUMAN_SCORE")):
        pred_row["HUMAN_SCORE"] = int(row["HUMAN_SCORE"])

    raw_row = {
        "status": "ok",
        "ROW_KEY": row_key,
        "EXECUTION_KEY": execution_key,
        "QUESTION_ID": row["QUESTION_ID"],
        "PARTICIPANT_ID": row["PARTICIPANT_ID"],
        "GROUP_KEY": row["GROUP_KEY"],
        "HUMAN_SCORE": row.get("HUMAN_SCORE"),
        "prompt_mode": mode,
        "perturbation": perturbation,
        "frame_variant": frame_variant,
        "output_variant": output_variant,
        "state_dimensions": state_dims,
        "prompt": prompt,
        "raw_text": trace.raw_text,
        "parser_used": trace.parser_used,
        "generation_attempts": trace.attempts,
        "used_repair": trace.used_repair,
        "used_structured_output": trace.used_structured_output,
        "parsed": parsed,
        "rule_judge": rule_judge,
        "llm_judge": llm_judge,
    }
    return pred_row, raw_row


def _run_row_with_recovery(
    client,
    config: AppConfig,
    meta: dict[str, dict],
    mode: str,
    row: dict[str, Any],
    question_text: str,
    score_min: int,
    score_max: int,
    perturbation: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        pred_row, raw_row = _build_single_response(
            client=client,
            config=config,
            meta=meta,
            mode=mode,
            row=row,
            question_text=question_text,
            score_min=score_min,
            score_max=score_max,
            perturbation=perturbation,
            frame_variant=_frame_variant_for_perturbation(perturbation),
            output_variant="standard",
        )
        raw_row["recovery_attempted"] = False
        return pred_row, raw_row
    except Exception as primary_error:
        logger.warning(
            "Robustness primary generation failed; attempting recovery | mode=%s perturbation=%s qid=%s participant=%s error=%s",
            mode, perturbation, row.get("QUESTION_ID"), row.get("PARTICIPANT_ID"), repr(primary_error)
        )
        recovery_output_variant = "flat_state" if mode.startswith("interview_state_guided") else "standard"
        pred_row, raw_row = _build_single_response(
            client=client,
            config=config,
            meta=meta,
            mode=mode,
            row=row,
            question_text=question_text,
            score_min=score_min,
            score_max=score_max,
            perturbation=perturbation,
            frame_variant="minimal",
            output_variant=recovery_output_variant,
        )
        raw_row["recovery_attempted"] = True
        raw_row["recovery_reason"] = repr(primary_error)
        raw_row["primary_attempts"] = getattr(primary_error, "attempts", [])
        return pred_row, raw_row


def build_client(config: AppConfig):
    return create_client(
        base_url=config.api.base_url,
        api_key=config.api.api_key,
        model=config.api.model,
        timeout_seconds=config.api.timeout_seconds,
        api_mode=config.api.api_mode,
        use_structured_output=config.api.use_structured_output,
        structured_output_strict=config.api.structured_output_strict,
        max_completion_tokens=config.api.max_completion_tokens,
        request_delay_seconds=config.api.request_delay_seconds,
    )


def load_meta(config: AppConfig) -> dict[str, dict]:
    return load_question_metadata(config.paths.data_dir / config.paths.question_metadata_file)


def summarize_robustness(pred_df: pd.DataFrame, raw_rows: list[dict[str, Any]], compare_to: str = "original") -> dict[str, Any]:
    summary: dict[str, Any] = {"judge_summary": aggregate_judge_summary(raw_rows), "per_mode": {}}
    if pred_df.empty:
        return summary

    base_df = pred_df[pred_df["perturbation"] == compare_to].copy()
    if base_df.empty:
        summary["warning"] = f"No base perturbation={compare_to} found; cannot compute deltas."
        return summary

    for mode, mode_df in pred_df.groupby("prompt_mode"):
        mode_summary: dict[str, Any] = {}
        base_mode = base_df[base_df["prompt_mode"] == mode][["ROW_KEY", "MODEL_SCORE"]].rename(columns={"MODEL_SCORE": "BASE_SCORE"})
        for perturbation, pdf in mode_df.groupby("perturbation"):
            merged = pdf.merge(base_mode, on="ROW_KEY", how="left")
            mean_abs_delta = None
            flip_rate = None
            coverage = None
            if perturbation != compare_to and not merged.empty:
                valid = merged.dropna(subset=["BASE_SCORE"])
                coverage = float(len(valid) / len(pdf)) if len(pdf) else None
                if not valid.empty:
                    abs_delta = (valid["MODEL_SCORE"] - valid["BASE_SCORE"]).abs()
                    mean_abs_delta = float(abs_delta.mean())
                    flip_rate = float((abs_delta > 0).mean())
            else:
                coverage = 1.0
                mean_abs_delta = 0.0
                flip_rate = 0.0

            mode_summary[perturbation] = {
                "rows": int(len(pdf)),
                "coverage_vs_base": coverage,
                "mean_abs_score_delta": mean_abs_delta,
                "score_flip_rate": flip_rate,
            }
        summary["per_mode"][mode] = mode_summary
    return summary
