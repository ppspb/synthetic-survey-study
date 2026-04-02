from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from .config import AppConfig
from .eval_utils import (
    aggregate_judge_summary,
    clamp_score,
    evaluate_optional_distribution,
    safe_int,
)
from .judge import run_optional_llm_judge, run_rule_judge
from .lm_client import GenerationFailure, create_client
from .metadata import get_question_scale, get_question_text, load_question_metadata
from .prompts import build_prompt
from .question_state import build_response_format, select_state_dimensions

logger = logging.getLogger(__name__)


def benchmark_row_key(row: dict[str, Any], mode: str) -> str:
    return f"{row['QUESTION_ID']}::{row['PARTICIPANT_ID']}::{mode}"


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


def _build_prompt_and_schema(
    config: AppConfig,
    meta: dict[str, dict],
    mode: str,
    row: dict[str, Any],
    question_text: str,
    score_min: int,
    score_max: int,
    frame_variant: str = "default",
    output_variant: str = "standard",
) -> tuple[str, dict[str, Any], list[tuple[str, str]]]:
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
    schema_hint = build_response_format(
        state_dims,
        score_min,
        score_max,
        nested_state=(output_variant != "flat_state"),
    )
    return prompt, schema_hint, state_dims or []


def _build_single_response(
    client,
    config: AppConfig,
    meta: dict[str, dict],
    mode: str,
    row: dict[str, Any],
    question_text: str,
    score_min: int,
    score_max: int,
    frame_variant: str = "default",
    output_variant: str = "standard",
    temperature: float | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    prompt, schema_hint, state_dims = _build_prompt_and_schema(
        config=config,
        meta=meta,
        mode=mode,
        row=row,
        question_text=question_text,
        score_min=score_min,
        score_max=score_max,
        frame_variant=frame_variant,
        output_variant=output_variant,
    )

    trace = client.generate_json_trace(
        prompt=prompt,
        temperature=config.api.temperature if temperature is None else temperature,
        max_retries=config.api.max_retries,
        schema_hint=schema_hint,
    )
    state_keys = [key for key, _ in state_dims]
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
        if llm_judge and "judge_error" in llm_judge and config.judge.fail_on_judge_error:
            raise RuntimeError(f"Judge failed: {llm_judge['judge_error']}")

    row_key = benchmark_row_key(row, mode)
    pred_row = {
        "ROW_KEY": row_key,
        "QUESTION_ID": row["QUESTION_ID"],
        "PARTICIPANT_ID": row["PARTICIPANT_ID"],
        "GROUP_KEY": row["GROUP_KEY"],
        "MODEL_SCORE": model_score,
        "prompt_mode": mode,
    }
    if "HUMAN_SCORE" in row and pd.notna(row.get("HUMAN_SCORE")):
        pred_row["HUMAN_SCORE"] = int(row["HUMAN_SCORE"])

    raw_row = {
        "status": "ok",
        "ROW_KEY": row_key,
        "QUESTION_ID": row["QUESTION_ID"],
        "PARTICIPANT_ID": row["PARTICIPANT_ID"],
        "GROUP_KEY": row["GROUP_KEY"],
        "Continent": row["Continent"],
        "Urban_Rural": row["Urban_Rural"],
        "Education": row["Education"],
        "HUMAN_SCORE": row.get("HUMAN_SCORE"),
        "prompt_mode": mode,
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


def _build_error_row(row: dict[str, Any], mode: str, error: Exception, recovery_attempted: bool, attempt_logs: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "status": "error",
        "ROW_KEY": benchmark_row_key(row, mode),
        "QUESTION_ID": row.get("QUESTION_ID"),
        "PARTICIPANT_ID": row.get("PARTICIPANT_ID"),
        "GROUP_KEY": row.get("GROUP_KEY"),
        "Continent": row.get("Continent"),
        "Urban_Rural": row.get("Urban_Rural"),
        "Education": row.get("Education"),
        "HUMAN_SCORE": row.get("HUMAN_SCORE"),
        "prompt_mode": mode,
        "error": repr(error),
        "recovery_attempted": recovery_attempted,
        "generation_attempts": attempt_logs or getattr(error, "attempts", []),
    }


def _run_row_with_recovery(
    client,
    config: AppConfig,
    meta: dict[str, dict],
    mode: str,
    row: dict[str, Any],
    question_text: str,
    score_min: int,
    score_max: int,
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
            frame_variant="default",
            output_variant="standard",
            temperature=config.api.temperature,
        )
        raw_row["recovery_attempted"] = False
        return pred_row, raw_row
    except Exception as primary_error:
        primary_attempts = getattr(primary_error, "attempts", [])
        logger.warning(
            "Primary generation failed; attempting recovery | mode=%s qid=%s participant=%s error=%s",
            mode,
            row.get("QUESTION_ID"),
            row.get("PARTICIPANT_ID"),
            repr(primary_error),
        )
        recovery_frame = "minimal"
        recovery_output_variant = "flat_state" if mode.startswith("interview_state_guided") else "standard"
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
                frame_variant=recovery_frame,
                output_variant=recovery_output_variant,
                temperature=config.api.temperature,
            )
            raw_row["recovery_attempted"] = True
            raw_row["recovery_reason"] = repr(primary_error)
            raw_row["primary_attempts"] = primary_attempts
            return pred_row, raw_row
        except Exception as recovery_error:
            combined_attempts = []
            combined_attempts.extend(primary_attempts)
            combined_attempts.extend(getattr(recovery_error, "attempts", []))
            raise GenerationFailure(str(recovery_error), attempts=combined_attempts) from recovery_error


def summarize_benchmark_with_meta(pred_df: pd.DataFrame, raw_rows: list[dict[str, Any]], meta: dict[str, dict]) -> dict[str, Any]:
    return {
        "judge_summary": aggregate_judge_summary(raw_rows),
        "distribution_summary": evaluate_optional_distribution(pred_df, meta),
    }


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


def get_question_bundle(config: AppConfig, qid: str) -> tuple[str, int, int]:
    meta = load_question_metadata(config.paths.data_dir / config.paths.question_metadata_file)
    qtext = get_question_text(meta, qid)
    smin, smax = get_question_scale(meta, qid)
    return qtext, smin, smax
