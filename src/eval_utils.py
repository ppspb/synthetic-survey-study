from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

logger = logging.getLogger(__name__)


def safe_int(value: Any, fallback: int) -> int:
    try:
        return int(value)
    except Exception:
        return fallback


def clamp_score(value: int, score_min: int, score_max: int) -> int:
    return max(score_min, min(score_max, value))


def normalize_scores(scores: list[int], score_min: int, score_max: int) -> list[float]:
    if score_max == score_min:
        return [0.0 for _ in scores]
    arr = np.array(scores, dtype=float)
    return ((arr - score_min) / (score_max - score_min)).tolist()


def _mode_value(values: list[int]) -> int | None:
    if not values:
        return None
    counts = pd.Series(values).value_counts().sort_values(ascending=False)
    max_count = counts.iloc[0]
    tied = sorted([int(idx) for idx, count in counts.items() if count == max_count])
    return tied[0]


def _variance_ratio(human: list[int], model: list[int]) -> float | None:
    if len(human) < 2 or len(model) < 2:
        return None
    hvar = float(np.var(human))
    mvar = float(np.var(model))
    if hvar == 0:
        return None
    return mvar / hvar


def _self_correlation_distance(valid_df: pd.DataFrame) -> float | None:
    if valid_df.empty:
        return None
    human_pivot = valid_df.pivot_table(index="PARTICIPANT_ID", columns="QUESTION_ID", values="HUMAN_SCORE", aggfunc="mean")
    model_pivot = valid_df.pivot_table(index="PARTICIPANT_ID", columns="QUESTION_ID", values="MODEL_SCORE", aggfunc="mean")
    common_cols = [c for c in human_pivot.columns if c in model_pivot.columns]
    if len(common_cols) < 2:
        return None
    hcorr = human_pivot[common_cols].corr(min_periods=2)
    mcorr = model_pivot[common_cols].corr(min_periods=2)
    aligned = (hcorr - mcorr).abs().stack().dropna()
    if aligned.empty:
        return None
    return float(aligned.mean())


def aggregate_judge_summary(raw_rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(raw_rows)
    successful = [r for r in raw_rows if r.get("status") == "ok"]
    failed = [r for r in raw_rows if r.get("status") != "ok"]
    passed = [r for r in successful if r.get("rule_judge", {}).get("passed")]
    rule_scores = [r.get("rule_judge", {}).get("rule_score") for r in successful if r.get("rule_judge")]
    rationale_lengths = [r.get("rule_judge", {}).get("rationale_word_count") for r in successful if r.get("rule_judge")]
    llm_consistency = [
        r.get("llm_judge", {}).get("consistency_score")
        for r in successful
        if isinstance(r.get("llm_judge"), dict) and isinstance(r.get("llm_judge", {}).get("consistency_score"), (int, float))
    ]
    llm_naturalness = [
        r.get("llm_judge", {}).get("naturalness_score")
        for r in successful
        if isinstance(r.get("llm_judge"), dict) and isinstance(r.get("llm_judge", {}).get("naturalness_score"), (int, float))
    ]
    llm_judge_errors = [r for r in successful if isinstance(r.get("llm_judge"), dict) and "judge_error" in r.get("llm_judge", {})]
    repaired = [r for r in successful if r.get("used_repair")]
    structured = [r for r in successful if r.get("used_structured_output")]
    recovery_successes = [r for r in successful if r.get("recovery_attempted")]
    return {
        "total_rows": total,
        "successful_rows": len(successful),
        "failed_rows": len(failed),
        "rule_pass_rate": (len(passed) / len(successful)) if successful else None,
        "mean_rule_score": float(np.mean(rule_scores)) if rule_scores else None,
        "mean_rationale_word_count": float(np.mean(rationale_lengths)) if rationale_lengths else None,
        "llm_judge_rows": len(llm_consistency),
        "mean_llm_consistency_score": float(np.mean(llm_consistency)) if llm_consistency else None,
        "mean_llm_naturalness_score": float(np.mean(llm_naturalness)) if llm_naturalness else None,
        "llm_judge_error_rows": len(llm_judge_errors),
        "repair_success_rows": len(repaired),
        "structured_output_success_rows": len(structured),
        "recovery_success_rows": len(recovery_successes),
    }


def evaluate_optional_distribution(pred_df: pd.DataFrame, question_meta: dict[str, dict]) -> dict[str, Any]:
    if pred_df.empty:
        return {"message": "Prediction dataframe is empty; skipping distribution metrics."}

    if "HUMAN_SCORE" not in pred_df.columns:
        return {
            "message": "No HUMAN_SCORE available; skipping distribution metrics.",
            "hint": "samples.tsv from WVB probe contains participant IDs and demographics but not the true answer column. Provide paths.full_value_qa_file to attach ground truth from the full dataset, or use a custom subset with HUMAN_SCORE already included.",
            "available_columns": list(pred_df.columns),
        }

    valid_df = pred_df.dropna(subset=["HUMAN_SCORE", "MODEL_SCORE"]).copy()
    if valid_df.empty:
        return {
            "message": "HUMAN_SCORE column exists but no valid rows remain after dropping NaNs.",
            "available_columns": list(pred_df.columns),
        }

    out: dict[str, Any] = {
        "mean_w1": None,
        "per_question_w1": {},
        "per_group_w1": {},
        "valid_rows": int(len(valid_df)),
        "top_option_hit_rate": None,
        "mean_variance_ratio": None,
        "self_correlation_distance": _self_correlation_distance(valid_df),
    }
    question_scores = []
    variance_ratios = []
    top_option_hits = []

    for qid, qdf in valid_df.groupby("QUESTION_ID"):
        score_min = int(question_meta[qid]["answer_scale_min"])
        score_max = int(question_meta[qid]["answer_scale_max"])
        gt = normalize_scores(qdf["HUMAN_SCORE"].astype(int).tolist(), score_min, score_max)
        pred = normalize_scores(qdf["MODEL_SCORE"].astype(int).tolist(), score_min, score_max)
        w1 = float(wasserstein_distance(gt, pred))
        out["per_question_w1"][qid] = w1
        question_scores.append(w1)

    for group_key, gdf in valid_df.groupby("GROUP_KEY"):
        vals = []
        for qid, qdf in gdf.groupby("QUESTION_ID"):
            score_min = int(question_meta[qid]["answer_scale_min"])
            score_max = int(question_meta[qid]["answer_scale_max"])
            gt_int = qdf["HUMAN_SCORE"].astype(int).tolist()
            pred_int = qdf["MODEL_SCORE"].astype(int).tolist()
            gt = normalize_scores(gt_int, score_min, score_max)
            pred = normalize_scores(pred_int, score_min, score_max)
            vals.append(float(wasserstein_distance(gt, pred)))
            ratio = _variance_ratio(gt_int, pred_int)
            if ratio is not None:
                variance_ratios.append(ratio)
            hmode = _mode_value(gt_int)
            mmode = _mode_value(pred_int)
            if hmode is not None and mmode is not None:
                top_option_hits.append(float(hmode == mmode))
        out["per_group_w1"][group_key] = float(np.mean(vals)) if vals else None

    out["mean_w1"] = float(np.mean(question_scores)) if question_scores else None
    out["top_option_hit_rate"] = float(np.mean(top_option_hits)) if top_option_hits else None
    out["mean_variance_ratio"] = float(np.mean(variance_ratios)) if variance_ratios else None
    logger.info(
        "Distribution evaluation complete: mean_w1=%s top_option_hit_rate=%s variance_ratio=%s self_corr_dist=%s valid_rows=%d",
        out["mean_w1"], out["top_option_hit_rate"], out["mean_variance_ratio"], out["self_correlation_distance"], len(valid_df),
    )
    return out
