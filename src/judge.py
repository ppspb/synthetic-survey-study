from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)
EXPECTED_STATE_VALUES = {"low", "medium", "high"}


def _word_count(text: str) -> int:
    return len(str(text).strip().split()) if text else 0


def run_rule_judge(
    parsed: dict[str, Any],
    mode: str,
    score_min: int,
    score_max: int,
    explanation_max_words: int,
    expected_state_keys: set[str] | None = None,
) -> dict[str, Any]:
    issues: list[str] = []
    has_rationale = isinstance(parsed.get("rationale"), str) and bool(parsed.get("rationale").strip())
    if not has_rationale:
        issues.append("missing_or_empty_rationale")

    score = parsed.get("answer_score")
    score_valid = isinstance(score, int) and score_min <= score <= score_max
    if not score_valid:
        issues.append("invalid_or_out_of_range_score")

    rationale_words = _word_count(parsed.get("rationale", "")) if has_rationale else 0
    rationale_within_limit = rationale_words <= explanation_max_words
    if not rationale_within_limit:
        issues.append("rationale_too_long")

    state_complete = True
    state_valid = True
    if mode.startswith("interview_state_guided"):
        expected_state_keys = expected_state_keys or set()
        state = parsed.get("state")
        if not isinstance(state, dict):
            state_complete = False
            state_valid = False
            issues.append("missing_state")
        else:
            if expected_state_keys and set(state.keys()) != expected_state_keys:
                state_complete = False
                issues.append("state_keys_mismatch")
            invalid_values = [k for k, v in state.items() if v not in EXPECTED_STATE_VALUES]
            if invalid_values:
                state_valid = False
                issues.append(f"invalid_state_values:{','.join(invalid_values)}")

    checks = {
        "has_rationale": has_rationale,
        "score_valid": score_valid,
        "rationale_within_limit": rationale_within_limit,
        "state_complete": state_complete,
        "state_valid": state_valid,
    }
    passed = all(checks.values())
    rule_score = sum(bool(v) for v in checks.values()) / len(checks)
    return {
        "passed": passed,
        "rule_score": rule_score,
        "checks": checks,
        "issues": issues,
        "rationale_word_count": rationale_words,
    }


def run_optional_llm_judge(
    client,
    enabled: bool,
    model: str | None,
    max_retries: int,
    mode: str,
    question_text: str,
    parsed: dict[str, Any],
) -> dict[str, Any] | None:
    if not enabled:
        return None

    prompt = f"""
You are evaluating a synthetic survey response.

Mode: {mode}
Question: {question_text}
Parsed response JSON: {parsed}

Return JSON only with these exact keys:
{{
  "consistency_score": integer between 1 and 5,
  "naturalness_score": integer between 1 and 5,
  "brief_comment": "one short sentence"
}}
""".strip()
    try:
        judged = client.generate_json(prompt=prompt, temperature=0.0, max_retries=max_retries, model=model)
        return judged
    except Exception as e:
        logger.exception("LLM judge failed: %s", repr(e))
        return {"judge_error": repr(e)}
