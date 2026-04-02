from __future__ import annotations

from typing import Any

GENERIC_STATE = [
    ("tradition_vs_change", "orientation toward preserving tradition versus embracing social change"),
    ("institutional_trust", "baseline trust in institutions and formal organizations"),
    ("autonomy_vs_conformity", "preference for personal autonomy versus conformity to norms"),
    ("uncertainty_avoidance", "comfort with ambiguity, uncertainty, and unfamiliar situations"),
]

SOCIAL_VALUES_STATE = [
    ("tradition_vs_change", "orientation toward preserving tradition versus embracing social change"),
    ("religiosity", "importance of religion and sacred values in everyday life"),
    ("authority_orientation", "preference for authority, obedience, and respect for hierarchy"),
    ("collectivism_vs_individualism", "preference for collective duty versus individual self-direction"),
]

SOCIAL_CAPITAL_STATE = [
    ("institutional_trust", "baseline trust in institutions and formal organizations"),
    ("interpersonal_trust", "baseline trust in other people and unfamiliar others"),
    ("social_cynicism", "tendency to expect corruption, bad faith, or hidden motives"),
    ("order_vs_fairness", "priority given to order, stability, and control versus fairness and openness"),
]

SCI_TECH_CHANGE_STATE = [
    ("tech_optimism", "belief that science and technology improve life and society"),
    ("change_acceptance", "willingness to accept social and technological change"),
    ("uncertainty_avoidance", "comfort with ambiguity, uncertainty, and unfamiliar situations"),
    ("openness_to_outgroups", "openness toward unfamiliar groups, new ideas, and external influences"),
]

WELLBEING_STATE = [
    ("life_control", "sense of agency and control over life circumstances"),
    ("security_sensitivity", "sensitivity to insecurity, threat, and instability"),
    ("status_quo_preference", "preference for familiar arrangements over disruptive change"),
    ("future_optimism", "expectation that the near future can improve or remain manageable"),
]


def select_state_dimensions(meta: dict[str, dict], qid: str) -> list[tuple[str, str]]:
    entry = meta.get(qid, {})
    category = str(entry.get("category", "")).lower()
    question = str(entry.get("question", "")).lower()

    if "social capital" in category or "trust" in category or "corruption" in question or "confidence" in question:
        return SOCIAL_CAPITAL_STATE
    if "science" in question or "technology" in question or "immigration" in question or "immigrant" in question:
        return SCI_TECH_CHANGE_STATE
    if "happiness" in category or "wellbeing" in category or "life satisfaction" in question or "health" in question:
        return WELLBEING_STATE
    if "social values" in category or "religion" in question or "family" in question or "politics" in question:
        return SOCIAL_VALUES_STATE
    return GENERIC_STATE


def build_response_format(state_dims: list[tuple[str, str]] | None, score_min: int, score_max: int, *, nested_state: bool) -> dict[str, Any]:
    def enum_field() -> dict[str, Any]:
        return {"type": "string", "enum": ["low", "medium", "high"]}

    score_schema: dict[str, Any] = {
        "type": "integer",
        "minimum": int(score_min),
        "maximum": int(score_max),
    }

    if state_dims:
        state_props = {key: enum_field() for key, _ in state_dims}
        state_required = [key for key, _ in state_dims]
        if nested_state:
            schema = {
                "type": "object",
                "properties": {
                    "state": {
                        "type": "object",
                        "properties": state_props,
                        "required": state_required,
                        "additionalProperties": False,
                    },
                    "rationale": {"type": "string"},
                    "answer_score": score_schema,
                },
                "required": ["state", "rationale", "answer_score"],
                "additionalProperties": False,
            }
        else:
            flat_props = dict(state_props)
            flat_props["rationale"] = {"type": "string"}
            flat_props["answer_score"] = score_schema
            schema = {
                "type": "object",
                "properties": flat_props,
                "required": state_required + ["rationale", "answer_score"],
                "additionalProperties": False,
            }
        name = "state_guided_response"
    else:
        schema = {
            "type": "object",
            "properties": {
                "rationale": {"type": "string"},
                "answer_score": score_schema,
            },
            "required": ["rationale", "answer_score"],
            "additionalProperties": False,
        }
        name = "survey_response"

    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "strict": True,
            "schema": schema,
        },
    }
