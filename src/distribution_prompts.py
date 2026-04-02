from __future__ import annotations

from typing import Any


def build_distribution_prompt(
    mode: str,
    question_text: str,
    continent: str,
    urban_rural: str,
    education: str,
    score_min: int,
    score_max: int,
) -> str:
    allowed = ", ".join(str(i) for i in range(score_min, score_max + 1))
    schema_note = (
        'Return valid JSON only with keys: "option_probs", "most_likely_answer", and "rationale_short". '
        f'"option_probs" must contain every allowed answer from {score_min} to {score_max} exactly once, '
        'as probabilities summing to 1.0. "most_likely_answer" must be one allowed integer.'
    )

    if mode == 'no_demo_distribution':
        return f'''
This is a survey simulation task, not an assistant task.
You are producing an aggregate response distribution for an unspecified generic survey respondent population.
Do not mention being an AI assistant. Do not refuse. Do not add any biography or demographic assumptions beyond the question itself.
Your goal is not to choose one answer, but to estimate a plausible distribution over the allowed survey options.

Question:
{question_text}

Allowed answers: {allowed}

{schema_note}
Keep rationale_short to one short sentence.
'''.strip()

    if mode == 'direct_persona_distribution':
        return f'''
This is a survey simulation task.
Estimate a plausible distribution of answers for respondents with this demographic context.
Use the demographic context as survey conditioning only. Do not invent names, stories, or extra backstory.

Respondent demographic context:
- Region: {continent}
- Settlement type: {urban_rural}
- Education: {education}

Question:
{question_text}

Allowed answers: {allowed}

{schema_note}
Keep rationale_short to one short sentence describing the main driver of the distribution.
'''.strip()

    if mode == 'interview_direct_distribution':
        return f'''
This is a survey interview simulation.
Estimate a plausible distribution of answers for respondents with this demographic context.
Do not answer as an assistant. Do not explain the survey. Use the interview framing only to stay grounded and natural.

Interviewer notes about the respondent group:
- Region: {continent}
- Settlement type: {urban_rural}
- Education: {education}

Survey question:
{question_text}

Allowed answers: {allowed}

{schema_note}
Keep rationale_short to one short sentence describing why the mass is concentrated where it is.
'''.strip()

    raise ValueError(f'Unknown distribution mode: {mode}')


def build_distribution_schema(score_min: int, score_max: int) -> dict[str, Any]:
    prob_props = {str(i): {"type": "number", "minimum": 0.0, "maximum": 1.0} for i in range(score_min, score_max + 1)}
    return {
        "type": "object",
        "properties": {
            "option_probs": {
                "type": "object",
                "properties": prob_props,
                "required": [str(i) for i in range(score_min, score_max + 1)],
                "additionalProperties": False,
            },
            "most_likely_answer": {"type": "integer", "minimum": score_min, "maximum": score_max},
            "rationale_short": {"type": "string"},
        },
        "required": ["option_probs", "most_likely_answer", "rationale_short"],
        "additionalProperties": False,
    }
