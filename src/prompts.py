from __future__ import annotations


def _common_json_format(score_min: int, score_max: int, explanation_max_words: int) -> str:
    return f'''
Return valid JSON only with these exact keys:
{{
  "rationale": "up to {explanation_max_words} words",
  "answer_score": integer between {score_min} and {score_max}
}}
'''.strip()


def _state_json_format(state_dims: list[tuple[str, str]], score_min: int, score_max: int, explanation_max_words: int, *, nested_state: bool) -> str:
    dim_lines = []
    for key, _ in state_dims:
        dim_lines.append(f'    "{key}": "low|medium|high"')
    dim_block = ",\n".join(dim_lines)
    if nested_state:
        return f'''
Return valid JSON only with these exact keys:
{{
  "state": {{
{dim_block}
  }},
  "rationale": "up to {explanation_max_words} words",
  "answer_score": integer between {score_min} and {score_max}
}}
'''.strip()
    return f'''
Return valid JSON only with these exact keys:
{{
{dim_block},
  "rationale": "up to {explanation_max_words} words",
  "answer_score": integer between {score_min} and {score_max}
}}
'''.strip()


def build_prompt(
    mode: str,
    question_text: str,
    continent: str,
    urban_rural: str,
    education: str,
    score_min: int,
    score_max: int,
    explanation_max_words: int,
    frame_variant: str = "default",
    output_variant: str = "standard",
    state_dims: list[tuple[str, str]] | None = None,
) -> str:
    if frame_variant == "minimal":
        frame_prefix = "Answer like a normal survey respondent. Keep it brief and natural."
    else:
        frame_prefix = "Answer like a normal survey participant. Do not sound like an expert. Keep it brief and natural."

    common_format = _common_json_format(score_min, score_max, explanation_max_words)

    if mode == "no_demo":
        return f'''
This is a survey simulation task, not an assistant task.

You are producing the answer of an unspecified generic survey respondent.
{frame_prefix}
Use only the wording of the question itself.
Do not mention being an AI assistant.
Do not refuse, do not give safety disclaimers, and do not explain your limitations.
Do not invent a detailed biography, demographic profile, or extra context beyond the question.
Choose exactly one valid score on the allowed scale.

Question:
{question_text}

{common_format}
'''.strip()

    if mode in {"direct_persona", "direct_persona_minimal"}:
        extra = "Choose exactly one valid score on the allowed scale."
        if mode == "direct_persona_minimal":
            extra = "Treat the demographic attributes as survey conditioning only. Do not add any extra story or personality beyond them. Choose exactly one valid score on the allowed scale."
        return f'''
This is a survey simulation task.

Respondent demographic context:
- Region: {continent}
- Settlement type: {urban_rural}
- Education: {education}

{frame_prefix}
{extra}

Question:
{question_text}

{common_format}
'''.strip()

    if mode in {"interview_direct", "interview_direct_stripped"}:
        extra = "Do not explain the survey itself."
        if mode == "interview_direct_stripped":
            extra = "Do not explain the survey itself. Keep the rationale extremely short and avoid decorative phrasing."
        return f'''
This is a survey interview.

Interviewer notes about the respondent:
- Region: {continent}
- Settlement type: {urban_rural}
- Education: {education}

{frame_prefix}
{extra}

Survey question:
{question_text}

{common_format}
'''.strip()

    if mode in {"interview_state_guided", "interview_state_guided_cot", "interview_state_guided_flat_topic_specific"}:
        if not state_dims:
            raise ValueError("state_dims must be provided for state-guided prompting")
        dim_desc = "\n".join(f"- {key}: {desc}" for key, desc in state_dims)
        nested_state = output_variant != "flat_state"
        output_format = _state_json_format(state_dims, score_min, score_max, explanation_max_words, nested_state=nested_state)

        reasoning_instruction = "Then answer the survey question consistently with that state."
        if mode == "interview_state_guided_cot":
            reasoning_instruction = (
                "First decide the respondent's likely stance from the state, then answer consistently with that stance. "
                "Keep the rationale short and non-technical."
            )
        elif mode == "interview_state_guided_flat_topic_specific":
            reasoning_instruction = "Use the state as a compact latent profile only. Do not add extra internal reasoning or biography."

        return f'''
This is a survey interview.

Respondent demographic context:
- Region: {continent}
- Settlement type: {urban_rural}
- Education: {education}

Before answering, infer a compact respondent state from this demographic context.
Use exactly these dimensions:
{dim_desc}

{frame_prefix}
{reasoning_instruction}
Use short, simple JSON.

Survey question:
{question_text}

{output_format}
'''.strip()

    raise ValueError(f"Unknown mode: {mode}")
