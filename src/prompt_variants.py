from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)


def paraphrase_question_text(question_text: str) -> str:
    substitutions = [
        (r"\bhow important is\b", "how important would you say"),
        (r"\bhow interested are you\b", "how interested would you say you are"),
        (r"\bhow often do\b", "how frequently do"),
        (r"\bgenerally speaking, would you say\b", "in general, would you say"),
        (r"\bhow much do you agree or disagree\b", "to what extent do you agree or disagree"),
        (r"\bhow would you rate\b", "how would you evaluate"),
        (r"\bhow much confidence do you have\b", "how much confidence would you say you have"),
    ]
    out = question_text
    changed = False
    for pattern, repl in substitutions:
        new_out = re.sub(pattern, repl, out, flags=re.IGNORECASE)
        if new_out != out:
            changed = True
            out = new_out
    if changed:
        return out
    return f"Please answer the following survey item carefully: {question_text}"


def reverse_anchor_order(question_text: str) -> str:
    pattern = re.compile(
        r"^(On a scale of\s+([^,]+),\s*)(\d+)\s+meaning\s+'([^']+)'\s+and\s+(\d+)\s+meaning\s+'([^']+)'(,\s*.*)$",
        flags=re.IGNORECASE,
    )
    m = pattern.match(question_text.strip())
    if not m:
        logger.warning("reverse_anchor_order: could not parse question text, using original")
        return question_text

    prefix, scale_text, left_num, left_text, right_num, right_text, suffix = m.groups()
    return f"On a scale of {scale_text}, {right_num} meaning '{right_text}' and {left_num} meaning '{left_text}'{suffix}"


def anchor_softened(question_text: str) -> str:
    out = question_text
    replacements = [
        ("Very important", "Extremely important"),
        ("Very happy", "Extremely happy"),
        ("Very good", "Quite good"),
        ("Very bad", "Quite bad"),
        ("Completely agree", "Strongly agree"),
        ("Completely disagree", "Strongly disagree"),
        ("Trust completely", "Trust very strongly"),
    ]
    for src, dst in replacements:
        out = out.replace(src, dst)
    return out


def apply_question_perturbation(question_text: str, perturbation: str) -> str:
    if perturbation == "original":
        return question_text
    if perturbation == "question_paraphrase":
        return paraphrase_question_text(question_text)
    if perturbation == "reverse_anchor_order":
        return reverse_anchor_order(question_text)
    if perturbation == "anchor_softened":
        return anchor_softened(question_text)
    return question_text
