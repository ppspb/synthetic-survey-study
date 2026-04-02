from __future__ import annotations

import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def load_question_metadata(path: Path) -> dict[str, dict]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError("question_metadata.json must be a dict")
    logger.info("Loaded question metadata for %d questions from %s", len(raw), path)
    return raw


def get_question_text(meta: dict[str, dict], qid: str) -> str:
    q = meta.get(qid)
    if not q:
        raise KeyError(f"Question metadata not found for {qid}")
    return str(q["question"])


def get_question_scale(meta: dict[str, dict], qid: str) -> tuple[int, int]:
    q = meta.get(qid)
    if not q:
        raise KeyError(f"Question metadata not found for {qid}")
    min_v = q.get("answer_scale_min")
    max_v = q.get("answer_scale_max")
    if min_v == "" or max_v == "":
        raise ValueError(f"Question {qid} does not define an ordinal answer scale")
    return int(min_v), int(max_v)
