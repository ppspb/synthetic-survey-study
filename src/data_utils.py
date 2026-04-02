from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable
import pandas as pd

logger = logging.getLogger(__name__)

POSSIBLE_PARTICIPANT_COLS = ["D_INTERVIEW", "participant_id", "PARTICIPANT_ID"]
POSSIBLE_QUESTION_COLS = ["Question", "QUESTION_ID", "question_id"]
POSSIBLE_SCORE_COLS = ["Answer", "SCORE", "score", "Value", "HUMAN_SCORE"]
POSSIBLE_CONTINENT_COLS = ["Continent", "continent"]
POSSIBLE_URBAN_COLS = ["Urban / Rural", "Urban/Rural", "urban_rural", "Urban_Rural"]
POSSIBLE_EDU_COLS = ["Education", "education"]


def _pick_col(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def resolve_data_path(path_value: str | Path | None, data_dir: Path, project_root: Path | None = None) -> Path | None:
    if path_value is None:
        return None
    p = Path(path_value)
    candidates = []
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.extend([p, data_dir / p])
        if project_root is not None:
            candidates.append(project_root / p)
    for c in candidates:
        if c.exists():
            logger.info("Resolved data path %s -> %s", path_value, c)
            return c
    logger.warning("Could not resolve data path %s. Tried: %s", path_value, [str(c) for c in candidates])
    return candidates[-1] if candidates else p


def load_samples(path: Path) -> pd.DataFrame:
    logger.info("Loading samples from %s", path)
    df = pd.read_csv(path, sep="	")
    logger.info("Loaded samples: shape=%s columns=%s", df.shape, list(df.columns))
    return df


def canonicalize_samples(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    participant_col = _pick_col(df, POSSIBLE_PARTICIPANT_COLS)
    question_col = _pick_col(df, POSSIBLE_QUESTION_COLS)
    score_col = _pick_col(df, POSSIBLE_SCORE_COLS)
    continent_col = _pick_col(df, POSSIBLE_CONTINENT_COLS)
    urban_col = _pick_col(df, POSSIBLE_URBAN_COLS)
    edu_col = _pick_col(df, POSSIBLE_EDU_COLS)

    rename_map = {}
    if participant_col:
        rename_map[participant_col] = "PARTICIPANT_ID"
    if question_col:
        rename_map[question_col] = "QUESTION_ID"
    if score_col:
        rename_map[score_col] = "HUMAN_SCORE"
    if continent_col:
        rename_map[continent_col] = "Continent"
    if urban_col:
        rename_map[urban_col] = "Urban_Rural"
    if edu_col:
        rename_map[edu_col] = "Education"

    out = out.rename(columns=rename_map)

    required = ["QUESTION_ID", "PARTICIPANT_ID", "Continent", "Urban_Rural", "Education"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise KeyError(f"Missing required columns after canonicalization: {missing}")

    out["QUESTION_ID"] = out["QUESTION_ID"].astype(str)
    out["PARTICIPANT_ID"] = out["PARTICIPANT_ID"].astype(str)

    if "HUMAN_SCORE" in out.columns:
        out["HUMAN_SCORE"] = pd.to_numeric(out["HUMAN_SCORE"], errors="coerce")
        non_null = int(out["HUMAN_SCORE"].notna().sum())
        logger.info("Detected HUMAN_SCORE in samples with %d non-null rows", non_null)
    else:
        logger.warning(
            "No HUMAN_SCORE-like column found in samples. Local distribution metrics will be unavailable unless human data is provided."
        )

    out["GROUP_KEY"] = (
        out["Continent"].astype(str) + " | " +
        out["Urban_Rural"].astype(str) + " | " +
        out["Education"].astype(str)
    )
    return out


def _load_human_answer_frame(human_data_path: Path, question_ids: list[str]) -> pd.DataFrame:
    question_ids = sorted(set(str(q) for q in question_ids))
    usecols = ["D_INTERVIEW"] + question_ids
    suffixes = ''.join(human_data_path.suffixes).lower()
    logger.info("Loading human answers from %s for questions=%s", human_data_path, question_ids)

    read_kwargs = {"usecols": lambda c: c in set(usecols), "low_memory": False}

    if suffixes.endswith('.zip'):
        df = pd.read_csv(human_data_path, compression='zip', **read_kwargs)
    elif suffixes.endswith('.csv'):
        df = pd.read_csv(human_data_path, **read_kwargs)
    elif suffixes.endswith('.tsv') or suffixes.endswith('.txt'):
        df = pd.read_csv(human_data_path, sep='	', **read_kwargs)
    else:
        raise ValueError(f"Unsupported human data file type: {human_data_path}")

    if "D_INTERVIEW" not in df.columns:
        raise KeyError(f"D_INTERVIEW column not found in human data file: {human_data_path}")

    df["D_INTERVIEW"] = df["D_INTERVIEW"].astype(str)
    df = df.set_index("D_INTERVIEW")
    logger.info("Loaded human answer frame shape=%s columns=%s", df.shape, list(df.columns))
    return df


def attach_human_scores_from_full(subset_df: pd.DataFrame, full_value_qa_path: Path | None) -> pd.DataFrame:
    if full_value_qa_path is None:
        logger.info("No human data configured; skipping external HUMAN_SCORE lookup")
        return subset_df

    if not full_value_qa_path.exists():
        logger.warning("Configured human data file does not exist: %s", full_value_qa_path)
        return subset_df

    subset = subset_df.copy()
    question_ids = subset["QUESTION_ID"].astype(str).unique().tolist()
    full_df = _load_human_answer_frame(full_value_qa_path, question_ids)

    scores = []
    missing_count = 0
    for row in subset[["QUESTION_ID", "PARTICIPANT_ID"]].to_dict(orient="records"):
        pid = str(row["PARTICIPANT_ID"])
        qid = str(row["QUESTION_ID"])
        value = None
        try:
            if pid in full_df.index and qid in full_df.columns:
                value = full_df.at[pid, qid]
        except Exception:
            value = None
        scores.append(value)
        if value is None or pd.isna(value):
            missing_count += 1

    subset["HUMAN_SCORE"] = pd.to_numeric(scores, errors="coerce")
    attached = int(subset["HUMAN_SCORE"].notna().sum())
    logger.info(
        "External HUMAN_SCORE lookup finished: attached=%d missing=%d total=%d source=%s",
        attached, missing_count, len(subset), full_value_qa_path,
    )
    return subset


def build_subset(
    samples_df: pd.DataFrame,
    question_ids: list[str],
    groups: list[dict[str, str]],
    participants_per_group_question: int,
    random_seed: int = 42,
) -> pd.DataFrame:
    df = canonicalize_samples(samples_df)
    df = df[df["QUESTION_ID"].isin(question_ids)].copy()

    parts = []
    for qid in question_ids:
        qdf = df[df["QUESTION_ID"] == qid]
        for group in groups:
            gdf = qdf[
                (qdf["Continent"] == group["continent"]) &
                (qdf["Urban_Rural"] == group["urban_rural"]) &
                (qdf["Education"] == group["education"])
            ].copy()
            if gdf.empty:
                logger.warning("No rows for qid=%s group=%s", qid, group)
                continue
            n = min(participants_per_group_question, len(gdf))
            sampled = gdf.sample(n=n, random_state=random_seed)
            parts.append(sampled)
            logger.info("Selected %d rows for qid=%s group=%s", n, qid, group)

    if not parts:
        raise ValueError("Subset is empty. Check question IDs and group labels.")

    subset = pd.concat(parts, ignore_index=True)
    base_cols = ["QUESTION_ID", "PARTICIPANT_ID", "Continent", "Urban_Rural", "Education", "GROUP_KEY"]
    if "HUMAN_SCORE" in subset.columns:
        base_cols.insert(2, "HUMAN_SCORE")
    subset = subset[base_cols].copy()
    logger.info("Built subset with shape=%s columns=%s", subset.shape, list(subset.columns))
    return subset
