from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import sys
import time
import zipfile
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque

import httpx
import numpy as np
import pandas as pd
import yaml

from src.config_resolver import expand_env_tree, make_config_env, resolve_repo_path
from tqdm import tqdm

MAX_OPTION_COL = 10
BASE_FIELDS = [
    'QUESTION_ID','PARTICIPANT_ID','GROUP_KEY','Continent','Urban_Rural','Education',
    'MODEL_SCORE','MOST_LIKELY_ANSWER','DECLARED_ENTROPY','prompt_mode','model','HUMAN_SCORE'
]
ALL_FIELDS = BASE_FIELDS + [f'P_{i}' for i in range(1, MAX_OPTION_COL+1)]

DEFAULT_36_QIDS = [
    'Q1','Q2','Q3','Q46','Q47','Q48','Q57','Q58','Q59','Q106','Q107','Q108',
    'Q112','Q113','Q114','Q121','Q122','Q123','Q131','Q132','Q133','Q158','Q159','Q160',
    'Q164','Q165','Q166','Q176','Q177','Q178','Q199','Q200','Q201','Q235','Q236','Q237'
]
DEFAULT_GROUPS_6 = [
    {'continent': 'Europe', 'urban_rural': 'Urban', 'education': 'Tertiary'},
    {'continent': 'Europe', 'urban_rural': 'Rural', 'education': 'Lower Secondary'},
    {'continent': 'Asia', 'urban_rural': 'Urban', 'education': 'Upper to Post Secondary'},
    {'continent': 'Africa', 'urban_rural': 'Rural', 'education': 'Primary or No Education'},
    {'continent': 'North America', 'urban_rural': 'Urban', 'education': 'Tertiary'},
    {'continent': 'South America', 'urban_rural': 'Urban', 'education': 'Lower Secondary'},
]

logger = logging.getLogger('cerebras_distribution')


def setup_logging(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    log_dir = run_dir / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"cerebras_distribution_{time.strftime('%Y%m%d_%H%M%S')}.log"
    handlers = [logging.StreamHandler(), logging.FileHandler(log_path, encoding='utf-8')]
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(name)s | %(message)s', handlers=handlers)
    logger.info('Logging initialized. Log file: %s', log_path)


@dataclass
class Config:
    raw: dict[str, Any]
    @property
    def api(self) -> dict[str, Any]: return self.raw['api']
    @property
    def paths(self) -> dict[str, Any]: return self.raw['paths']
    @property
    def benchmark(self) -> dict[str, Any]: return self.raw['benchmark']
    @property
    def limits(self) -> dict[str, Any]: return self.raw['limits']
    @property
    def retries(self) -> dict[str, Any]: return self.raw.get('retries', {})


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Run WVS distribution benchmark on Cerebras with persistent rate limiting and retries.')
    p.add_argument('--config', required=True)
    p.add_argument('--run-name', default=None)
    p.add_argument('--resume-run-dir', default=None)
    p.add_argument('--test-run', action='store_true', help='Only use first 6 questions')
    p.add_argument('--fail-on-error', action='store_true')
    p.add_argument('--model', default=None, help='Override api.model from YAML')
    p.add_argument('--base-url', default=None, help='Override api.base_url from YAML')
    p.add_argument('--api-key-env', default=None, help='Override api.api_key_env from YAML')
    p.add_argument('--outputs-dir', default=None, help='Override paths.outputs_dir from YAML')
    return p.parse_args()


def load_config(path: Path) -> Config:
    env = make_config_env(path)
    with open(path, 'r', encoding='utf-8') as f:
        return Config(expand_env_tree(yaml.safe_load(f), env))


def resolve_path(project_root: Path, p: str) -> Path:
    candidate = Path(p)
    if candidate.is_absolute():
        return candidate
    return (project_root / candidate).resolve()


def load_samples(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep='\t')
    logger.info('Loaded samples: shape=%s columns=%s', df.shape, list(df.columns))
    return df.rename(columns={'Question': 'QUESTION_ID', 'D_INTERVIEW': 'PARTICIPANT_ID', 'Urban / Rural': 'Urban_Rural'})


def build_subset(df: pd.DataFrame, question_ids: list[str], groups: list[dict[str, str]], participants_per_group_question: int, random_seed: int) -> pd.DataFrame:
    parts = []
    for qid in question_ids:
        qdf = df[df['QUESTION_ID'] == qid]
        for group in groups:
            gdf = qdf[(qdf['Continent'] == group['continent']) & (qdf['Urban_Rural'] == group['urban_rural']) & (qdf['Education'] == group['education'])].copy()
            if gdf.empty:
                logger.warning('No rows for qid=%s group=%s', qid, group)
                continue
            n = min(participants_per_group_question, len(gdf))
            samp = gdf.sample(n=n, random_state=random_seed)
            samp['PARTICIPANT_ID'] = samp['PARTICIPANT_ID'].astype(str)
            samp['QUESTION_ID'] = samp['QUESTION_ID'].astype(str)
            samp['GROUP_KEY'] = samp['Continent'].astype(str) + ' | ' + samp['Urban_Rural'].astype(str) + ' | ' + samp['Education'].astype(str)
            parts.append(samp[['QUESTION_ID', 'PARTICIPANT_ID', 'Continent', 'Urban_Rural', 'Education', 'GROUP_KEY']])
    if not parts:
        raise RuntimeError('Subset is empty')
    out = pd.concat(parts, ignore_index=True)
    logger.info('Built subset shape=%s', out.shape)
    return out


def load_question_metadata(path: Path) -> dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def attach_human_scores(subset: pd.DataFrame, raw_zip_path: Path, question_ids: list[str]) -> pd.DataFrame:
    with zipfile.ZipFile(raw_zip_path, 'r') as zf:
        csv_names = [n for n in zf.namelist() if n.lower().endswith('.csv')]
        if not csv_names:
            raise RuntimeError('No CSV found in raw WVS zip')
        with zf.open(csv_names[0]) as f:
            full = pd.read_csv(f, usecols=lambda c: c in set(['D_INTERVIEW'] + question_ids), low_memory=False)
    long = full.melt(id_vars=['D_INTERVIEW'], var_name='QUESTION_ID', value_name='HUMAN_SCORE').dropna(subset=['HUMAN_SCORE']).copy()
    long['PARTICIPANT_ID'] = long['D_INTERVIEW'].astype(str)
    long['QUESTION_ID'] = long['QUESTION_ID'].astype(str)
    long['HUMAN_SCORE'] = pd.to_numeric(long['HUMAN_SCORE'], errors='coerce')
    long = long.dropna(subset=['HUMAN_SCORE'])
    long['HUMAN_SCORE'] = long['HUMAN_SCORE'].astype(int)
    merged = subset.merge(long[['PARTICIPANT_ID', 'QUESTION_ID', 'HUMAN_SCORE']], on=['PARTICIPANT_ID', 'QUESTION_ID'], how='left')
    logger.info('Attached HUMAN_SCORE: attached=%s missing=%s', merged['HUMAN_SCORE'].notna().sum(), merged['HUMAN_SCORE'].isna().sum())
    return merged


def json_candidates(text: str) -> list[str]:
    text = (text or '').strip()
    if text.startswith('```'):
        text = text.replace('```json', '').replace('```', '').strip()
    cands = [text]
    a, b = text.find('{'), text.rfind('}')
    if a != -1 and b != -1 and b > a:
        cands.append(text[a:b+1])
    out = []
    for c in cands:
        if c and c not in out:
            out.append(c)
    return out


def repair_json_text(s: str) -> str:
    return s.replace(',}', '}').replace(',]', ']')


def parse_json(text: str) -> dict[str, Any]:
    last = None
    for cand in json_candidates(text):
        for variant in [cand, repair_json_text(cand)]:
            try:
                return json.loads(variant)
            except Exception as e:
                last = e
    raise RuntimeError(f'JSON parse failed: {last}')


def normalize_probs(raw: dict[str, Any], score_min: int, score_max: int) -> dict[str, float]:
    probs = {}
    for i in range(score_min, score_max + 1):
        key = str(i)
        try:
            probs[key] = max(0.0, float(raw.get(key, 0.0)))
        except Exception:
            probs[key] = 0.0
    s = sum(probs.values())
    if s <= 0:
        n = score_max - score_min + 1
        return {str(i): 1.0 / n for i in range(score_min, score_max + 1)}
    return {k: v / s for k, v in probs.items()}


def sample_score(option_probs: dict[str, float], participant_id: str, qid: str, mode: str, seed_base: int = 17) -> int:
    keys = sorted(option_probs.keys(), key=lambda x: int(x))
    probs = np.array([option_probs[k] for k in keys], dtype=float)
    probs = probs / probs.sum()
    digest = hashlib.sha256(f'{participant_id}|{qid}|{mode}|{seed_base}'.encode('utf-8')).hexdigest()
    seed = int(digest[:16], 16) % (2**32)
    rng = np.random.default_rng(seed)
    return int(rng.choice([int(k) for k in keys], p=probs))


def entropy(option_probs: dict[str, float]) -> float:
    vals = np.array(list(option_probs.values()), dtype=float)
    vals = vals[vals > 0]
    return float(-(vals * np.log(vals)).sum()) if len(vals) else 0.0


def build_prompt(group: dict[str, str], qmeta: dict[str, Any], prompt_mode: str) -> list[dict[str, str]]:
    lo, hi = int(qmeta['answer_scale_min']), int(qmeta['answer_scale_max'])
    system = (
        'You are generating synthetic survey response distributions, not answering as an assistant. '
        'Do not mention being an AI system. Do not refuse. Stay inside the provided demographic context. '
        'Return only valid JSON.'
    )
    if prompt_mode == 'no_demo_distribution':
        body = f"""
Task: Estimate a plausible distribution of answers for a generic survey population.
Question category: {qmeta.get('category','')}
Question: {qmeta['question']}
Allowed integer answers: {lo} to {hi}
Return a probability for every allowed answer. Probabilities must sum to 1.
""".strip()
    elif prompt_mode == 'interview_direct_distribution':
        body = f"""
This is a survey interview simulation.
Respondent group:
- Continent: {group['continent']}
- Settlement type: {group['urban_rural']}
- Education: {group['education']}
Question category: {qmeta.get('category','')}
Survey question: {qmeta['question']}
Allowed integer answers: {lo} to {hi}
Estimate a plausible answer distribution for this group. Do not invent a biography.
""".strip()
    else:
        body = f"""
Estimate a plausible distribution of answers for respondents with this demographic context.
Respondent group:
- Continent: {group['continent']}
- Settlement type: {group['urban_rural']}
- Education: {group['education']}
Question category: {qmeta.get('category','')}
Question: {qmeta['question']}
Allowed integer answers: {lo} to {hi}
Return probabilities over the answer scale for this group.
""".strip()
    schema_description = (
        'JSON object with fields: option_probs (object with stringified answer values as keys and probabilities as values), '
        'most_likely_answer (integer), rationale_short (short string, max 30 words).'
    )
    return [
        {'role': 'system', 'content': system},
        {'role': 'user', 'content': body + '\n' + schema_description}
    ]


def build_json_schema(lo: int, hi: int) -> dict[str, Any]:
    option_props = {str(i): {'type': 'number'} for i in range(lo, hi + 1)}
    return {
        'type': 'object',
        'additionalProperties': False,
        'properties': {
            'option_probs': {
                'type': 'object',
                'additionalProperties': False,
                'properties': option_props,
                'required': [str(i) for i in range(lo, hi + 1)]
            },
            'most_likely_answer': {'type': 'integer'},
            'rationale_short': {'type': 'string'}
        },
        'required': ['option_probs', 'most_likely_answer', 'rationale_short']
    }


class SlidingWindowRateLimiter:
    def __init__(self, limits: dict[str, int], state_path: Path):
        self.req_min = int(limits['requests_per_minute'])
        self.req_hour = int(limits['requests_per_hour'])
        self.req_day = int(limits['requests_per_day'])
        self.tok_min = int(limits['tokens_per_minute'])
        self.tok_hour = int(limits['tokens_per_hour'])
        self.tok_day = int(limits['tokens_per_day'])
        self.state_path = state_path
        self.req_events: Deque[tuple[float, int]] = deque()
        self.tok_events: Deque[tuple[float, int]] = deque()
        self._load()

    def _load(self) -> None:
        if not self.state_path.exists():
            return
        try:
            data = json.loads(self.state_path.read_text(encoding='utf-8'))
            now = time.time()
            for ts, v in data.get('requests', []):
                if now - ts < 86400:
                    self.req_events.append((float(ts), int(v)))
            for ts, v in data.get('tokens', []):
                if now - ts < 86400:
                    self.tok_events.append((float(ts), int(v)))
        except Exception as e:
            logger.warning('Failed to load rate limit state: %s', e)

    def _save(self) -> None:
        payload = {
            'requests': list(self.req_events),
            'tokens': list(self.tok_events),
        }
        self.state_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')

    def _prune(self, now: float) -> None:
        while self.req_events and now - self.req_events[0][0] >= 86400:
            self.req_events.popleft()
        while self.tok_events and now - self.tok_events[0][0] >= 86400:
            self.tok_events.popleft()

    @staticmethod
    def _sum_within(events: Deque[tuple[float, int]], now: float, seconds: float) -> int:
        return sum(v for ts, v in events if now - ts < seconds)

    def acquire(self, est_tokens: int) -> None:
        while True:
            now = time.time()
            self._prune(now)
            req_min = self._sum_within(self.req_events, now, 60)
            req_hour = self._sum_within(self.req_events, now, 3600)
            req_day = self._sum_within(self.req_events, now, 86400)
            tok_min = self._sum_within(self.tok_events, now, 60)
            tok_hour = self._sum_within(self.tok_events, now, 3600)
            tok_day = self._sum_within(self.tok_events, now, 86400)

            waits = []
            if req_min + 1 > self.req_min:
                waits.append(60 - (now - min(ts for ts, _ in self.req_events if now - ts < 60)))
            if req_hour + 1 > self.req_hour:
                waits.append(3600 - (now - min(ts for ts, _ in self.req_events if now - ts < 3600)))
            if req_day + 1 > self.req_day:
                waits.append(86400 - (now - min(ts for ts, _ in self.req_events if now - ts < 86400)))
            if tok_min + est_tokens > self.tok_min:
                waits.append(60 - (now - min(ts for ts, _ in self.tok_events if now - ts < 60)))
            if tok_hour + est_tokens > self.tok_hour:
                waits.append(3600 - (now - min(ts for ts, _ in self.tok_events if now - ts < 3600)))
            if tok_day + est_tokens > self.tok_day:
                waits.append(86400 - (now - min(ts for ts, _ in self.tok_events if now - ts < 86400)))

            if not waits:
                self.req_events.append((now, 1))
                self.tok_events.append((now, int(est_tokens)))
                self._save()
                return
            sleep_s = max(1.0, max(waits) + 0.25)
            logger.info('Rate limiter sleeping %.1fs (est_tokens=%s)', sleep_s, est_tokens)
            time.sleep(sleep_s)

    def finalize(self, est_tokens: int, actual_tokens: int | None) -> None:
        if actual_tokens is None or actual_tokens == est_tokens:
            return
        delta = int(actual_tokens - est_tokens)
        now = time.time()
        self._prune(now)
        self.tok_events.append((now, delta))
        self._save()


def estimate_tokens(messages: list[dict[str, str]], max_completion_tokens: int) -> int:
    text = '\n'.join(m['content'] for m in messages)
    est_prompt = max(1, int(len(text) / 3.5))
    return est_prompt + int(max_completion_tokens)


def call_cerebras_with_retries(base_url: str, api_key: str, model: str, messages: list[dict[str, str]], schema: dict[str, Any], api_cfg: dict[str, Any], limiter: SlidingWindowRateLimiter, retry_cfg: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    max_attempts = int(retry_cfg.get('max_attempts', 6))
    base_sleep = float(retry_cfg.get('base_sleep_seconds', 2.0))
    timeout = int(api_cfg.get('timeout_seconds', 180))
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json',
    }
    if api_cfg.get('x_cerebras_version_patch') is not None:
        headers['X-Cerebras-Version-Patch'] = str(api_cfg['x_cerebras_version_patch'])

    payload: dict[str, Any] = {
        'model': model,
        'messages': messages,
        'temperature': float(api_cfg.get('temperature', 0.0)),
        'top_p': float(api_cfg.get('top_p', 1.0)),
        'max_completion_tokens': int(api_cfg.get('max_completion_tokens', 220)),
        'response_format': {
            'type': 'json_schema',
            'json_schema': {
                'name': 'survey_distribution',
                'strict': True,
                'schema': schema,
            },
        },
    }
    if api_cfg.get('clear_thinking') is not None:
        payload['clear_thinking'] = bool(api_cfg['clear_thinking'])

    est_tokens = estimate_tokens(messages, payload['max_completion_tokens'])
    last_err: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        limiter.acquire(est_tokens)
        try:
            with httpx.Client(timeout=timeout) as client:
                resp = client.post(f"{base_url.rstrip('/')}/chat/completions", headers=headers, json=payload)
            if resp.status_code in {408, 409, 425, 429, 500, 502, 503, 504}:
                raise httpx.HTTPStatusError(f'Transient status {resp.status_code}: {resp.text[:300]}', request=resp.request, response=resp)
            resp.raise_for_status()
            data = resp.json()
            usage = data.get('usage', {}) if isinstance(data, dict) else {}
            actual_tokens = usage.get('total_tokens')
            limiter.finalize(est_tokens, actual_tokens)
            choice = data['choices'][0]
            content = choice['message']['content']
            return content, {'usage': usage}
        except Exception as e:
            last_err = e
            transient = isinstance(e, httpx.HTTPStatusError) or isinstance(e, httpx.TransportError) or isinstance(e, httpx.TimeoutException)
            if attempt >= max_attempts or not transient:
                break
            sleep_s = min(120.0, base_sleep * (2 ** (attempt - 1)))
            logger.warning('API attempt %s/%s failed: %s ; sleeping %.1fs', attempt, max_attempts, e, sleep_s)
            time.sleep(sleep_s)
    assert last_err is not None
    raise last_err


def append_jsonl(path: Path, obj: dict[str, Any]) -> None:
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(obj, ensure_ascii=False) + '\n')


def read_completed_keys(path: Path) -> set[tuple[str, str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    try:
        df = pd.read_csv(path, sep='\t')
    except Exception:
        return set()
    cols = {'QUESTION_ID', 'GROUP_KEY', 'prompt_mode'}
    if not cols.issubset(df.columns):
        return set()
    return set(map(tuple, df[['QUESTION_ID', 'GROUP_KEY', 'prompt_mode']].drop_duplicates().itertuples(index=False, name=None)))


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config))
    project_root = resolve_repo_path(ROOT, '.')
    if args.model:
        cfg.api['model'] = args.model
    if args.base_url:
        cfg.api['base_url'] = args.base_url
    if args.api_key_env:
        cfg.api['api_key_env'] = args.api_key_env
    if args.outputs_dir:
        cfg.paths['outputs_dir'] = args.outputs_dir
    run_name = args.run_name or f"cerebras_distribution_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(args.resume_run_dir) if args.resume_run_dir else resolve_path(project_root, cfg.paths.get('outputs_dir', 'outputs/runs')) / run_name
    setup_logging(run_dir)

    api_key = os.getenv(cfg.api.get('api_key_env', 'CEREBRAS_API_KEY'))
    if not api_key:
        raise RuntimeError(f"Environment variable {cfg.api.get('api_key_env','CEREBRAS_API_KEY')} is not set")

    samples_path = resolve_path(project_root, cfg.paths['samples_file'])
    meta_path = resolve_path(project_root, cfg.paths['question_metadata_file'])
    wvs_zip = resolve_path(project_root, cfg.paths['full_value_qa_file'])

    question_ids = list(cfg.benchmark.get('questions', DEFAULT_36_QIDS))
    if args.test_run:
        question_ids = question_ids[:6]
    groups = list(cfg.benchmark.get('groups', DEFAULT_GROUPS_6))
    participants_per_group_question = int(cfg.benchmark.get('participants_per_group_question', 5))
    random_seed = int(cfg.benchmark.get('random_seed', 42))
    sampling_seed_base = int(cfg.benchmark.get('sampling_seed_base', 17))
    prompt_mode = cfg.benchmark.get('prompt_mode', 'direct_persona_distribution')
    model = cfg.api['model']

    samples = load_samples(samples_path)
    subset = build_subset(samples, question_ids, groups, participants_per_group_question, random_seed)
    subset = attach_human_scores(subset, wvs_zip, question_ids)
    meta = load_question_metadata(meta_path)

    subset_path = run_dir / 'subset.tsv'
    run_dir.mkdir(parents=True, exist_ok=True)
    subset.to_csv(subset_path, sep='\t', index=False)

    predictions_path = run_dir / 'predictions.tsv'
    raw_path = run_dir / 'raw.jsonl'
    errors_path = run_dir / 'errors.jsonl'
    state_path = run_dir / 'rate_limit_state.json'

    if not predictions_path.exists():
        with open(predictions_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=ALL_FIELDS, delimiter='\t')
            writer.writeheader()

    completed = read_completed_keys(predictions_path)
    cells = subset[['QUESTION_ID', 'GROUP_KEY', 'Continent', 'Urban_Rural', 'Education']].drop_duplicates().to_dict(orient='records')
    logger.info('Resume status: completed_cells=%s total_cells=%s', len(completed), len(cells))

    limiter = SlidingWindowRateLimiter(cfg.limits, state_path)

    pbar = tqdm(cells, desc='cerebras_distribution', unit='cell')
    for cell in pbar:
        key = (cell['QUESTION_ID'], cell['GROUP_KEY'], prompt_mode)
        if key in completed:
            continue
        qid = cell['QUESTION_ID']
        qmeta = meta[qid]
        lo, hi = int(qmeta['answer_scale_min']), int(qmeta['answer_scale_max'])
        group = {'continent': cell['Continent'], 'urban_rural': cell['Urban_Rural'], 'education': cell['Education']}
        messages = build_prompt(group, qmeta, prompt_mode)
        schema = build_json_schema(lo, hi)
        try:
            content, aux = call_cerebras_with_retries(
                base_url=cfg.api.get('base_url', 'https://api.cerebras.ai/v1'),
                api_key=api_key,
                model=model,
                messages=messages,
                schema=schema,
                api_cfg=cfg.api,
                limiter=limiter,
                retry_cfg=cfg.retries,
            )
            parsed = parse_json(content)
            probs = normalize_probs(parsed['option_probs'], lo, hi)
            declared_entropy = entropy(probs)
            cell_rows = subset[(subset['QUESTION_ID'] == qid) & (subset['GROUP_KEY'] == cell['GROUP_KEY'])].copy()
            append_jsonl(raw_path, {
                'qid': qid,
                'group_key': cell['GROUP_KEY'],
                'prompt_mode': prompt_mode,
                'model': model,
                'messages': messages,
                'raw_content': content,
                'parsed': parsed,
                'usage': aux.get('usage'),
            })
            rows_to_write = []
            for _, row in cell_rows.iterrows():
                out = {k: '' for k in ALL_FIELDS}
                out.update({
                    'QUESTION_ID': row['QUESTION_ID'],
                    'PARTICIPANT_ID': row['PARTICIPANT_ID'],
                    'GROUP_KEY': row['GROUP_KEY'],
                    'Continent': row['Continent'],
                    'Urban_Rural': row['Urban_Rural'],
                    'Education': row['Education'],
                    'MODEL_SCORE': sample_score(probs, str(row['PARTICIPANT_ID']), qid, prompt_mode, sampling_seed_base),
                    'MOST_LIKELY_ANSWER': int(parsed['most_likely_answer']),
                    'DECLARED_ENTROPY': declared_entropy,
                    'prompt_mode': prompt_mode,
                    'model': model,
                    'HUMAN_SCORE': int(row['HUMAN_SCORE']) if pd.notna(row['HUMAN_SCORE']) else '',
                })
                for i in range(1, MAX_OPTION_COL + 1):
                    out[f'P_{i}'] = probs.get(str(i), '')
                rows_to_write.append(out)
            with open(predictions_path, 'a', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=ALL_FIELDS, delimiter='\t')
                writer.writerows(rows_to_write)
            completed.add(key)
        except Exception as e:
            logger.exception('Cell failed qid=%s group=%s', qid, cell['GROUP_KEY'])
            append_jsonl(errors_path, {
                'qid': qid,
                'group_key': cell['GROUP_KEY'],
                'prompt_mode': prompt_mode,
                'model': model,
                'error': repr(e),
            })
            if args.fail_on_error:
                raise

    logger.info('Finished. predictions=%s raw=%s errors=%s', predictions_path, raw_path, errors_path)
    print('\nNext step: run offline rectification on the produced predictions.tsv using your existing rectification scripts.')


if __name__ == '__main__':
    main()
