from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd
import yaml
from scipy.stats import wasserstein_distance
from tqdm import tqdm

import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config_resolver import expand_env_tree, make_config_env, resolve_repo_path

from src.distribution_prompts import build_distribution_prompt, build_distribution_schema
from src.lmstudio_rest import LMStudioRestClient

logger = logging.getLogger("distribution_benchmark")
MAX_OPTION_COL = 10
BASE_FIELDS = [
    'QUESTION_ID','PARTICIPANT_ID','GROUP_KEY','Continent','Urban_Rural','Education',
    'MODEL_SCORE','MOST_LIKELY_ANSWER','DECLARED_ENTROPY','prompt_mode','model','HUMAN_SCORE'
]
ALL_FIELDS = BASE_FIELDS + [f'P_{i}' for i in range(1, MAX_OPTION_COL+1)]


def setup_logging(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    log_dir = run_dir / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"distribution_{time.strftime('%Y%m%d_%H%M%S')}.log"
    handlers = [logging.StreamHandler(), logging.FileHandler(log_path, encoding='utf-8')]
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(name)s | %(message)s', handlers=handlers)
    logger.info('Logging initialized. Log file: %s', log_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--run-name', default=None)
    p.add_argument('--resume-run-dir', default=None)
    p.add_argument('--test-run', action='store_true')
    p.add_argument('--mock', action='store_true')
    p.add_argument('--fail-on-error', action='store_true')
    p.add_argument('--model', default=None, help='Override api.model from YAML')
    p.add_argument('--base-url', default=None, help='Override api.base_url from YAML')
    p.add_argument('--api-key', default=None, help='Override api.api_key from YAML')
    p.add_argument('--outputs-dir', default=None, help='Override paths.outputs_dir from YAML')
    return p.parse_args()


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
    def model_management(self) -> dict[str, Any]: return self.raw.get('model_management', {})


def load_config(path: Path) -> Config:
    env = make_config_env(path)
    with open(path, 'r', encoding='utf-8') as f:
        return Config(expand_env_tree(yaml.safe_load(f), env))


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
    s = s.replace(',}', '}').replace(',]', ']')
    # common minor quote cleanup not aggressive
    return s


def parse_json(text: str) -> dict[str, Any]:
    last = None
    for cand in json_candidates(text):
        for variant in [cand, repair_json_text(cand)]:
            try:
                return json.loads(variant)
            except Exception as e:
                last = e
    raise RuntimeError(f'JSON parse failed: {last}')


def chat_completion(base_url: str, api_key: str, model: str, messages: list[dict[str, str]], schema: dict[str, Any], cfg: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    payload: dict[str, Any] = {'model': model, 'messages': messages, 'temperature': float(cfg.get('temperature', 0))}
    if cfg.get('top_p') is not None:
        payload['top_p'] = float(cfg['top_p'])
    if cfg.get('max_completion_tokens') is not None:
        payload['max_completion_tokens'] = int(cfg['max_completion_tokens'])
    if cfg.get('seed') is not None:
        payload['seed'] = int(cfg['seed'])
    if cfg.get('use_structured_output', True):
        payload['response_format'] = {'type': 'json_schema', 'json_schema': {'name': 'survey_distribution', 'strict': bool(cfg.get('structured_output_strict', True)), 'schema': schema}}
    if cfg.get('logprobs'):
        payload['logprobs'] = True
        payload['top_logprobs'] = int(cfg.get('top_logprobs', 5))
    with httpx.Client(timeout=int(cfg.get('timeout_seconds', 120))) as client:
        resp = client.post(f"{base_url.rstrip('/')}/chat/completions", headers={'Authorization': f"Bearer {api_key}", 'Content-Type': 'application/json'}, json=payload)
        resp.raise_for_status()
        data = resp.json()
    choice = data['choices'][0]
    return choice['message']['content'], {'logprobs': choice.get('logprobs')}


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


def inter_group_std_ratio(pred_df: pd.DataFrame) -> float | None:
    if 'HUMAN_SCORE' not in pred_df.columns: return None
    vals = []
    for _, qdf in pred_df.groupby('QUESTION_ID'):
        hm = qdf.groupby('GROUP_KEY')['HUMAN_SCORE'].mean()
        mm = qdf.groupby('GROUP_KEY')['MODEL_SCORE'].mean()
        if hm.std(ddof=0) > 0:
            vals.append(float(mm.std(ddof=0) / hm.std(ddof=0)))
    return float(np.mean(vals)) if vals else None


def mean_unique_scores_per_cell(pred_df: pd.DataFrame, score_col: str) -> float:
    return float(pred_df.groupby(['QUESTION_ID', 'GROUP_KEY'])[score_col].nunique().mean())


def zero_variance_cell_rate(pred_df: pd.DataFrame, score_col: str) -> float:
    grp = pred_df.groupby(['QUESTION_ID', 'GROUP_KEY'])[score_col].nunique()
    return float((grp == 1).mean())


def mean_cell_entropy(pred_df: pd.DataFrame, score_col: str) -> float:
    ents = []
    for _, g in pred_df.groupby(['QUESTION_ID', 'GROUP_KEY']):
        p = g[score_col].value_counts(normalize=True)
        ents.append(float(-(p * np.log(p)).sum()))
    return float(np.mean(ents)) if ents else 0.0


def evaluate(pred_df: pd.DataFrame, meta: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if 'HUMAN_SCORE' in pred_df.columns and pred_df['HUMAN_SCORE'].notna().all():
        per_q, top_hits, q_deltas = {}, [], []
        for qid, g in pred_df.groupby('QUESTION_ID'):
            qmeta = meta[qid]
            lo, hi = int(qmeta['answer_scale_min']), int(qmeta['answer_scale_max'])
            human = ((g['HUMAN_SCORE'].astype(float) - lo) / (hi - lo)).tolist()
            model = ((g['MODEL_SCORE'].astype(float) - lo) / (hi - lo)).tolist()
            per_q[qid] = float(wasserstein_distance(human, model))
            # compare top human option per cell then average across cells
            for _, cg in g.groupby('GROUP_KEY'):
                top_hits.append(int(cg['HUMAN_SCORE'].mode().iat[0] == cg['MODEL_SCORE'].mode().iat[0]))
            q_deltas.append(float(g['MODEL_SCORE'].mean() - g['HUMAN_SCORE'].mean()))
        out['mean_w1'] = float(np.mean(list(per_q.values())))
        out['per_question_w1'] = per_q
        out['top_option_hit_rate'] = float(np.mean(top_hits))
        out['mean_score_delta'] = float(np.mean(q_deltas))
        out['human_unique_scores_per_cell'] = mean_unique_scores_per_cell(pred_df, 'HUMAN_SCORE')
        out['human_zero_variance_cell_rate'] = zero_variance_cell_rate(pred_df, 'HUMAN_SCORE')
        out['human_mean_cell_entropy'] = mean_cell_entropy(pred_df, 'HUMAN_SCORE')
    out['model_unique_scores_per_cell'] = mean_unique_scores_per_cell(pred_df, 'MODEL_SCORE')
    out['model_zero_variance_cell_rate'] = zero_variance_cell_rate(pred_df, 'MODEL_SCORE')
    out['model_mean_cell_entropy'] = mean_cell_entropy(pred_df, 'MODEL_SCORE')
    out['mean_declared_option_entropy'] = float(pred_df['DECLARED_ENTROPY'].mean()) if 'DECLARED_ENTROPY' in pred_df else None
    out['inter_group_std_ratio'] = inter_group_std_ratio(pred_df)
    return out


def maybe_load_model(cfg: Config):
    mm = cfg.model_management
    if not (mm.get('enabled') and mm.get('load_before_run')):
        return None, None
    client = LMStudioRestClient(base_url_v1=cfg.api['base_url'], api_key=cfg.api.get('api_key', 'lm-studio'), timeout_seconds=int(cfg.api.get('timeout_seconds', 120)))
    if not client.ping():
        raise RuntimeError(f"LM Studio REST API is not reachable at {cfg.api['base_url']}")
    instance_id = client.load_model(model=cfg.api['model'], context_length=mm.get('context_length'), eval_batch_size=mm.get('eval_batch_size'), flash_attention=mm.get('flash_attention'), num_experts=mm.get('num_experts'), offload_kv_cache_to_gpu=mm.get('offload_kv_cache_to_gpu'))
    return client, instance_id


def maybe_unload_model(cfg: Config, client, instance_id):
    if client and instance_id and cfg.model_management.get('enabled') and cfg.model_management.get('unload_after_run'):
        client.unload_model(instance_id)
    if client:
        client.close()


def robust_read_predictions(path: Path) -> pd.DataFrame:
    rows=[]
    with open(path,'r',encoding='utf-8', newline='') as f:
        reader=csv.DictReader(f, delimiter='\t')
        for rec in reader:
            rows.append(rec)
    if not rows:
        return pd.DataFrame(columns=ALL_FIELDS)
    df=pd.DataFrame(rows)
    for col in ['MODEL_SCORE','MOST_LIKELY_ANSWER','HUMAN_SCORE']+[f'P_{i}' for i in range(1,MAX_OPTION_COL+1)]:
        if col in df.columns:
            df[col]=pd.to_numeric(df[col], errors='coerce')
    if 'DECLARED_ENTROPY' in df.columns:
        df['DECLARED_ENTROPY']=pd.to_numeric(df['DECLARED_ENTROPY'], errors='coerce')
    return df


def existing_completed_keys(predictions_path: Path) -> set[tuple[str,str,str]]:
    if not predictions_path.exists():
        return set()
    try:
        df=robust_read_predictions(predictions_path)
        return set(zip(df['QUESTION_ID'].astype(str), df['PARTICIPANT_ID'].astype(str), df['prompt_mode'].astype(str)))
    except Exception as e:
        logger.warning('Could not read existing predictions for resume: %s', e)
        return set()


def row_to_fixed_fields(out: dict[str, Any]) -> dict[str, Any]:
    rec={k: out.get(k, '') for k in ALL_FIELDS}
    return rec


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config))
    if args.model:
        cfg.api['model'] = args.model
    if args.base_url:
        cfg.api['base_url'] = args.base_url
    if args.api_key:
        cfg.api['api_key'] = args.api_key
    if args.outputs_dir:
        cfg.paths['outputs_dir'] = args.outputs_dir
    mode = cfg.benchmark['prompt_modes'][0]
    model = cfg.api['model']
    qids = list(cfg.benchmark['questions'])[:12] if args.test_run else list(cfg.benchmark['questions'])
    groups = list(cfg.benchmark['groups'])[:3] if args.test_run else list(cfg.benchmark['groups'])

    if args.resume_run_dir:
        run_dir = Path(args.resume_run_dir)
    else:
        stamp = time.strftime('%Y%m%d_%H%M%S')
        suffix = (args.run_name or f"{model.split('/')[-1]}-{mode}").replace(' ', '_')
        run_dir = Path(cfg.paths['outputs_dir']) / 'runs' / f'distribution_{stamp}_{suffix}'
    setup_logging(run_dir)

    samples = load_samples(Path(cfg.paths['data_dir']) / cfg.paths['samples_file'])
    subset = build_subset(samples, qids, groups, int(cfg.benchmark['participants_per_group_question']), int(cfg.benchmark.get('random_seed', 42)))
    if cfg.paths.get('full_value_qa_file'):
        subset = attach_human_scores(subset, Path(cfg.paths['full_value_qa_file']), qids)
    subset.to_csv(run_dir / 'subset.tsv', sep='\t', index=False)
    meta = load_question_metadata(Path(cfg.paths['data_dir']) / cfg.paths['question_metadata_file'])

    predictions_path = run_dir / 'predictions.tsv'
    raw_path = run_dir / 'raw.jsonl'
    error_path = run_dir / 'errors.jsonl'

    if not predictions_path.exists():
        with open(predictions_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=ALL_FIELDS, delimiter='\t', extrasaction='ignore')
            writer.writeheader()

    completed = existing_completed_keys(predictions_path)
    logger.info('Resume status: already_completed=%s total_rows=%s', len(completed), len(subset))

    rest_client = None
    instance_id = None
    try:
        if not args.mock:
            rest_client, instance_id = maybe_load_model(cfg)

        with open(raw_path, 'a', encoding='utf-8') as rawf, open(error_path, 'a', encoding='utf-8') as errf, open(predictions_path, 'a', encoding='utf-8', newline='') as predf:
            writer = csv.DictWriter(predf, fieldnames=ALL_FIELDS, delimiter='\t', extrasaction='ignore')
            for row in tqdm(subset.to_dict(orient='records'), desc=f'{model}:{mode}'):
                key=(str(row['QUESTION_ID']), str(row['PARTICIPANT_ID']), mode)
                if key in completed:
                    continue
                qid = str(row['QUESTION_ID'])
                qmeta = meta[qid]
                lo, hi = int(qmeta['answer_scale_min']), int(qmeta['answer_scale_max'])
                prompt = build_distribution_prompt(mode, str(qmeta['question']), str(row['Continent']), str(row['Urban_Rural']), str(row['Education']), lo, hi)
                schema = build_distribution_schema(lo, hi)
                try:
                    if args.mock:
                        option_probs = {str(i): 1.0/(hi-lo+1) for i in range(lo, hi+1)}
                        parsed = {'option_probs': option_probs, 'most_likely_answer': lo, 'rationale_short': 'mock'}
                        raw_text = json.dumps(parsed, ensure_ascii=False)
                        extra = {}
                    else:
                        raw_text, extra = chat_completion(cfg.api['base_url'], cfg.api.get('api_key', 'lm-studio'), model, [
                            {'role': 'system', 'content': 'Return valid JSON only. No markdown. No extra text.'},
                            {'role': 'user', 'content': prompt},
                        ], schema, cfg.api)
                        parsed = parse_json(raw_text)
                        option_probs = normalize_probs(parsed.get('option_probs', {}), lo, hi)
                    declared_ent = entropy(option_probs)
                    sampled_score = sample_score(option_probs, str(row['PARTICIPANT_ID']), qid, mode, int(cfg.benchmark.get('sampling_seed_base', 17)))
                    out = {
                        'QUESTION_ID': qid,
                        'PARTICIPANT_ID': row['PARTICIPANT_ID'],
                        'GROUP_KEY': row['GROUP_KEY'],
                        'Continent': row['Continent'],
                        'Urban_Rural': row['Urban_Rural'],
                        'Education': row['Education'],
                        'MODEL_SCORE': sampled_score,
                        'MOST_LIKELY_ANSWER': int(parsed.get('most_likely_answer', sampled_score)),
                        'DECLARED_ENTROPY': declared_ent,
                        'prompt_mode': mode,
                        'model': model,
                        'HUMAN_SCORE': int(row['HUMAN_SCORE']) if ('HUMAN_SCORE' in row and pd.notna(row.get('HUMAN_SCORE'))) else '',
                    }
                    for i in range(1, MAX_OPTION_COL+1):
                        out[f'P_{i}'] = option_probs.get(str(i), '')
                    writer.writerow(row_to_fixed_fields(out))
                    predf.flush()
                    rawf.write(json.dumps({'prompt': prompt, 'raw_text': raw_text, 'parsed': parsed, 'extra': extra, **out}, ensure_ascii=False) + '\n')
                    rawf.flush()
                except Exception as e:
                    errf.write(json.dumps({'QUESTION_ID': qid, 'PARTICIPANT_ID': row['PARTICIPANT_ID'], 'GROUP_KEY': row['GROUP_KEY'], 'prompt_mode': mode, 'model': model, 'prompt': prompt, 'error': str(e), 'raw_text': locals().get('raw_text', None)}, ensure_ascii=False) + '\n')
                    errf.flush()
                    logger.exception('Row failed | qid=%s participant=%s', qid, row['PARTICIPANT_ID'])
                    if args.fail_on_error:
                        raise
        pred_df = robust_read_predictions(predictions_path)
        summary = evaluate(pred_df, meta)
        with open(run_dir / 'summary.json', 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        logger.info('Summary: %s', json.dumps(summary, ensure_ascii=False))
    finally:
        maybe_unload_model(cfg, rest_client, instance_id)


if __name__ == '__main__':
    main()
