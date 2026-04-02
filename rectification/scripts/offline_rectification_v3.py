
import argparse, json, math
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

PCOLS = [f'P_{i}' for i in range(1, 11)]

def aggregate_model_dist(df: pd.DataFrame) -> np.ndarray:
    vals = df[PCOLS].to_numpy(dtype=float)
    arr = np.nanmean(vals, axis=0)
    arr = arr[~np.isnan(arr)]
    s = float(arr.sum())
    if arr.size == 0 or s <= 0:
        raise ValueError("Invalid model probabilities")
    return arr / s

def aggregate_human_dist(df: pd.DataFrame) -> np.ndarray:
    scores = df['HUMAN_SCORE'].dropna().astype(int)
    if scores.empty:
        raise ValueError("No HUMAN_SCORE")
    m = int(scores.max())
    counts = np.bincount(scores, minlength=m + 1)[1:]
    return counts / counts.sum()

def emd_discrete(p: np.ndarray, q: np.ndarray) -> float:
    n = max(len(p), len(q))
    p = np.pad(np.array(p, dtype=float), (0, n - len(p)))
    q = np.pad(np.array(q, dtype=float), (0, n - len(q)))
    denom = (n - 1) if n > 1 else 1
    return float(np.abs(np.cumsum(p - q)).sum() / denom)

def entropy(p: np.ndarray) -> float:
    p = np.array(p, dtype=float)
    return float(-(p * np.log(np.clip(p, 1e-12, 1.0))).sum())

def build_cells(df: pd.DataFrame, qmeta: dict) -> pd.DataFrame:
    rows = []
    for (qid, group_key), gdf in df.groupby(['QUESTION_ID', 'GROUP_KEY']):
        p = aggregate_model_dist(gdf)
        h = aggregate_human_dist(gdf)
        n = max(len(p), len(h))
        if len(p) < n:
            p = np.pad(p, (0, n - len(p)))
            p = p / p.sum()
        if len(h) < n:
            h = np.pad(h, (0, n - len(h)))
            h = h / h.sum()
        rows.append({
            'QUESTION_ID': qid,
            'GROUP_KEY': group_key,
            'category': qmeta.get(qid, {}).get('category', 'UNKNOWN'),
            'question': qmeta.get(qid, {}).get('question', ''),
            'Continent': gdf['Continent'].iloc[0],
            'Urban_Rural': gdf['Urban_Rural'].iloc[0],
            'Education': gdf['Education'].iloc[0],
            'n_options': n,
            'model_dist': p,
            'human_dist': h,
            'baseline_w1': emd_discrete(p, h),
            'baseline_top_hit': int(np.argmax(p) == np.argmax(h)),
            'baseline_entropy': entropy(p),
        })
    return pd.DataFrame(rows)

def geometric_ratio(model_sum, human_sum, eps=1e-4):
    m = np.array(model_sum, dtype=float) + eps
    h = np.array(human_sum, dtype=float) + eps
    r = h / m
    gm = math.exp(float(np.log(r).mean()))
    return r / gm

def fit_bias_tables(train: pd.DataFrame, eps: float = 1e-4, shrink: float = 5.0):
    global_stats = defaultdict(lambda: {'m': None, 'h': None, 'count': 0})
    cat_stats = defaultdict(lambda: {'m': None, 'h': None, 'count': 0})

    def upd(bucket, key, p, h):
        if bucket[key]['m'] is None:
            bucket[key]['m'] = np.zeros_like(p, dtype=float)
            bucket[key]['h'] = np.zeros_like(h, dtype=float)
        bucket[key]['m'] += p
        bucket[key]['h'] += h
        bucket[key]['count'] += 1

    for _, r in train.iterrows():
        p, h = r['model_dist'], r['human_dist']
        n = int(r['n_options'])
        upd(global_stats, n, p, h)
        upd(cat_stats, (r['category'], n), p, h)

    glob = {}
    cat = {}
    for n, s in global_stats.items():
        glob[n] = geometric_ratio(s['m'], s['h'], eps=eps)

    for key, s in cat_stats.items():
        n = key[1]
        base = global_stats[n]
        m = s['m'] + shrink * (base['m'] / max(base['count'], 1))
        h = s['h'] + shrink * (base['h'] / max(base['count'], 1))
        cat[key] = geometric_ratio(m, h, eps=eps)
    return glob, cat

def correct_dist(row, glob, cat, lg=0.0, lc=1.5, selected_categories=None):
    p = np.array(row['model_dist'], dtype=float)
    n = int(row['n_options'])
    rg = np.array(glob.get(n, np.ones(n)), dtype=float)
    rc = np.array(cat.get((row['category'], n), np.ones(n)), dtype=float)
    if selected_categories is not None and row['category'] not in selected_categories:
        rc = np.ones(n)
    w = p * (rg ** lg) * (rc ** lc)
    s = float(w.sum())
    if s <= 0 or not np.isfinite(s):
        return p / p.sum()
    return w / s

def apply_bias_tables(cells: pd.DataFrame, glob, cat, lg, lc, selected_categories):
    out = cells.copy()
    out['corrected_dist'] = out.apply(
        lambda r: correct_dist(r, glob, cat, lg=lg, lc=lc, selected_categories=selected_categories), axis=1
    )
    out['corrected_w1'] = out.apply(lambda r: emd_discrete(r['corrected_dist'], r['human_dist']), axis=1)
    out['corrected_top_hit'] = out.apply(lambda r: int(np.argmax(r['corrected_dist']) == np.argmax(r['human_dist'])), axis=1)
    out['corrected_entropy'] = out['corrected_dist'].apply(entropy)
    return out

def summarize(df: pd.DataFrame, label: str) -> dict:
    return {
        'split': label,
        'rows': int(len(df)),
        'baseline_mean_w1': float(df['baseline_w1'].mean()),
        'corrected_mean_w1': float(df['corrected_w1'].mean()),
        'baseline_top_hit_rate': float(df['baseline_top_hit'].mean()),
        'corrected_top_hit_rate': float(df['corrected_top_hit'].mean()),
        'delta_mean_w1': float(df['corrected_w1'].mean() - df['baseline_w1'].mean()),
        'delta_top_hit_rate': float(df['corrected_top_hit'].mean() - df['baseline_top_hit'].mean()),
        'mean_corrected_entropy': float(df['corrected_entropy'].mean()),
    }

def bootstrap_delta(df: pd.DataFrame, n_boot: int = 3000, seed: int = 13) -> dict:
    rng = np.random.default_rng(seed)
    arr = (df['corrected_w1'] - df['baseline_w1']).to_numpy(dtype=float)
    if len(arr) == 0:
        return {}
    boots = []
    for _ in range(n_boot):
        sample = rng.choice(arr, size=len(arr), replace=True)
        boots.append(sample.mean())
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return {
        'delta_mean_w1_bootstrap_mean': float(np.mean(boots)),
        'delta_mean_w1_bootstrap_ci95_low': float(lo),
        'delta_mean_w1_bootstrap_ci95_high': float(hi),
    }

def parse_categories(s: str):
    if not s:
        return None
    return [x.strip() for x in s.split('||') if x.strip()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--predictions', required=True)
    ap.add_argument('--question-metadata', required=True)
    ap.add_argument('--calibration-qids', required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--shrink', type=float, default=5.0)
    ap.add_argument('--global-weight', type=float, default=0.0)
    ap.add_argument('--category-weight', type=float, default=1.5)
    ap.add_argument('--apply-categories', type=str, default='')
    args = ap.parse_args()

    selected_categories = parse_categories(args.apply_categories)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.predictions, sep='\t')
    qmeta = json.load(open(args.question_metadata, 'r', encoding='utf-8'))
    cal_qids = pd.read_csv(args.calibration_qids)['QUESTION_ID'].astype(str).tolist()

    cells = build_cells(df, qmeta)
    cells['split'] = np.where(cells['QUESTION_ID'].isin(cal_qids), 'calibration', 'evaluation')
    train = cells[cells['split'] == 'calibration'].copy()

    glob, cat = fit_bias_tables(train, shrink=args.shrink)
    corrected = apply_bias_tables(cells, glob, cat, args.global_weight, args.category_weight, selected_categories)
    cal_corr = corrected[corrected['split'] == 'calibration'].copy()
    eval_corr = corrected[corrected['split'] == 'evaluation'].copy()

    summary = {
        'method': 'selective_family_rectification_v3',
        'shrink': args.shrink,
        'weights': {'global': args.global_weight, 'category': args.category_weight},
        'selected_categories': selected_categories,
        'overall': summarize(corrected, 'overall'),
        'calibration': summarize(cal_corr, 'calibration'),
        'evaluation': summarize(eval_corr, 'evaluation'),
        'evaluation_bootstrap': bootstrap_delta(eval_corr),
        'warning': 'Targeted post-selection rectification. If families were chosen using prior held-out analysis, evaluation is optimistic and not a fresh generalization test.'
    }

    category_eval = (
        eval_corr.groupby('category')
        .agg(
            questions=('QUESTION_ID', 'nunique'),
            rows=('QUESTION_ID', 'count'),
            baseline_mean_w1=('baseline_w1', 'mean'),
            corrected_mean_w1=('corrected_w1', 'mean'),
            baseline_top_hit_rate=('baseline_top_hit', 'mean'),
            corrected_top_hit_rate=('corrected_top_hit', 'mean'),
            mean_corrected_entropy=('corrected_entropy', 'mean'),
        )
        .reset_index()
    )
    category_eval['delta_mean_w1'] = category_eval['corrected_mean_w1'] - category_eval['baseline_mean_w1']
    category_eval['delta_top_hit_rate'] = category_eval['corrected_top_hit_rate'] - category_eval['baseline_top_hit_rate']

    question_eval = (
        eval_corr.groupby(['QUESTION_ID', 'question', 'category'])
        .agg(
            rows=('QUESTION_ID', 'count'),
            baseline_mean_w1=('baseline_w1', 'mean'),
            corrected_mean_w1=('corrected_w1', 'mean'),
            baseline_top_hit_rate=('baseline_top_hit', 'mean'),
            corrected_top_hit_rate=('corrected_top_hit', 'mean'),
            n_options=('n_options', 'first'),
            mean_corrected_entropy=('corrected_entropy', 'mean'),
        )
        .reset_index()
    )
    question_eval['delta_mean_w1'] = question_eval['corrected_mean_w1'] - question_eval['baseline_mean_w1']
    question_eval['delta_top_hit_rate'] = question_eval['corrected_top_hit_rate'] - question_eval['baseline_top_hit_rate']

    group_eval = (
        eval_corr.groupby(['GROUP_KEY', 'Continent', 'Urban_Rural', 'Education'])
        .agg(
            rows=('QUESTION_ID', 'count'),
            baseline_mean_w1=('baseline_w1', 'mean'),
            corrected_mean_w1=('corrected_w1', 'mean'),
            baseline_top_hit_rate=('baseline_top_hit', 'mean'),
            corrected_top_hit_rate=('corrected_top_hit', 'mean'),
        )
        .reset_index()
    )
    group_eval['delta_mean_w1'] = group_eval['corrected_mean_w1'] - group_eval['baseline_mean_w1']
    group_eval['delta_top_hit_rate'] = group_eval['corrected_top_hit_rate'] - group_eval['baseline_top_hit_rate']

    save_cells = corrected.copy()
    save_cells['model_dist_json'] = save_cells['model_dist'].apply(lambda x: json.dumps(np.asarray(x).tolist(), ensure_ascii=False))
    save_cells['human_dist_json'] = save_cells['human_dist'].apply(lambda x: json.dumps(np.asarray(x).tolist(), ensure_ascii=False))
    save_cells['corrected_dist_json'] = save_cells['corrected_dist'].apply(lambda x: json.dumps(np.asarray(x).tolist(), ensure_ascii=False))
    save_cells = save_cells.drop(columns=['model_dist', 'human_dist', 'corrected_dist'])

    save_cells.to_csv(out_dir / 'cell_level_rectification_v3.tsv', sep='\t', index=False)
    category_eval.to_csv(out_dir / 'category_rectification_eval_v3.tsv', sep='\t', index=False)
    question_eval.to_csv(out_dir / 'question_rectification_eval_v3.tsv', sep='\t', index=False)
    group_eval.to_csv(out_dir / 'group_rectification_eval_v3.tsv', sep='\t', index=False)
    json.dump(summary, open(out_dir / 'summary_v3.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=2)

if __name__ == '__main__':
    main()
