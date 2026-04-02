
import argparse, json, math, itertools
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
            p = np.pad(p, (0, n - len(p))); p = p / p.sum()
        if len(h) < n:
            h = np.pad(h, (0, n - len(h))); h = h / h.sum()
        rows.append({
            'QUESTION_ID': qid,
            'GROUP_KEY': group_key,
            'category': qmeta.get(qid, {}).get('category', 'UNKNOWN'),
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

def make_clean_split(cells: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    split_map = {}
    for cat in sorted(cells['category'].unique()):
        qids = sorted(cells[cells['category'] == cat]['QUESTION_ID'].unique(), key=lambda x: int(str(x)[1:]))
        if len(qids) != 3:
            raise ValueError(f"Expected exactly 3 questions for category {cat}, got {len(qids)}")
        split_map[cat] = {'calibration': qids[0], 'dev': qids[1], 'test': qids[2]}
    qid_to_split = {}
    for cat, mapping in split_map.items():
        for split, qid in mapping.items():
            qid_to_split[qid] = split
    out = cells.copy()
    out['split'] = out['QUESTION_ID'].map(qid_to_split)
    return out, split_map

def geometric_ratio(model_sum, human_sum, eps=1e-4):
    m = np.array(model_sum, dtype=float) + eps
    h = np.array(human_sum, dtype=float) + eps
    r = h / m
    gm = math.exp(float(np.log(r).mean()))
    return r / gm

def fit_bias_tables(train: pd.DataFrame, eps: float = 1e-4, shrink: float = 5.0):
    global_stats = defaultdict(lambda: {'m': None, 'h': None, 'count': 0})
    cat_stats = defaultdict(lambda: {'m': None, 'h': None, 'count': 0})
    group_stats = defaultdict(lambda: {'m': None, 'h': None, 'count': 0})

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
        upd(group_stats, (r['GROUP_KEY'], n), p, h)

    glob, cat, grp = {}, {}, {}
    for n, s in global_stats.items():
        glob[n] = geometric_ratio(s['m'], s['h'], eps=eps)
    for key, s in cat_stats.items():
        n = key[1]; base = global_stats[n]
        m = s['m'] + shrink * (base['m'] / max(base['count'], 1))
        h = s['h'] + shrink * (base['h'] / max(base['count'], 1))
        cat[key] = geometric_ratio(m, h, eps=eps)
    for key, s in group_stats.items():
        n = key[1]; base = global_stats[n]
        m = s['m'] + shrink * (base['m'] / max(base['count'], 1))
        h = s['h'] + shrink * (base['h'] / max(base['count'], 1))
        grp[key] = geometric_ratio(m, h, eps=eps)
    return glob, cat, grp

def correct_row(row, glob, cat, grp, wg, wc, wr):
    p = np.array(row['model_dist'], dtype=float)
    n = int(row['n_options'])
    rg = np.array(glob.get(n, np.ones(n)), dtype=float)
    rc = np.array(cat.get((row['category'], n), np.ones(n)), dtype=float)
    rr = np.array(grp.get((row['GROUP_KEY'], n), np.ones(n)), dtype=float)
    w = p * (rg ** wg) * (rc ** wc) * (rr ** wr)
    s = float(w.sum())
    if s <= 0 or not np.isfinite(s):
        return p / p.sum()
    return w / s

def apply_selection(cells, glob, cat, grp, selection):
    out = cells.copy()
    corrs = []
    for _, r in out.iterrows():
        sel = selection.get(r['category'])
        if sel is None:
            corr = np.array(r['model_dist'], dtype=float)
        else:
            wg, wc, wr = sel['weights']
            corr = correct_row(r, glob, cat, grp, wg, wc, wr)
        corrs.append(corr)
    out['corrected_dist'] = corrs
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
        'baseline_entropy': float(df['baseline_entropy'].mean()),
        'corrected_entropy': float(df['corrected_entropy'].mean()),
        'delta_mean_w1': float(df['corrected_w1'].mean() - df['baseline_w1'].mean()),
        'delta_top_hit_rate': float(df['corrected_top_hit'].mean() - df['baseline_top_hit'].mean()),
        'delta_entropy': float(df['corrected_entropy'].mean() - df['baseline_entropy'].mean()),
    }

def bootstrap_delta(df: pd.DataFrame, n_boot: int = 5000, seed: int = 13) -> dict:
    rng = np.random.default_rng(seed)
    arr = (df['corrected_w1'] - df['baseline_w1']).to_numpy(dtype=float)
    if len(arr) == 0:
        return {}
    boots = [rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_boot)]
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return {
        'delta_mean_w1_bootstrap_mean': float(np.mean(boots)),
        'delta_mean_w1_bootstrap_ci95_low': float(lo),
        'delta_mean_w1_bootstrap_ci95_high': float(hi),
    }

def choose_categories_on_dev(dev: pd.DataFrame, glob, cat, grp):
    grid = list(itertools.product([0.0, 0.25, 0.5], [0.5, 1.0, 1.5, 2.0], [0.0, 0.25, 0.5]))
    selection = {}
    dev_grid_records = []
    for category, cdev in dev.groupby('category'):
        best = None
        for wg, wc, wr in grid:
            sel = {category: {'weights': (wg, wc, wr)}}
            corr = apply_selection(cdev, glob, cat, grp, sel)
            summ = summarize(corr, 'dev')
            ok = (
                summ['delta_mean_w1'] < -0.005 and
                summ['delta_top_hit_rate'] >= -0.05 and
                summ['corrected_entropy'] >= 0.75 * summ['baseline_entropy']
            )
            score = (
                summ['corrected_mean_w1']
                + max(0.0, -summ['delta_top_hit_rate']) * 0.05
                + max(0.0, 0.75 * summ['baseline_entropy'] - summ['corrected_entropy']) * 0.05
            )
            dev_grid_records.append({
                'category': category, 'wg': wg, 'wc': wc, 'wr': wr, 'ok': ok,
                **summ
            })
            if ok and (best is None or score < best[0]):
                best = (score, (wg, wc, wr), summ)
        if best is not None:
            selection[category] = {'weights': best[1], 'dev_summary': best[2]}
    return selection, pd.DataFrame(dev_grid_records)

def build_tables(corrected: pd.DataFrame):
    cat = corrected.groupby('category').agg(
        rows=('QUESTION_ID', 'count'),
        questions=('QUESTION_ID', 'nunique'),
        baseline_mean_w1=('baseline_w1', 'mean'),
        corrected_mean_w1=('corrected_w1', 'mean'),
        baseline_top_hit_rate=('baseline_top_hit', 'mean'),
        corrected_top_hit_rate=('corrected_top_hit', 'mean'),
        baseline_entropy=('baseline_entropy', 'mean'),
        corrected_entropy=('corrected_entropy', 'mean'),
    ).reset_index()
    cat['delta_mean_w1'] = cat['corrected_mean_w1'] - cat['baseline_mean_w1']
    cat['delta_top_hit_rate'] = cat['corrected_top_hit_rate'] - cat['baseline_top_hit_rate']
    cat['delta_entropy'] = cat['corrected_entropy'] - cat['baseline_entropy']

    q = corrected.groupby(['QUESTION_ID', 'category']).agg(
        rows=('QUESTION_ID', 'count'),
        baseline_mean_w1=('baseline_w1', 'mean'),
        corrected_mean_w1=('corrected_w1', 'mean'),
        baseline_top_hit_rate=('baseline_top_hit', 'mean'),
        corrected_top_hit_rate=('corrected_top_hit', 'mean'),
        baseline_entropy=('baseline_entropy', 'mean'),
        corrected_entropy=('corrected_entropy', 'mean'),
    ).reset_index()
    q['delta_mean_w1'] = q['corrected_mean_w1'] - q['baseline_mean_w1']
    q['delta_top_hit_rate'] = q['corrected_top_hit_rate'] - q['baseline_top_hit_rate']
    q['delta_entropy'] = q['corrected_entropy'] - q['baseline_entropy']

    g = corrected.groupby(['GROUP_KEY', 'Continent', 'Urban_Rural', 'Education']).agg(
        rows=('QUESTION_ID', 'count'),
        baseline_mean_w1=('baseline_w1', 'mean'),
        corrected_mean_w1=('corrected_w1', 'mean'),
        baseline_top_hit_rate=('baseline_top_hit', 'mean'),
        corrected_top_hit_rate=('corrected_top_hit', 'mean'),
    ).reset_index()
    g['delta_mean_w1'] = g['corrected_mean_w1'] - g['baseline_mean_w1']
    g['delta_top_hit_rate'] = g['corrected_top_hit_rate'] - g['baseline_top_hit_rate']
    return cat, q, g

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--predictions', required=True)
    ap.add_argument('--question-metadata', required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--shrink', type=float, default=5.0)
    args = ap.parse_args()

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.predictions, sep='\t')
    qmeta = json.load(open(args.question_metadata, 'r', encoding='utf-8'))

    cells = build_cells(df, qmeta)
    cells, split_map = make_clean_split(cells)

    train = cells[cells['split'] == 'calibration'].copy()
    dev = cells[cells['split'] == 'dev'].copy()
    test = cells[cells['split'] == 'test'].copy()

    glob, cat, grp = fit_bias_tables(train, shrink=args.shrink)
    selection, dev_grid = choose_categories_on_dev(dev, glob, cat, grp)
    corrected = apply_selection(cells, glob, cat, grp, selection)

    cal_corr = corrected[corrected['split'] == 'calibration'].copy()
    dev_corr = corrected[corrected['split'] == 'dev'].copy()
    test_corr = corrected[corrected['split'] == 'test'].copy()

    summary = {
        'method': 'clean_family_group_rectification_v4',
        'design': '12 calibration / 12 dev / 12 test, one question per family in each split',
        'shrink': args.shrink,
        'split_map': split_map,
        'selected_categories': list(selection.keys()),
        'selection': selection,
        'overall': summarize(corrected, 'overall'),
        'calibration': summarize(cal_corr, 'calibration'),
        'dev': summarize(dev_corr, 'dev'),
        'test': summarize(test_corr, 'test'),
        'test_bootstrap': bootstrap_delta(test_corr),
        'warning': 'Clean v4 selection: categories and weights were chosen using only dev after fitting bias tables on calibration; test remained untouched.'
    }

    cat_tbl, q_tbl, g_tbl = build_tables(test_corr)
    cell_save = corrected.copy()
    cell_save['model_dist_json'] = cell_save['model_dist'].apply(lambda x: json.dumps(np.asarray(x).tolist(), ensure_ascii=False))
    cell_save['human_dist_json'] = cell_save['human_dist'].apply(lambda x: json.dumps(np.asarray(x).tolist(), ensure_ascii=False))
    cell_save['corrected_dist_json'] = cell_save['corrected_dist'].apply(lambda x: json.dumps(np.asarray(x).tolist(), ensure_ascii=False))
    cell_save = cell_save.drop(columns=['model_dist', 'human_dist', 'corrected_dist'])

    dev_grid.to_csv(out_dir / 'dev_grid_search_v4.tsv', sep='\t', index=False)
    cell_save.to_csv(out_dir / 'cell_level_rectification_v4.tsv', sep='\t', index=False)
    cat_tbl.to_csv(out_dir / 'category_rectification_eval_v4.tsv', sep='\t', index=False)
    q_tbl.to_csv(out_dir / 'question_rectification_eval_v4.tsv', sep='\t', index=False)
    g_tbl.to_csv(out_dir / 'group_rectification_eval_v4.tsv', sep='\t', index=False)
    json.dump(summary, open(out_dir / 'summary_v4.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=2)

if __name__ == '__main__':
    main()
