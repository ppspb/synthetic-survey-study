import argparse
import json
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings('ignore')

PCOLS = [f'P_{i}' for i in range(1, 11)]


def aggregate_model_dist(df: pd.DataFrame) -> np.ndarray:
    arr = np.nanmean(df[PCOLS].to_numpy(dtype=float), axis=0)
    valid = ~np.isnan(arr)
    arr = arr[valid]
    if arr.size == 0:
        raise ValueError('No model probabilities available')
    s = float(arr.sum())
    if s <= 0:
        raise ValueError('Model probabilities sum to zero')
    return arr / s


def aggregate_human_dist(df: pd.DataFrame) -> np.ndarray:
    scores = df['HUMAN_SCORE'].dropna().astype(int)
    if scores.empty:
        raise ValueError('No HUMAN_SCORE values available')
    m = int(scores.max())
    counts = np.bincount(scores, minlength=m + 1)[1:]
    return counts / counts.sum()


def emd_discrete(p: np.ndarray, q: np.ndarray) -> float:
    n = max(len(p), len(q))
    p = np.pad(np.array(p, dtype=float), (0, n - len(p)))
    q = np.pad(np.array(q, dtype=float), (0, n - len(q)))
    denom = (n - 1) if n > 1 else 1
    return float(np.abs(np.cumsum(p - q)).sum() / denom)


def expected_score(p: np.ndarray) -> float:
    idx = np.arange(1, len(p) + 1)
    return float((idx * p).sum())


def normalized_mean(p: np.ndarray) -> float:
    n = len(p)
    return 0.0 if n <= 1 else float((expected_score(p) - 1) / (n - 1))


def tilt_distribution_to_target_mean(p: np.ndarray, target_norm_mean: float) -> np.ndarray:
    p = np.array(p, dtype=float)
    n = len(p)
    idx = np.arange(1, n + 1)
    target = 1 + float(np.clip(target_norm_mean, 0.0, 1.0)) * (n - 1)
    lo, hi = -8.0, 8.0
    for _ in range(80):
        mid = (lo + hi) / 2.0
        w = p * np.exp(mid * idx)
        w = w / w.sum()
        ex = float((idx * w).sum())
        if ex < target:
            lo = mid
        else:
            hi = mid
    lam = (lo + hi) / 2.0
    w = p * np.exp(lam * idx)
    w = w / w.sum()
    return w


def build_cell_frame(df: pd.DataFrame, qmeta: dict) -> pd.DataFrame:
    rows = []
    for (qid, group_key), gdf in df.groupby(['QUESTION_ID', 'GROUP_KEY']):
        p = aggregate_model_dist(gdf)
        h = aggregate_human_dist(gdf)
        rows.append({
            'QUESTION_ID': qid,
            'GROUP_KEY': group_key,
            'category': qmeta[qid]['category'],
            'question': qmeta[qid].get('question', ''),
            'Continent': gdf['Continent'].iloc[0],
            'Urban_Rural': gdf['Urban_Rural'].iloc[0],
            'Education': gdf['Education'].iloc[0],
            'n_options': len(p),
            'model_dist': p,
            'human_dist': h,
            'model_norm_mean': normalized_mean(p),
            'human_norm_mean': normalized_mean(h),
            'baseline_w1': emd_discrete(p, h),
            'baseline_top_hit': int(np.argmax(p) == np.argmax(h)),
        })
    return pd.DataFrame(rows)


def fit_ridge_rectifier(train: pd.DataFrame, alpha: float = 10.0):
    features = ['model_norm_mean', 'category', 'Continent', 'Urban_Rural', 'Education']
    pre = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), ['model_norm_mean']),
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['category', 'Continent', 'Urban_Rural', 'Education']),
        ]
    )
    model = Pipeline([
        ('pre', pre),
        ('reg', Ridge(alpha=alpha)),
    ])
    model.fit(train[features], train['human_norm_mean'])
    return model


def apply_ridge_rectifier(cells: pd.DataFrame, model) -> pd.DataFrame:
    features = ['model_norm_mean', 'category', 'Continent', 'Urban_Rural', 'Education']
    out = cells.copy()
    out['pred_norm_mean_corrected'] = np.clip(model.predict(out[features]), 0.0, 1.0)
    out['corrected_dist'] = out.apply(
        lambda r: tilt_distribution_to_target_mean(r['model_dist'], r['pred_norm_mean_corrected']), axis=1
    )
    out['corrected_w1'] = out.apply(lambda r: emd_discrete(r['corrected_dist'], r['human_dist']), axis=1)
    out['corrected_top_hit'] = out.apply(
        lambda r: int(np.argmax(r['corrected_dist']) == np.argmax(r['human_dist'])), axis=1
    )
    return out


def summarize(split_df: pd.DataFrame, label: str) -> dict:
    return {
        'split': label,
        'rows': int(len(split_df)),
        'baseline_mean_w1': float(split_df['baseline_w1'].mean()),
        'corrected_mean_w1': float(split_df['corrected_w1'].mean()),
        'baseline_top_hit_rate': float(split_df['baseline_top_hit'].mean()),
        'corrected_top_hit_rate': float(split_df['corrected_top_hit'].mean()),
        'delta_mean_w1': float(split_df['corrected_w1'].mean() - split_df['baseline_w1'].mean()),
        'delta_top_hit_rate': float(split_df['corrected_top_hit'].mean() - split_df['baseline_top_hit'].mean()),
    }


def bootstrap_delta(split_df: pd.DataFrame, n_boot: int = 2000, seed: int = 13) -> dict:
    rng = np.random.default_rng(seed)
    arr = (split_df['corrected_w1'] - split_df['baseline_w1']).to_numpy(dtype=float)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--predictions', required=True)
    ap.add_argument('--question-metadata', required=True)
    ap.add_argument('--calibration-qids', required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--alpha', type=float, default=10.0)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.predictions, sep='\t')
    qmeta = json.load(open(args.question_metadata, 'r', encoding='utf-8'))
    cal_qids = pd.read_csv(args.calibration_qids)['QUESTION_ID'].astype(str).tolist()

    cells = build_cell_frame(df, qmeta)
    cells['split'] = np.where(cells['QUESTION_ID'].isin(cal_qids), 'calibration', 'evaluation')

    train = cells[cells['split'] == 'calibration'].copy()
    eval_df = cells[cells['split'] == 'evaluation'].copy()

    model = fit_ridge_rectifier(train, alpha=args.alpha)
    corrected = apply_ridge_rectifier(cells, model)

    cal_corr = corrected[corrected['split'] == 'calibration'].copy()
    eval_corr = corrected[corrected['split'] == 'evaluation'].copy()

    summary = {
        'alpha': args.alpha,
        'overall': summarize(corrected, 'overall'),
        'calibration': summarize(cal_corr, 'calibration'),
        'evaluation': summarize(eval_corr, 'evaluation'),
        'evaluation_bootstrap': bootstrap_delta(eval_corr),
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
            model_norm_mean=('model_norm_mean', 'mean'),
            pred_norm_mean_corrected=('pred_norm_mean_corrected', 'mean'),
            human_norm_mean=('human_norm_mean', 'mean'),
        )
        .reset_index()
    )
    question_eval['delta_mean_w1'] = question_eval['corrected_mean_w1'] - question_eval['baseline_mean_w1']
    question_eval['delta_top_hit_rate'] = question_eval['corrected_top_hit_rate'] - question_eval['baseline_top_hit_rate']

    corrected.drop(columns=['model_dist', 'human_dist', 'corrected_dist']).to_csv(out_dir / 'cell_level_rectification.tsv', sep='\t', index=False)
    category_eval.to_csv(out_dir / 'category_rectification_eval.tsv', sep='\t', index=False)
    question_eval.to_csv(out_dir / 'question_rectification_eval.tsv', sep='\t', index=False)
    json.dump(summary, open(out_dir / 'summary.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=2)

    lines = []
    lines.append(f"Rectification summary for {Path(args.predictions).name}")
    for key in ['overall', 'calibration', 'evaluation']:
        s = summary[key]
        lines.append(
            f"- {key}: baseline_mean_w1={s['baseline_mean_w1']:.4f}, corrected_mean_w1={s['corrected_mean_w1']:.4f}, "
            f"baseline_top_hit={s['baseline_top_hit_rate']:.4f}, corrected_top_hit={s['corrected_top_hit_rate']:.4f}"
        )
    b = summary['evaluation_bootstrap']
    lines.append(
        f"- evaluation bootstrap delta_mean_w1: mean={b['delta_mean_w1_bootstrap_mean']:.4f}, "
        f"95% CI=[{b['delta_mean_w1_bootstrap_ci95_low']:.4f}, {b['delta_mean_w1_bootstrap_ci95_high']:.4f}]"
    )
    (out_dir / 'README.txt').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    main()
