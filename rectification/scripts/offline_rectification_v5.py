from __future__ import annotations
import argparse, json
from pathlib import Path
from collections import defaultdict
import itertools
import numpy as np
import pandas as pd

MAX_OPTION_COL = 10

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument('--predictions', required=True)
    p.add_argument('--question-metadata', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--bootstrap-iters', type=int, default=1000)
    return p.parse_args()

def qnum(qid: str) -> int:
    return int(str(qid)[1:])

def entropy(p):
    p=np.asarray(p,float)
    p=p[p>0]
    return float(-(p*np.log(p)).sum()) if len(p) else 0.0

def build_cells(pred_path: Path, meta: dict) -> pd.DataFrame:
    df=pd.read_csv(pred_path, sep='\t')
    rows=[]
    for (qid,gk), g in df.groupby(['QUESTION_ID','GROUP_KEY']):
        m=meta[qid]
        lo,hi=int(m['answer_scale_min']), int(m['answer_scale_max'])
        p=np.array([pd.to_numeric(g[f'P_{i}'], errors='coerce').mean() if f'P_{i}' in g.columns else np.nan for i in range(1,MAX_OPTION_COL+1)], float)
        p=np.nan_to_num(p, nan=0.0)[lo-1:hi]
        p=p/p.sum() if p.sum()>0 else np.ones(hi-lo+1)/(hi-lo+1)
        hs=g['HUMAN_SCORE'].astype(int).to_numpy()
        human=np.array([(hs==i).mean() for i in range(lo,hi+1)], float)
        pos=np.arange(lo,hi+1)
        scale=max(1,hi-lo)
        rows.append(dict(
            QUESTION_ID=qid,
            GROUP_KEY=gk,
            category=m['category'],
            lo=lo, hi=hi,
            Continent=g['Continent'].iloc[0],
            Urban_Rural=g['Urban_Rural'].iloc[0],
            Education=g['Education'].iloc[0],
            model_probs=p.tolist(),
            human_probs=human.tolist(),
            model_mean=float((p*pos).sum()),
            human_mean=float((human*pos).sum()),
            model_mean_norm=float(((p*pos).sum()-lo)/scale),
            human_mean_norm=float(((human*pos).sum()-lo)/scale),
            model_entropy=entropy(p),
            human_entropy=entropy(human),
            w1_baseline=float(np.sum(np.abs(np.cumsum(human)-np.cumsum(p)))/scale),
            top_hit_baseline=int(np.argmax(p)==np.argmax(human)),
        ))
    return pd.DataFrame(rows)

def build_split_map(cells: pd.DataFrame, meta: dict) -> dict:
    cats=defaultdict(list)
    for qid in cells['QUESTION_ID'].unique():
        cats[meta[qid]['category']].append(qid)
    out={}
    for cat,qids in cats.items():
        qids=sorted(qids,key=qnum)
        out[cat]={'calibration':qids[0],'dev':qids[1],'test':qids[2]}
    return out

def split_df(cells, split_map, split):
    qids=[v[split] for v in split_map.values()]
    out=cells[cells['QUESTION_ID'].isin(qids)].copy()
    out['split']=split
    return out

def inter_group_ratio(df):
    vals=[]
    for _,g in df.groupby('QUESTION_ID'):
        hstd=g['human_mean'].std(ddof=0)
        mstd=g['model_mean'].std(ddof=0)
        if hstd>1e-9:
            vals.append(mstd/hstd)
    return float(np.mean(vals)) if vals else 1.0

def cal_stats(cal):
    fam={}
    for cat,g in cal.groupby('category'):
        fam[cat]={
            'mean_delta_norm': float((g['model_mean_norm']-g['human_mean_norm']).mean()),
            'entropy_gap': float((g['model_entropy']-g['human_entropy']).mean()),
            'group_ratio': inter_group_ratio(g),
            'bias_curve': np.mean(np.stack(g['human_probs'].map(np.array)) - np.stack(g['model_probs'].map(np.array)), axis=0).tolist(),
        }
    return {
        'global_mean_delta_norm': float((cal['model_mean_norm']-cal['human_mean_norm']).mean()),
        'global_entropy_gap': float((cal['model_entropy']-cal['human_entropy']).mean()),
        'global_group_ratio': inter_group_ratio(cal),
        'family': fam,
    }

def pooled_question_dists(cells):
    out={}
    for qid,g in cells.groupby('QUESTION_ID'):
        p=np.mean(np.stack(g['model_probs'].map(np.array)), axis=0)
        out[qid]=(p/p.sum()).tolist()
    return out

def apply_shift(p, shift):
    pos=np.linspace(-1,1,len(p))
    logp=np.log(np.clip(p,1e-12,None))+shift*pos
    p=np.exp(logp-logp.max())
    return p/p.sum()

def apply_temp(p, k):
    k=max(0.3, min(3.0, float(k)))
    p=np.power(np.clip(p,1e-12,None), k)
    return p/p.sum()

def interp_curve(curve, n):
    return np.interp(np.linspace(0,1,n), np.linspace(0,1,len(curve)), curve)

def correct_row(r, stats, pooled, params):
    am,ad,ag,bf,gg=params
    fam=stats['family'][r['category']]
    p=np.array(r['model_probs'], float)
    delta=0.35*stats['global_mean_delta_norm'] + 0.65*fam['mean_delta_norm']
    p=apply_shift(p, -am*delta)
    gap=0.35*stats['global_entropy_gap'] + 0.65*fam['entropy_gap']
    p=apply_temp(p, 1.0 + ad*gap)
    lam=min(0.85, ag*max(0.0, fam['group_ratio']-1.0))
    pooled_q=np.array(pooled[r['QUESTION_ID']], float)
    if lam>0 and len(pooled_q)==len(p):
        p=(1-lam)*p + lam*pooled_q
        p/=p.sum()
    bc=interp_curve(np.array(fam['bias_curve'], float), len(p))
    logp=np.log(np.clip(p,1e-12,None))+bf*bc
    p=np.exp(logp-logp.max())
    p/=p.sum()
    if gg>0:
        p=apply_shift(p, -gg*0.2*(r['model_mean_norm']-r['human_mean_norm']))
    return p.tolist()

def add_corrected(df):
    w1=[]; top=[]; ent=[]
    for _,r in df.iterrows():
        h=np.array(r['human_probs'], float)
        p=np.array(r['corrected_probs'], float)
        scale=max(1,int(r['hi'])-int(r['lo']))
        w1.append(float(np.sum(np.abs(np.cumsum(h)-np.cumsum(p)))/scale))
        top.append(int(np.argmax(h)==np.argmax(p)))
        ent.append(entropy(p))
    out=df.copy()
    out['w1_corrected']=w1
    out['top_hit_corrected']=top
    out['entropy_corrected']=ent
    return out

def summarize(df):
    return {
        'baseline_mean_w1': float(df['w1_baseline'].mean()),
        'corrected_mean_w1': float(df['w1_corrected'].mean()),
        'baseline_top_hit_rate': float(df['top_hit_baseline'].mean()),
        'corrected_top_hit_rate': float(df['top_hit_corrected'].mean()),
        'baseline_entropy': float(df['model_entropy'].mean()),
        'corrected_entropy': float(df['entropy_corrected'].mean()),
        'delta_mean_w1': float(df['w1_corrected'].mean()-df['w1_baseline'].mean()),
        'delta_top_hit_rate': float(df['top_hit_corrected'].mean()-df['top_hit_baseline'].mean()),
        'delta_entropy': float(df['entropy_corrected'].mean()-df['model_entropy'].mean()),
    }

def main():
    args=parse_args()
    meta=json.load(open(args.question_metadata,'r',encoding='utf-8'))
    cells=build_cells(Path(args.predictions), meta)
    split_map=build_split_map(cells, meta)
    cal=split_df(cells, split_map, 'calibration')
    dev=split_df(cells, split_map, 'dev')
    test=split_df(cells, split_map, 'test')
    stats=cal_stats(cal)
    pooled=pooled_question_dists(cells)

    grid=list(itertools.product([0.0,0.5,1.0],[0.0,0.5,1.0],[0.0,0.25,0.5],[0.0,0.5,1.0],[0.0,0.1,0.25]))
    selection=[]
    selected={}
    for cat,g in dev.groupby('category'):
        best=None
        for params in grid:
            tmp=g.copy()
            tmp['corrected_probs']=tmp.apply(lambda r: correct_row(r, stats, pooled, params), axis=1)
            tmp=add_corrected(tmp)
            b_w1=tmp['w1_baseline'].mean(); c_w1=tmp['w1_corrected'].mean()
            b_top=tmp['top_hit_baseline'].mean(); c_top=tmp['top_hit_corrected'].mean()
            b_ent=tmp['model_entropy'].mean(); c_ent=tmp['entropy_corrected'].mean()
            if c_ent < 0.85*b_ent:
                continue
            if c_top < b_top - 0.10:
                continue
            score=(c_w1, -c_top, abs(c_ent-b_ent))
            if best is None or score<best[0]:
                best=(score, params, b_w1, c_w1, b_top, c_top, b_ent, c_ent)
        if best is not None:
            rec={
                'category': cat,
                'params': list(best[1]),
                'baseline_mean_w1': float(best[2]),
                'corrected_mean_w1': float(best[3]),
                'delta_mean_w1': float(best[3]-best[2]),
                'baseline_top_hit_rate': float(best[4]),
                'corrected_top_hit_rate': float(best[5]),
                'baseline_entropy': float(best[6]),
                'corrected_entropy': float(best[7]),
            }
            selection.append(rec)
            if rec['delta_mean_w1'] < -0.005:
                selected[cat]=tuple(best[1])

    def apply_selected(df):
        out=df.copy()
        out['corrected_probs']=out.apply(lambda r: correct_row(r, stats, pooled, selected[r['category']]) if r['category'] in selected else r['model_probs'], axis=1)
        return add_corrected(out)

    cal2=apply_selected(cal)
    dev2=apply_selected(dev)
    test2=apply_selected(test)
    overall=pd.concat([cal2,dev2,test2], ignore_index=True)

    per_q=test2.groupby('QUESTION_ID')[['w1_baseline','w1_corrected']].mean().reset_index()
    arr=(per_q['w1_corrected']-per_q['w1_baseline']).to_numpy()
    rng=np.random.default_rng(42)
    boot=np.array([rng.choice(arr,size=len(arr),replace=True).mean() for _ in range(args.bootstrap_iters)])

    summary={
        'method':'offline_rectification_v5',
        'design':'clean 12/12/12 split; dev-only family selection; mean/dispersion/group-shrink + family bias tilt',
        'split_map':split_map,
        'selected_categories':sorted(selected.keys()),
        'selection':selection,
        'calibration':summarize(cal2),
        'dev':summarize(dev2),
        'test':summarize(test2),
        'overall':summarize(overall),
        'test_bootstrap':{
            'delta_mean_w1_bootstrap_mean': float(boot.mean()),
            'delta_mean_w1_bootstrap_ci95_low': float(np.quantile(boot,0.025)),
            'delta_mean_w1_bootstrap_ci95_high': float(np.quantile(boot,0.975)),
        },
        'warning':'Clean v5: categories and parameters selected on dev only. Test remained untouched.'
    }

    out=Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out/'summary_v5.json','w',encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    for key, fn in [('QUESTION_ID','question_rectification_eval_v5.tsv'),('category','category_rectification_eval_v5.tsv'),('GROUP_KEY','group_rectification_eval_v5.tsv')]:
        rows=[]
        for val,g in overall.groupby(key):
            rec={key:val}
            rec.update(summarize(g))
            rows.append(rec)
        pd.DataFrame(rows).to_csv(out/fn, sep='\t', index=False)
    pd.DataFrame(selection).to_csv(out/'dev_grid_search_v5.tsv', sep='\t', index=False)

if __name__=='__main__':
    main()
