"""Analyze only the complete fixed six-job functional pilot, never partial efficacy."""
from __future__ import annotations
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
OUT=ROOT/'runs/z13_distribution_pilot_20261015'
SRC=ROOT/'runs/z13_spatial_v2_20261013'

def read(p): return json.loads(p.read_text(encoding='utf-8'))

def main():
    from sklearn.metrics import average_precision_score
    from scripts.z13_train import success
    from tracking.z13_checkpoint import atomic_json
    p=read(OUT/'protocol.json')
    assert all((OUT/a/l/'completed.json').exists() for a in p['arms'] for l in p['losses']), 'All six jobs required'
    assert not (OUT/'report.json').exists(), 'Do not overwrite results'
    for name,h in p['source_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h,name
    eg=[]
    for v in p['evaluation']:
        with np.load(ROOT/'runs/z13_data'/f'{v}.npz') as z: eg.extend([v]*(len(z['ep_offset'])-1))
    eg=np.array(eg)
    report={'scope':'Functional train26 diagnostic, no tracker action or HOTA', 'arms':{},'comparisons':{},'source_sha256':{}}
    scores={}; errors={}; group=None; expected_keys=None
    for arm in p['arms']:
        for loss in ['original']+p['losses']:
            file=SRC/arm/'full/predictions.json' if loss=='original' else OUT/arm/loss/'predictions.json'
            rows=read(file); report['source_sha256'][str(file.relative_to(ROOT))]=hashlib.sha256(file.read_bytes()).hexdigest()
            keys=[(r['label'],r['episode'],r['search_position'],r['kind']) for r in rows]
            if expected_keys is None:
                expected_keys=keys; group=eg[[r['episode'] for r in rows]]
            else: assert keys==expected_keys
            known=np.array([r['kind']>=0 for r in rows]); pos=np.array([r['kind']==1 for r in rows]); neg=np.array([r['kind']==0 for r in rows])
            ok=np.array([success(r) for r in rows]); pres=np.array([r['p_present'] for r in rows])
            relative=np.array([r['err']/r['scale'] for r in rows if r['kind']==1])
            label=arm+'_'+loss
            scores[label]=ok[pos]; errors[label]=relative
            vids=np.unique(group[pos])
            medians=[np.median(relative[group[pos]==v]) for v in vids]
            report['arms'][label]=dict(rows=len(rows),known=int(known.sum()),positive=int(pos.sum()),unknown=int((~known).sum()),
                useful_count=int(ok[pos].sum()),useful_fraction=float(ok[pos].mean()),
                median_relative_center_error=float(np.median(relative)),mean_video_median_relative_error=float(np.mean(medians)),
                presence_AP=float(average_precision_score(pos[known],pres[known])),
                mean_presence_on_known_absent=float(pres[neg].mean()),videos_with_positives=len(vids))
    group=group[pos]; vids=np.unique(group); idx={v:np.flatnonzero(group==v) for v in vids}
    comparisons=[(a+'_spatial',a+'_hard') for a in 'BCD']+[('D_spatial','B_spatial'),('D_spatial','C_spatial'),('D_spatial','D_original')]
    rng=np.random.default_rng(20261015); boots={a+'-'+b:[] for a,b in comparisons}
    for _ in range(2000):
        sampled=np.concatenate([idx[v] for v in rng.choice(vids,len(vids),replace=True)])
        for a,b in comparisons: boots[a+'-'+b].append(float(scores[a][sampled].mean()-scores[b][sampled].mean()))
    for a,b in comparisons:
        report['comparisons'][a+'-'+b]=dict(useful_fraction_delta=float(scores[a].mean()-scores[b].mean()),
            ci95=np.quantile(boots[a+'-'+b],[.025,.975]).tolist(),
            mean_video_median_error_delta=report['arms'][a]['mean_video_median_relative_error']-report['arms'][b]['mean_video_median_relative_error'])
    diff=report['comparisons']['D_spatial-D_hard']
    report['criteria']=[diff['ci95'][0]>0,diff['mean_video_median_error_delta']<=0,
        report['comparisons']['D_spatial-B_spatial']['ci95'][0]>0]
    report['functional_gate_passed']=all(report['criteria'][:2])
    report['all_gates_passed']=all(report['criteria'])
    report['finished']=datetime.now(timezone(timedelta(hours=3))).isoformat()
    report['limitations']=p['caveats']+['Success is a geometric proposal criterion, not an applied correction or association outcome',
        'No threshold selected or quality scorer retrained after changing localization',
        'One seed and one fixed smoothing width; pilot does not prove a causal explanation for all failures']
    atomic_json(OUT/'report.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
