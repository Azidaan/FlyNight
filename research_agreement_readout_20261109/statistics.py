"""Descriptive paired video intervals; no model selection by intervals."""
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf8'))
rows=read(OUT/'predictions.json')['records'];old=read(ROOT/'research_signal_scaling_20261109/predictions.json')['records'];assert len(rows)==len(old)==26
for a,b in zip(rows,old):assert all(a[k]==b[k] for k in ['video','frame','old_id','raw_id','label'])
videos=sorted({r['video'] for r in rows});groups=np.array([videos.index(r['video']) for r in rows]);own=np.array([r['label']=='own' for r in rows]);weights=np.random.default_rng(20261109).multinomial(len(videos),np.full(len(videos),1/len(videos)),size=4000)[:,groups];valid=(weights[:,own].sum(1)>0)&(weights[:,~own].sum(1)>0);w=weights[valid]
def ba(z):
    yes=np.array(z)>0;return .5*((w[:,own]*yes[own]).sum(1)/w[:,own].sum(1)+(w[:,~own]*(~yes[~own])).sum(1)/w[:,~own].sum(1))
comparisons={}
for arm in ['C34','R134_spatial','R134_temporal']:
    base=ba([r['scores'][arm+'__common'] for r in old]);a=ba([r['scores'][arm+'__agreement'] for r in rows]);v=ba([r['scores'][arm+'__variance_only'] for r in rows])
    for name,diff in [('agreement-baseline',a-base),('variance_only-baseline',v-base),('agreement-variance_only',a-v)]:comparisons[arm+'__'+name]=dict(ci95=np.quantile(diff,[.025,.975]).tolist(),bootstrap_mean=float(diff.mean()))
with (OUT/'uncertainty.json').open('x',encoding='utf8') as f:json.dump(dict(passed=True,comparisons=comparisons,draws=4000,valid_draws=int(valid.sum()),invalid_missing_class=int((~valid).sum()),seed=20261109,scope='Descriptive unadjusted95% intervals on reused development cohort, not an independent test or an adoption criterion.'),f,indent=2)
print('AGREEMENT_DESCRIPTIVE_INTERVALS_READY',flush=True)
