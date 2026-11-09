"""Descriptive paired video-cluster intervals, not an independent test."""
import os,sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=Path(__file__).resolve().parent;os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
def read(p):return json.loads(p.read_text(encoding='utf8'))
new=read(OUT/'predictions.json')['records'];old=read(ROOT/'research_domain_matched_reader_20261109/predictions.json')['records'];assert len(new)==len(old)==26
for a,b in zip(new,old):assert all(a[k]==b[k] for k in ['video','frame','old_id','raw_id','label'])
videos=sorted({r['video'] for r in old});groups=np.array([videos.index(r['video']) for r in old]);own=np.array([r['label']=='own' for r in old]);weights=np.random.default_rng(20261109).multinomial(len(videos),np.full(len(videos),1/len(videos)),size=4000)[:,groups];valid=(weights[:,own].sum(1)>0)&(weights[:,~own].sum(1)>0);w=weights[valid]
def ba(z):
    yes=np.asarray(z)>0;return .5*((w[:,own]*yes[own]).sum(1)/w[:,own].sum(1)+(w[:,~own]*(~yes[~own])).sum(1)/w[:,~own].sum(1))
comparisons={}
for arm in ['C34','R134_spatial','R134_temporal']:
    baseline=ba([r['scores'][arm] for r in old]);values={kind:ba([r['scores'][arm+'__'+kind] for r in new]) for kind in ['unit','common']}
    for a,b,diff in [('unit','raw',values['unit']-baseline),('common','raw',values['common']-baseline),('unit','common',values['unit']-values['common'])]:comparisons[arm+'__'+a+'-'+b]=dict(ci95=np.quantile(diff,[.025,.975]).tolist(),bootstrap_mean=float(diff.mean()))
with (OUT/'uncertainty.json').open('x',encoding='utf8') as f:json.dump(dict(passed=True,comparisons=comparisons,draws=4000,valid_draws=int(valid.sum()),invalid_missing_class=int((~valid).sum()),seed=20261109,cluster='video',scope='Unadjusted descriptive95%intervals on reused development cohort; not confirmed benefit, model selection or an independent test.',goal_achieved=False),f,indent=2)
print('SCALE_DESCRIPTIVE_INTERVALS_READY',int(valid.sum()),flush=True)
