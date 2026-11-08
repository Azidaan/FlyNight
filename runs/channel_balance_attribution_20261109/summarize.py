"""Matched score comparisons; no new feature or old-score computation."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import json,hashlib,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'runs/flyvis_repairs_20261106/unification_20261107'))
from system_memory_guard import guard_entrypoint
guard_entrypoint(timeout=600)
import numpy as np
from sklearn.metrics import roc_auc_score
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def save(name,v):
    with (OUT/name).open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n')
result=read(OUT/'results.json');rows=[read(OUT/f) for f in result['files']]
known=[r for r in rows if r['supported'] and r['kind']!='unknown'];assert len(known)==45
names=list(known[0]['variants']);matched=[r for r in known if all(r['variants'][n]['maximum'] is not None for n in names)]
pos=[r for r in matched if r['kind']=='own'];neg=[r for r in matched if r['kind']=='foreign'];videos=sorted({r['video'] for r in matched});vi={v:i for i,v in enumerate(videos)}
w=np.random.default_rng(42).multinomial(len(videos),np.full(len(videos),1/len(videos)),5000);wp=w[:,[vi[r['video']] for r in pos]];wn=w[:,[vi[r['video']] for r in neg]];den=wp.sum(1)*wn.sum(1);valid=den>0
scores={};boot={}
for name in names:
    scores[name]={}
    for field in ['baseline_max','maximum']:
        a=np.array([r['variants'][name][field] for r in pos]);b=np.array([r['variants'][name][field] for r in neg]);m=(a[:,None]>b).astype(float)+.5*(a[:,None]==b)
        au=float(m.mean());assert abs(au-roc_auc_score([r['kind']=='own' for r in matched],[r['variants'][name][field] for r in matched]))<1e-12
        scores[name][field]=au;boot[name,field]=np.einsum('bi,ij,bj->b',wp,m,wn)[valid]/den[valid]
    delta=boot[name,'maximum']-boot[name,'baseline_max'];scores[name]['delta']=scores[name]['maximum']-scores[name]['baseline_max'];scores[name]['ci95']=np.quantile(delta,[.025,.975]).tolist()
comparisons={}
for fly,controls in [('FlyVis',['center8','neighbor8','input8_temporal']),('C34',['R134_temporal','R134_spatial'])]:
    for other in controls:
        delta=boot[fly,'maximum']-boot[other,'maximum'];comparisons[fly+'-'+other]=dict(delta=scores[fly]['maximum']-scores[other]['maximum'],ci95=np.quantile(delta,[.025,.975]).tolist())
transfer=read(OUT/'transfer.json')['records'];ts={}
for name in names[:4]:
    subset=[r for r in transfer if r['representation']==name];assert len(subset)==7
    good=[r for r in subset if r['kind']=='own'];bad=[r for r in subset if r['kind']=='foreign'];ts[name]={}
    for field in ['baseline_max','maximum']:
        a=np.array([r[field] for r in good]);b=np.array([r[field] for r in bad]);ts[name][field]=float(((a[:,None]>b)+.5*(a[:,None]==b)).mean())
own=next(r for r in rows if r['old']==88 and r['new']==144 and r['video']=='00982f76-83a093ef');foreign=next(r for r in rows if r['old']==73 and r['new']==106 and r['video']=='00d7268f-fd4487be')
local={n:dict(own=own['variants'][n]['maximum'],foreign=foreign['variants'][n]['maximum'],margin=own['variants'][n]['maximum']-foreign['variants'][n]['maximum'],correct=own['variants'][n]['maximum']>foreign['variants'][n]['maximum']) for n in names}
local_gate=local['FlyVis']['correct'] and local['C34']['correct']
broad_gate=all(scores[n]['maximum']>=scores[n]['baseline_max'] for n in ['FlyVis','C34']) and ts['FlyVis']['maximum']>=ts['FlyVis']['baseline_max']
save('summary.json',dict(events=len(rows),supported=sum(r['supported'] for r in rows),conditional_known=45,matched_known=len(matched),own=len(pos),foreign=len(neg),videos=len(videos),unknown=sum(r['kind']=='unknown' for r in rows),scores=scores,matched_input_comparisons=comparisons,transfer7=ts,local_pairs=local,local_gate=bool(local_gate),broader_gate=bool(broad_gate),proceed=bool(local_gate and broad_gate),bootstrap=dict(seed=42,samples=5000,valid=int(valid.sum()),unit='video',scope='Descriptive fit20 intervals; no independent tracking claim or multiple-comparison-adjusted significance.')))
print('Local',local,'\nScores',scores,'\nTransfer',ts,'\nGates',local_gate,broad_gate)
