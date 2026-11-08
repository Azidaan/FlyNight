import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'runs/flyvis_repairs_20261106/unification_20261107'))
from system_memory_guard import guard_entrypoint
guard_entrypoint(timeout=600)
import numpy as np
from sklearn.metrics import roc_auc_score
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def save(n,v):
    with (OUT/n).open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n')
rows=[read(OUT/'events'/n) for n in read(OUT/'results.json')['files']];known=[r for r in rows if r['kind']!='unknown'];names=list(known[0]['variants']);fields=['baseline_global','baseline_balanced','global_max','balanced_max']
matched=[r for r in known if all(r['variants'][n][f] is not None for n in names for f in fields)]
pos=[r for r in matched if r['kind']=='own'];neg=[r for r in matched if r['kind']=='foreign'];videos=sorted({r['video'] for r in matched});vi={v:i for i,v in enumerate(videos)}
w=np.random.default_rng(42).multinomial(len(videos),np.full(len(videos),1/len(videos)),5000);wp=w[:,[vi[r['video']] for r in pos]];wn=w[:,[vi[r['video']] for r in neg]];den=wp.sum(1)*wn.sum(1);valid=den>0
scores={};boot={};comparisons={}
for n in names:
    scores[n]={}
    for f in fields:
        a=np.array([r['variants'][n][f] for r in pos]);b=np.array([r['variants'][n][f] for r in neg]);matrix=(a[:,None]>b).astype(float)+.5*(a[:,None]==b);score=float(matrix.mean())
        assert abs(score-roc_auc_score([r['kind']=='own' for r in matched],[r['variants'][n][f] for r in matched]))<1e-12
        scores[n][f]=score;boot[n,f]=np.einsum('bi,ij,bj->b',wp,matrix,wn)[valid]/den[valid]
    comparisons[n]={}
    for newer,older in [('global_max','baseline_global'),('balanced_max','baseline_balanced'),('balanced_max','baseline_global')]:
        delta=boot[n,newer]-boot[n,older];comparisons[n][newer+'-'+older]=dict(delta=scores[n][newer]-scores[n][older],ci95=np.quantile(delta,[.025,.975]).tolist())
local={};traces=read(OUT/'pair_traces.json')['records'];own=next(r for r in traces if r['old']==88);foreign=next(r for r in traces if r['old']==73)
for n in names:local[n]={f:dict(own=own['variants'][n][f],foreign=foreign['variants'][n][f],correct=own['variants'][n][f]>foreign['variants'][n][f]) for f in fields}
coverage={f:sum(all(r['variants'][n][f] is not None for n in names) for r in known) for f in fields}
signal={}
for fly,controls in [('FlyVis',['center8','neighbor8','input8_temporal']),('C34',['R134_temporal','R134_spatial'])]:
    for control in controls:
        d=boot[fly,'balanced_max']-boot[control,'balanced_max'];signal[fly+'-'+control]=dict(delta=scores[fly]['balanced_max']-scores[control]['balanced_max'],ci95=np.quantile(d,[.025,.975]).tolist())
save('summary.json',dict(events=len(rows),known=len(known),matched_known=len(matched),own=len(pos),foreign=len(neg),videos=len(videos),changed=sum(r['anchor']['changed'] for r in rows),reused=sum(r['reused'] for r in rows),coverage_known=coverage,excluded=[dict(video=r['video'],frame=r['frame'],old=r['old'],new=r['new'],old_cells=len(r['baseline_cells']),new_cells=len(r['geometry']['cells'])) for r in known if r not in matched],scores=scores,comparisons=comparisons,local=local,primary_local_gate=bool(all(local[n]['global_max']['correct'] for n in ['FlyVis','C34'])),joint_local_gate=bool(all(local[n]['balanced_max']['correct'] for n in ['FlyVis','C34'])),joint_signal_comparisons=signal,bootstrap=dict(seed=42,samples=5000,unit='video',scope='Descriptive matched fit20 diagnostic, no new independent-tracker claim.')))
print('Coverage',coverage,'\nScores',scores,'\nLocal',local)
