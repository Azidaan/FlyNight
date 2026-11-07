"""Парное сравнение сохранённых полей; контроль — прежние результаты по SHA."""
from pathlib import Path
import json,sys,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tracking.flyvis_latency_metrics import response_metrics
from scripts.audit_flyvis_y1_latency_fp64_20261104 import independent_metrics
OUT=ROOT/'runs/flyvis_tau_half_20261104';BASE=ROOT/'runs/flyvis_y1_latency_fp64_20261104'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    p=json.loads((OUT/'protocol.json').read_text());g=np.load(BASE/'geometry.npz');mask=g['mask']
    assert json.loads((OUT/'inference_completed.json').read_text())['complete']
    assert json.loads((OUT/'tau_intervention.json').read_text())['other_parameters_exact']
    rows=[]
    for case in p['cases']:
        new=OUT/'outputs'/(case['id']+'.npz');old=BASE/'outputs'/(case['id']+'.npz')
        nm=json.loads(new.with_suffix('.json').read_text());om=json.loads(old.with_suffix('.json').read_text())
        assert sha(new)==nm['sha256'] and sha(old)==om['sha256']
        assert nm['input_sha256']==om['input_sha256']
        A=np.load(new);B=np.load(old)
        assert np.max(np.abs(A['hexes']-B['hexes']))<=1e-5
        assert np.isfinite(A['flow']).all()
        for mode,idx in [('hold',0),('linear',1)]:
            for agg,j in [('mean',0),('last',1)]:
                a=A['flow'][idx,j][:,mask].mean(1);b=B['flow'][idx,j][:,mask].mean(1)
                x=response_metrics(a,case['axis'],case['speed'],case['event']);y=response_metrics(b,case['axis'],case['speed'],case['event'])
                for z,trace in [(x,a),(y,b)]:
                    v=independent_metrics(trace,case)
                    for key in v:
                        if isinstance(v[key],float):assert abs(v[key]-z[key])<1e-10
                        else:assert v[key]==z[key]
                rows.append(dict(**case,mode=mode,aggregation=agg,fast=x,stock=y))
    groups={}
    for ct in [1.,.25]:
        candidates=[r for r in rows if r['event']=='reverse' and r['contrast']==ct and r['mode']=='hold' and r['aggregation']=='last']
        assert len(candidates)==16
        valid=[r for r in candidates if all(r[x]['eligible'] and r[x]['latency_frames'] is not None for x in ['fast','stock'])]
        scene=[]
        for s in range(4):
            v=[r['stock']['latency_frames']-r['fast']['latency_frames'] for r in valid if r['scene']==s]
            scene.append(float(np.mean(v)) if len(v)==4 else None)
        ci=None
        if len(valid)==16:
            v=np.array(scene);bs=v[np.random.default_rng(42).integers(0,4,(10000,4))].mean(1);ci=np.quantile(bs,[.025,.975]).tolist()
        groups[str(ct)]=dict(valid=len(valid),total=16,scene_delta_frames=scene,ci95=ci,
            mean_reduction_frames=float(np.mean(scene)) if ci else None,
            nominal_epe_fast=float(np.mean([r['fast']['epe_nominal_px'] for r in candidates])),
            nominal_epe_stock=float(np.mean([r['stock']['epe_nominal_px'] for r in candidates])))
    a,b=groups['1.0'],groups['0.25']
    passed=bool(a['ci95'] and b['ci95'] and a['mean_reduction_frames']>=1 and a['ci95'][0]>0 and
       b['mean_reduction_frames']>=0 and a['nominal_epe_fast']<=a['nominal_epe_stock'] and b['nominal_epe_fast']<=b['nominal_epe_stock'])
    result=dict(complete=True,passed=True,scientific_gate=passed,groups=groups,rows=rows,
       limitation='Same four historical backgrounds, one model, artificial global shifts; no HOTA, no adaptive rule, no independent final test. FP64 CPU; full input and output hashes checked; network was not independently replayed.')
    (OUT/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},ensure_ascii=False))
if __name__=='__main__':main()
