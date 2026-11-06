"""Independent saved-output evaluation, without importing experiment runner."""
from pathlib import Path
import hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/flyvis_history_state_swap_20261106';SRC=ROOT/'runs/flyvis_offset_recovery2_20261104'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    p=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'));done=json.loads((OUT/'inference_completed.json').read_text());assert done['complete'] and not done['cuda_initialized'] and done['protocol_sha256']==sha(OUT/'protocol.json')
    assert sha(Path(__file__))==p['frozen_sources'][str(Path(__file__).relative_to(ROOT)).replace('\\','/')]
    mask=np.load(SRC/'geometry.npz')['mask'];rows=[]
    for c in p['cases']:
        q=OUT/'cases'/(c['id']+'.npz');m=json.loads(q.with_suffix('.json').read_text());assert sha(q)==m['sha256'] and m['protocol_sha256']==sha(OUT/'protocol.json')
        refpath=SRC/'stock/outputs'/(c['id']+'.npz');assert sha(refpath)==m['reference_sha256']==p['reference_sha256'][c['id']]
        ref=np.load(refpath);z=np.load(q);f=z['flow'];assert f.shape==(6,24,721,2) and np.isfinite(f).all()
        assert np.array_equal(z['suffix'],ref['hexes'][55:79]) and np.array_equal(z['last_history_input'],ref['hexes'][54])
        assert np.max(np.abs(f[0]-ref['flow'][0,1,55:79]))<=1e-5
        for a,b in [(0,2),(1,3),(0,4),(1,5)]:
            assert np.array_equal(f[a],f[b])
            assert all(h[a]==h[b] for h in m['step_state_sha256'])
        assert np.max(z['state_diffs'])==0 and m['donor_unchanged'] and m['parameters_unchanged'] and m['decoder_buffers_unchanged']
        y=f[:,:,mask,:].mean(axis=2);truth=np.zeros((24,2));truth[:,c['axis']]=-c['speed'];metrics={}
        for i,name in enumerate(p['arms']):
            good=y[i,:,c['axis']]<0
            hit=next((j for j in range(22) if bool(np.all(good[j:j+3]))),None)
            metrics[name]=dict(epe=float(np.sqrt(((y[i]-truth)**2).sum(-1)).mean()),latency=hit,capped_latency=24 if hit is None else hit,correct_sign_fraction=float(good.mean()))
        rows.append(dict(case=c,metrics=metrics,history_max_field_difference=float(np.max(np.abs(f[0]-f[1]))),history_mean_vector_difference=float(np.linalg.norm(y[0]-y[1],axis=-1).mean()),boundary_max_state_difference=float(np.max(np.abs(z['boundary_activity'][0]-z['boundary_activity'][1]))),sha256=m['sha256']))
    groups={};rng=np.random.default_rng(42);draw=rng.integers(0,4,(20000,4))
    for contrast in (1.,.25):
        rs=[r for r in rows if r['case']['contrast']==contrast];g={}
        for arm in ('moving','stationary'):
            ms=[r['metrics'][arm] for r in rs];g[arm]=dict(epe=float(np.mean([m['epe'] for m in ms])),capped_latency=float(np.mean([m['capped_latency'] for m in ms])),not_found=sum(m['latency'] is None for m in ms),correct_sign_fraction=float(np.mean([m['correct_sign_fraction'] for m in ms])))
        for metric in ('epe','capped_latency'):
            ds=np.array([np.mean([r['metrics']['stationary'][metric]-r['metrics']['moving'][metric] for r in rs if r['case']['scene']==s]) for s in (4,5,6,7)])
            ci=np.quantile(ds[draw].mean(1),[.025,.975]);g['stationary_minus_moving_'+metric]=dict(mean=float(ds.mean()),ci95=ci.tolist(),scene_differences=ds.tolist())
        g['history_mean_vector_difference']=float(np.mean([r['history_mean_vector_difference'] for r in rs]));groups[str(contrast)]=g
    report=dict(passed=True,cases=len(rows),protocol_sha256=sha(OUT/'protocol.json'),inference_sha256=sha(OUT/'inference_completed.json'),identity_and_full_swap_exact=True,history_effect_cases=sum(r['history_max_field_difference']>1e-5 for r in rows),groups=groups,rows=rows,limits=p['limitations'],audit_scope='Independently recomputed metrics and controls from saved arrays, checking per-step recurrent-state hashes recorded by runner; no independent full model reexecution.')
    (OUT/'independent_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');print(json.dumps({k:v for k,v in report.items() if k!='rows'},ensure_ascii=False))
if __name__=='__main__':main()
