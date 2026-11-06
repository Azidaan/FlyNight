"""Independent numerical evaluation and partial-copy invariants on saved tensors."""
from pathlib import Path
import hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/flyvis_partial_state_swap_20261106';OLD=ROOT/'runs/flyvis_history_state_swap_20261106';SRC=ROOT/'runs/flyvis_offset_recovery2_20261104'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    p=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'));done=json.loads((OUT/'inference_completed.json').read_text());assert done['complete'] and done['cases']==32 and not done['cuda_initialized']
    assert done['protocol_sha256']==sha(OUT/'protocol.json') and sha(Path(__file__))==p['frozen_sources']['scripts/audit_flyvis_partial_state_swap_20261106.py']
    assert sha(OUT/'indices.npz')==done['indices_sha256'] and sha(OUT/'textures.npz')==done['textures_sha256'] and sha(SRC/'geometry.npz')==p['geometry_sha256']
    indices=np.load(OUT/'indices.npz');textures=np.load(OUT/'textures.npz');mask=np.load(SRC/'geometry.npz')['mask'];rows=[]
    sources=json.loads((SRC/'source_images.json').read_text())['sources']
    for k in textures.files:assert hashlib.sha256(textures[k].tobytes()).hexdigest()==sources[int(k)]['frame_sha256']
    for c in p['cases']:
        q=OUT/'cases'/(c['id']+'.npz');m=json.loads(q.with_suffix('.json').read_text());assert sha(q)==m['sha256'] and m['protocol_sha256']==sha(OUT/'protocol.json')
        oldpath=OLD/'cases'/(c['id']+'.npz');assert sha(oldpath)==m['state_reference_sha256']==p['reference_sha256'][c['id']]['state_sha256']
        z=np.load(q);old=np.load(oldpath);a,b=old['boundary_activity'];initial=z['initial_states'];f=z['flow']
        assert f.shape==(2,16,24,721,2) and np.isfinite(f).all() and np.isfinite(initial).all()
        for i,v in enumerate([a,b,a,b]):assert np.array_equal(initial[i],v)
        partition=np.concatenate([indices[g] for g in list(p['groups'])[:5]]);assert np.array_equal(np.sort(partition),np.arange(len(a)))
        assert np.array_equal(np.sort(indices['t4t5']),np.sort(np.concatenate([indices['t4'],indices['t5']])))
        for gi,g in enumerate(p['groups']):
            ix=indices[g];keep=np.ones(len(a),bool);keep[ix]=False
            erase,insert=initial[4+gi*2:6+gi*2]
            assert np.array_equal(erase[ix],b[ix]) and np.array_equal(erase[keep],a[keep])
            assert np.array_equal(insert[ix],a[ix]) and np.array_equal(insert[keep],b[keep])
        assert np.array_equal(f[:,0],f[:,2]) and np.array_equal(f[:,1],f[:,3])
        assert np.max(np.abs(f[0,0]-old['flow'][0]))<=1e-10 and np.max(np.abs(f[0,1]-old['flow'][1]))<=1e-10
        assert np.array_equal(z['reverse_hexes'],old['suffix']) and np.max(np.abs(z['continue_hexes'][0]-old['last_history_input']))<=1e-12
        assert all(m[k] for k in ['identity_states_exact','donors_unchanged','parameters_and_buffers_unchanged']) and m['eye_max_diff']<=1e-5
        image=textures[str(c['scene'])];height,width=image.shape;top=height-512 if c['axis']==1 and c['speed']==4 else (height-512)//2;left=(width-512)//2
        for frame in range(25):
            offset=c['speed']*(24+frame);y=top-(offset if c['axis']==1 else 0);x=left-(offset if c['axis']==0 else 0)
            raw=image[y:y+512,x:x+512].astype(np.float32)
            expected=np.rint(np.clip(127.5+c['contrast']*(raw-127.5),0,255)).astype(np.uint8)
            assert np.array_equal(expected,z['continue_frames'][frame])
        vectors=f[:,:,:,mask,:].mean(axis=3);metrics={}
        for si,scenario in enumerate(p['scenarios']):
            truth=np.zeros((24,2));truth[:,c['axis']]=c['speed']*(-1 if scenario=='reverse' else 1);arms={}
            for ai,name in enumerate(p['arms']):
                y=vectors[si,ai];errors=np.sqrt(np.sum((y-truth)**2,axis=-1));sign=y[:,c['axis']]*truth[:,c['axis']]>0
                hit=next((j for j in range(22) if np.all(sign[j:j+3])),None)
                arms[name]=dict(epe=float(errors.mean()),early_epe=float(errors[:6].mean()),late_epe=float(errors[-6:].mean()),latency=hit,capped_latency=24 if hit is None else hit,correct_sign_fraction=float(sign.mean()))
            metrics[scenario]=arms
        rows.append(dict(case=c,metrics=metrics,sha256=m['sha256']))
    groups={};pairs={};draw=np.random.default_rng(42).integers(0,4,(20000,4))
    for contrast in (1.,.25):
        cohort=[r for r in rows if r['case']['contrast']==contrast];gc={}
        for scenario in p['scenarios']:
            gc[scenario]={}
            for arm in p['arms']:
                ms=[r['metrics'][scenario][arm] for r in cohort]
                gc[scenario][arm]={k:float(np.mean([m[k] for m in ms])) for k in ('epe','early_epe','late_epe','capped_latency','correct_sign_fraction')}
                gc[scenario][arm]['not_found']=sum(m['latency'] is None for m in ms)
            for g in p['groups']:
                for direction,reference in [('erase','moving'),('insert','stationary')]:
                    for metric in ('capped_latency','epe','early_epe'):
                        ds=np.array([np.mean([r['metrics'][scenario][direction+'_'+g][metric]-r['metrics'][scenario][reference][metric] for r in cohort if r['case']['scene']==s]) for s in (4,5,6,7)])
                        samples=ds[draw].mean(axis=1);ordinary=np.quantile(samples,[.025,.975]);corrected=np.quantile(samples,[.05/12,1-.05/12])
                        pairs[f'{contrast}:{scenario}:{direction}_{g}:{metric}']=dict(mean=float(ds.mean()),ci95=ordinary.tolist(),ci_family6=corrected.tolist(),scene_differences=ds.tolist())
        groups[str(contrast)]=gc
    candidates={}
    for g in p['groups']:
        checks=dict(corrected_latency_reduction=pairs[f'1.0:reverse:erase_{g}:capped_latency']['ci_family6'][1]<0)
        for ct in ('1.0','0.25'):
            for scenario in p['scenarios']:
                current=groups[ct][scenario]['erase_'+g];base=groups[ct][scenario]['moving']
                for metric in ('epe','early_epe'):checks[f'{ct}:{scenario}:{metric}_not_worse']=current[metric]<=base[metric]+1e-12
                if scenario=='reverse':checks[f'{ct}:missing_not_more']=current['not_found']<=base['not_found']
                else:checks[f'{ct}:correct_sign_not_less']=current['correct_sign_fraction']>=base['correct_sign_fraction']-1e-12
        candidates[g]=dict(passed=all(checks.values()),checks=checks)
    result=dict(passed=True,cases=32,protocol_sha256=sha(OUT/'protocol.json'),inference_sha256=sha(OUT/'inference_completed.json'),groups=groups,paired=pairs,candidates=candidates,primary_met=any(c['passed'] for c in candidates.values()),rows=rows,limits=p['limitations'],audit_scope='Independent metric calculation and exact indexed-copy validation from saved full initial tensors/fields; exact original-frame reconstruction; runner eye comparison shares published BoxEye source. No second full network simulation.')
    (OUT/'independent_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(passed=True,cases=32,primary_met=result['primary_met'],candidates={g:dict(passed=v['passed'],failed=[k for k,b in v['checks'].items() if not b]) for g,v in candidates.items()},ordinary_reverse={g:groups['1.0']['reverse'][g] for g in ['moving','stationary']+['erase_'+g for g in p['groups']]}),ensure_ascii=False))
if __name__=='__main__':main()
