"""Independent audit of the finished parent pilot; no new tracking or inference."""
import os,sys,gzip
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from collections import Counter
from research_identity_assignment_run_20261108.common import *
from research_singleton_verify_20261108.common import guard


def main(obs):
    import numpy as np
    from ultralytics.trackers.utils import matching
    from datasets.bdd100k_loader import CLASS_NAMES
    from evaluation.hota_eval import VALID
    from research_singleton_free_20261108.full_audit import independent_returns
    verify_phase('parent_full_audit_phase.json');full=RUN/'full_precision';p=read(full/'protocol.json');report=read(full/'comparison_report.json')
    assert read(full/'analyses_complete.json')['passed'];checks=Counter();metrics={a:[] for a in p['parameters']};components={a:{k:[] for k in ['TP','FP','FN','AA']} for a in p['parameters']}
    totals={a:Counter() for a in p['parameters']}
    for number,(video,entry) in enumerate(p['videos'].items()):
        for n,h in entry['sources'].items():assert sha(ROOT/n)==h
        for n,h in read(full/'completed'/(video+'.json'))['artifacts'].items():assert sha(ROOT/n)==h
        anchors=read(full/'runtime_inputs'/(video+'.json'))['anchors'];source=ROOT/f'data/bdd100k/labels/box_track_20/val/{video}.json'
        annotation={int(f['frameIndex']):f for f in read(source)};truth={}
        for frame,clock in anchors.items():
            f=annotation[clock['frameIndex']];assert f['name']==clock['name'];objects=[]
            for obj in f.get('labels') or []:
                if obj['category'] not in CLASS_NAMES.values() or (obj.get('attributes') or {}).get('crowd',False):continue
                objects.append((str(obj['id']),obj['category'],[obj['box2d'][k] for k in ['x1','y1','x2','y2']]))
            truth[int(frame)]=objects
        with np.load(ROOT/entry['detections'],allow_pickle=False) as z:detections={n:z[n] for n in z.files}
        cache={};events_seen=set()
        for arm,param in p['parameters'].items():
            actual=p['aliases'].get(arm,arm);rawpath=ROOT/entry['raw_array'] if actual=='B0' else full/'rollouts'/actual/(video+'.npz')
            with np.load(rawpath,allow_pickle=False) as z:raw=z['outputs'];assert int(z['frame_count'])==entry['frames']
            signature=sha(rawpath)
            if signature not in cache:cache[signature]=independent_returns(raw,truth);checks['independent_return_trajectories']+=1
            expected=cache[signature];saved=read(full/'additional'/arm/(video+'.json'))
            assert saved['gt_sha256']==sha(source)
            for key in ['totals','events','duplicate_extras']:assert saved[key]==expected[key],(video,arm,key)
            totals[arm].update(expected['totals']);checks['return_reports']+=1
            if actual!='B0' and actual not in events_seen:
                events_seen.add(actual);receipt=read(full/'rollouts'/actual/(video+'.receipt.json'));assert receipt['passed']
                for n,h in {**receipt['artifacts'],**receipt['feature_files']}.items():assert sha(ROOT/n)==h
                with gzip.open(full/'rollouts'/actual/(video+'.events.json.gz'),'rt',encoding='utf-8') as f:events=json.load(f)
                for e in events:
                    C=np.array(e['C'],np.float32);D=C.copy();plan=e['plan']
                    for i,j in plan['votes']:D[i,j]-=param['alpha']
                    for j,col in plan['columns'].items():
                        if not col['geometry_valid'] or len(col['rows'])<2:continue
                        rows=np.array(col['rows'],int);values=col['scores'].get(actual)
                        if actual=='G_SHIFT':D[rows,int(j)]-=param['beta'];continue
                        if values is None or any(v is None for v in values):continue
                        values=np.clip(np.array(values,float),-1,1);D[rows,int(j)]-=param['beta']*(values-values.mean())
                    assert np.array_equal(D,np.array(e['corrected'],np.float32))
                    wanted=np.asarray(matching.linear_assignment(D,thresh=.8)[0],int).reshape(-1,2)
                    native=np.asarray(matching.linear_assignment(C,thresh=.8)[0],int).reshape(-1,2)
                    assert wanted.tolist()==e['actual_first'] and native.tolist()==e['native_first']
                    high=np.flatnonzero((detections['frame_idx']==e['frame'])&(detections['conf']>=.25))
                    for i,j in wanted:
                        tid=str(e['rows'][i]['tid'])
                        if tid in e['assigned']:assert e['assigned'][tid]==int(high[j])
                    checks['independent_cost_matrices']+=1
            m=read(full/'metrics'/arm/(video+'.json'));assert m['passed'];metrics[arm].append(m['metrics'])
            assert m['output_sha256']==sha(full/'outputs'/arm/(video+'.json'))
            details=full/'metrics'/arm/(video+'.details.json.gz');assert sha(details)==m['details_sha256']
            with gzip.open(details,'rt',encoding='utf-8') as f:d=json.load(f)[video]
            for key in ['TP','FP','FN']:
                value=np.sum([d[c]['HOTA']['HOTA_'+key] for c in VALID if c in d],axis=0)
                assert np.allclose(value,m['components'][key],rtol=0,atol=1e-10);components[arm][key].append(value)
            value=np.sum([np.array(d[c]['HOTA']['HOTA_TP'])*d[c]['HOTA']['AssA'] for c in VALID if c in d],axis=0)
            assert np.allclose(value,m['components']['AA'],rtol=0,atol=1e-10);components[arm]['AA'].append(value)
        status('independent_parent_full_audit',complete=number+1,total=len(p['videos']),checks=dict(checks),guard=obs.read())
    n=len(p['videos']);indices=np.random.default_rng(42).integers(0,n,(20000,n));counts=np.array([np.bincount(v,minlength=n) for v in indices])
    with np.load(full/'bootstrap_counts.npz',allow_pickle=False) as z:assert np.array_equal(counts,z['counts'])
    samples={};points={};weights=np.vstack([np.ones((1,n)),counts])
    for arm,c in components.items():
        c={k:np.array(v) for k,v in c.items()};tp=weights@c['TP'];aa=weights@c['AA'];den=tp+weights@c['FP']+weights@c['FN']
        values=100*np.mean(np.sqrt((tp/np.maximum(1,den))*(aa/np.maximum(1,tp))),axis=1)
        points[arm]=float(values[0]);samples[arm]=values[1:];assert abs(points[arm]-report['pooled'][arm]['HOTA'])<1e-9
        assert dict(totals[arm])==read(full/'additional_report.json')['arms'][arm]['totals']
    accepted={}
    for key,value in report['comparisons'].items():
        a,b=key.split('-');d=samples[a]-samples[b]
        assert abs(points[a]-points[b]-value['delta_pooled_HOTA'])<1e-9
        assert np.allclose(np.percentile(d,[2.5,97.5]),value['ci95'],rtol=0,atol=1e-9)
        ci=np.percentile(d,[.3125,99.6875]);assert np.allclose(ci,value['ci99_375'],rtol=0,atol=1e-9)
        harm=dict(AssA_mean=float(np.mean([m['AssA'] for m in metrics[a]])-np.mean([m['AssA'] for m in metrics[b]])),
            IDF1_mean=float(np.mean([m['IDF1'] for m in metrics[a]])-np.mean([m['IDF1'] for m in metrics[b]])),
            IDSw=sum(m['IDSw'] for m in metrics[a])-sum(m['IDSw'] for m in metrics[b]),
            FP=sum(m['CLR_FP'] for m in metrics[a])-sum(m['CLR_FP'] for m in metrics[b]),FN=sum(m['CLR_FN'] for m in metrics[a])-sum(m['CLR_FN'] for m in metrics[b]))
        assert all(abs(harm[k]-value['harm'][k])<1e-9 for k in harm)
        safe=harm['AssA_mean']>=0 and harm['IDF1_mean']>=0 and all(harm[k]<=0 for k in ['IDSw','FP','FN'])
        assert safe==value['harm_passed'];accepted[key]=bool(ci[0]>0 and safe);checks['independent_bootstraps']+=1
    assert report['accepted_geometry']==accepted['G-B0']
    assert report['accepted_flyvis_system_and_increment']==all(v for k,v in accepted.items() if k.startswith('G_FLYVIS-'))
    verify_phase('parent_full_audit_phase.json');write(full/'independent_audit.json',dict(passed=True,checks=dict(checks),accepted=accepted,
        statistics_sha256=sha(full/'comparison_report.json'),additional_sha256=sha(full/'additional_report.json'),
        scope='Saved-output independent audit only: costs, global first assignment, raw bindings, GT identities/returns, class components and all8paired-video intervals. No repeated scientific run.'),new=True)


if __name__=='__main__':main(guard())
