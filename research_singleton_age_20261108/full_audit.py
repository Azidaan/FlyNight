"""Independently check the new AGE trajectories; reuse the audited original study."""
import os,sys,gzip
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from collections import Counter
from research_singleton_age_20261108.common import *


def main(obs):
    import numpy as np
    from ultralytics.trackers.utils import matching
    from tracking.z13_readout import HexGeometry
    from tracking.z13_region_assignment import regions
    from tracking.spatial_identity_assignment import geometric_match
    from research_singleton_verify_full_audit_20261108.audit import score_from_saved
    from research_singleton_free_20261108.full_audit import independent_returns
    from datasets.bdd100k_loader import CLASS_NAMES
    from evaluation.hota_eval import VALID
    phase('full_phase.json');assert read(OWN_FULL/'independent_audit.json')['passed'];p=read(FULL/'protocol.json');report=read(FULL/'comparison_report.json');cap=read(RUN/'selection.json')['cap'];checks=Counter();totals=Counter();metrics=[]
    g=read(ROOT/'runs/z13_local_peak_20261021/geometry.json');geom=HexGeometry(np.array(g['points'],np.float32),g['eye_size']);components={k:[] for k in ['TP','FP','FN','AA']}
    for number,(video,entry) in enumerate(p['videos'].items()):
        receipt=read(FULL/'rollouts/AGE'/(video+'.receipt.json'));assert not receipt['core_imported'] and not receipt['feature_files']
        for name,h in {**entry['sources'],**receipt['artifacts'],**read(FULL/'completed'/(video+'.json'))['artifacts']}.items():assert sha(ROOT/name)==h
        with np.load(ROOT/entry['detections'],allow_pickle=False) as z:raw={k:z[k] for k in z.files}
        with gzip.open(FULL/'rollouts/AGE'/(video+'.events.json.gz'),'rt',encoding='utf-8') as f:events=json.load(f)
        for e in events:
            C=np.array(e['C'],dtype=e['C_dtype']);rows=e['rows'];native,rr,cc=matching.linear_assignment(C,thresh=.8);native=np.asarray(native,int).reshape(-1,2);assert native.tolist()==e['native_first']
            frame=e['frame'];ids=np.flatnonzero((raw['frame_idx']==frame)&(raw['conf']>=.25));boxes=raw['xyxy'][ids];classes=raw['cls'][ids].astype(int);confidence=raw['conf'][ids];assert classes.tolist()==e['classes']
            tb=np.array([r['mean'][:4] for r in rows]);tb[:,:2]-=tb[:,2:]/2;tb[:,2:]+=tb[:,:2];tb=tb.astype(np.float32)
            db=np.concatenate([(boxes[:,:2]+boxes[:,2:])/2,boxes[:,2:]-boxes[:,:2]],axis=1).astype(np.float64);db[:,:2]-=db[:,2:]/2;db=db.astype(np.float32);db[:,2:]+=db[:,:2]
            wh=np.maximum(0,np.minimum(tb[:,None,2:],db[None,:,2:])-np.maximum(tb[:,None,:2],db[None,:,:2]));inter=wh.prod(-1);union=(tb[:,2:]-tb[:,:2]).prod(-1)[:,None]+(db[:,2:]-db[:,:2]).prod(-1)[None,:]-inter
            first=1-inter/(union+1e-7);physical=1-(1-first)*confidence[None,:];assert np.max(np.abs(C-physical),initial=0)<1e-7
            allowed=[];wanted={}
            for ri in rr:
                i=int(ri);row=rows[i]
                if row['state']!=2 or not row['history']:continue
                choices,_,_=regions(geom,row['current_crop'],np.array(row['mean'][:2]),boxes,confidence,classes,row['cls']);choices=choices[1:]
                if len(choices)!=2:continue
                good=[]
                for j in choices:
                    transform=geometric_match(row['mean'],row['covariance'],row['history'],row['current_crop'],boxes[j],g)
                    if transform['valid']:good.append((int(j),transform))
                if len(good)!=1:continue
                j,transform=good[0]
                if j not in cc or C[i,j]>=1 or row['cls']!=classes[j]:continue
                _,support=score_from_saved('SUPPORT',video,row,boxes[j],frame,j,transform,None);gap=frame-row['history'][-1]['frame'];accept=support>=3 and gap<=cap
                wanted[(i,j)]=(support,gap,accept)
                if accept:allowed.append((i,j))
            measured={(d['row'],d['column']):d for d in e['decisions']};assert set(measured)==set(wanted)
            for edge,(support,gap,accept) in wanted.items():
                d=measured[edge];assert d['support']==support and d['gap']==gap and d['score']==-gap and d['threshold']==-cap and d['accepted']==accept;checks['independent_age_decisions']+=1
            assert sorted(allowed)==sorted(map(tuple,e['allowed_free_edges']))==sorted(map(tuple,e['plan']['votes']))
            D=np.full((len(rr),len(cc)),np.inf,dtype=C.dtype);ri={int(i):k for k,i in enumerate(rr)};ci={int(j):k for k,j in enumerate(cc)}
            for i,j in allowed:D[ri[i],ci[j]]=C[i,j]-.2
            added=np.asarray(matching.linear_assignment(D,thresh=.8)[0],int).reshape(-1,2);expected=sorted([*map(tuple,native),*((int(rr[i]),int(cc[j])) for i,j in added)])
            assert sorted(map(tuple,e['actual_first']))==expected
            for i,j in expected:
                tid=str(rows[i]['tid'])
                if tid in e['assigned']:assert e['assigned'][tid]==int(ids[j])
            checks['physical_and_residual_cost_matrices']+=1
        gt={int(f['frameIndex']):f for f in read(ROOT/f'data/bdd100k/labels/box_track_20/val/{video}.json')};truth={}
        for frame,anchor in read(FULL/'runtime_inputs'/(video+'.json'))['anchors'].items():
            f=gt[anchor['frameIndex']];assert f['name']==anchor['name'];truth[int(frame)]=[(str(o['id']),o['category'],[o['box2d'][k] for k in ['x1','y1','x2','y2']]) for o in f.get('labels') or [] if o['category'] in CLASS_NAMES.values() and not (o.get('attributes') or {}).get('crowd',False)]
        with np.load(FULL/'rollouts/AGE'/(video+'.npz'),allow_pickle=False) as z:trajectory=z['outputs'];assert int(z['frame_count'])==entry['frames']
        expected=independent_returns(trajectory,truth);saved=read(FULL/'additional/AGE'/(video+'.json'))
        for k in ['totals','events','duplicate_extras']:assert expected[k]==saved[k]
        totals.update(expected['totals']);checks['return_reports']+=1;m=read(FULL/'metrics/AGE'/(video+'.json'));metrics.append(m['metrics'])
        assert m['output_sha256']==sha(FULL/'outputs/AGE'/(video+'.json'));detail=FULL/'metrics/AGE'/(video+'.details.json.gz');assert sha(detail)==m['details_sha256']
        with gzip.open(detail,'rt',encoding='utf-8') as f:d=json.load(f)[video]
        local={}
        for key in ['TP','FP','FN']:
            value=np.sum([d[c]['HOTA']['HOTA_'+key] for c in VALID if c in d],axis=0);assert np.allclose(value,m['components'][key],rtol=0,atol=1e-10);components[key].append(value);local[key]=value
        aa=np.sum([np.array(d[c]['HOTA']['HOTA_TP'])*d[c]['HOTA']['AssA'] for c in VALID if c in d],axis=0);assert np.allclose(aa,m['components']['AA'],rtol=0,atol=1e-10);components['AA'].append(aa)
        assa=aa/np.maximum(1,local['TP']);deta=local['TP']/np.maximum(1,local['TP']+local['FP']+local['FN'])
        for key,value in [('HOTA',100*np.sqrt(assa*deta).mean()),('AssA',100*assa.mean()),('DetA',100*deta.mean())]:assert abs(m['metrics'][key]-float(value))<1e-9
        for key in ['IDTP','IDFP','IDFN']:assert sum(d[c]['Identity'][key] for c in VALID if c in d)==m['components'][key]
        identity=m['components'];idf1=100*identity['IDTP']/max(1.,identity['IDTP']+.5*identity['IDFP']+.5*identity['IDFN']);assert abs(idf1-m['metrics']['IDF1'])<1e-9
        for key in ['IDSw','CLR_FP','CLR_FN']:assert sum(d[c]['CLEAR'][key] for c in VALID if c in d)==m['metrics'][key]
        status('independent_recency_full_audit',complete=number+1,total=len(p['videos']),checks=dict(checks),guard=obs.read())
    assert dict(totals)==read(FULL/'additional_report.json')['arms']['AGE']['totals']
    videos=list(p['videos']);n=len(videos);ix=np.random.default_rng(42).integers(0,n,(20000,n));counts=np.array([np.bincount(i,minlength=n) for i in ix])
    with np.load(FULL/'bootstrap_counts.npz',allow_pickle=False) as z:assert np.array_equal(z['counts'],counts)
    weights=np.vstack([np.ones((1,n)),counts]);values={};points={}
    for arm in p['parameters']:
        c={k:np.array(v) for k,v in components.items()} if arm=='AGE' else {k:np.array([read(OWN_FULL/'metrics'/arm/(v+'.json'))['components'][k] for v in videos]) for k in components}
        tp=weights@c['TP'];aa=weights@c['AA'];den=tp+weights@c['FP']+weights@c['FN'];hota=100*np.mean(np.sqrt((tp/np.maximum(1,den))*(aa/np.maximum(1,tp))),axis=1);points[arm]=float(hota[0]);values[arm]=hota[1:];assert abs(points[arm]-report['pooled'][arm]['HOTA'])<1e-9
    accepted={}
    for key,m in report['comparisons'].items():
        a,b=key.split('-');delta=values[a]-values[b];ci=np.percentile(delta,[2.5/14,100-2.5/14]);assert np.allclose(ci,m['ci_joint14'],rtol=0,atol=1e-9) and np.allclose(np.percentile(delta,[2.5,97.5]),m['ci95'],rtol=0,atol=1e-9)
        assert abs(points[a]-points[b]-m['delta_pooled_HOTA'])<1e-9
        def point(arm):
            folder=FULL if arm=='AGE' else OWN_FULL;data=[read(folder/'metrics'/arm/(v+'.json'))['metrics'] for v in videos];t=read(FULL/'additional_report.json')['arms'][arm]['totals']
            return dict(AssA_mean=float(np.mean([m['AssA'] for m in data])),IDF1_mean=float(np.mean([m['IDF1'] for m in data])),IDSw=sum(m['IDSw'] for m in data),FP=sum(m['CLR_FP'] for m in data),FN=sum(m['CLR_FN'] for m in data),wrong_known_returns=t.get('wrong_returns',0))
        A=point(a);B=point(b);harm={k:A[k]-B[k] for k in A};assert all(abs(harm[k]-m['harm'][k])<1e-9 for k in harm)
        safe=harm['AssA_mean']>=0 and harm['IDF1_mean']>=0 and all(harm[k]<=0 for k in ['IDSw','FP','FN','wrong_known_returns']);assert safe==m['harm_passed'];accepted[key]=bool(ci[0]>0 and safe);checks['joint_intervals']+=1
    assert accepted==report['accepted'];extra=read(FULL/'additional_report.json')['arms'];gains={a:{k:extra[a]['totals'].get(k,0)-extra['B0']['totals'].get(k,0) for k in ['correct_returns','sustained_success']} for a in ['AGE','FLYVIS','IMAGE']}
    assert gains==report['identity_gain'];closure={a:bool(accepted.get(a+'-B0',False) and g['correct_returns']>0 and g['sustained_success']>0) for a,g in gains.items()};assert closure==report['problem1_acceptance']
    assert report['accepted_flyvis_system_and_increment']==all(accepted['FLYVIS-'+a] for a in p['parameters'] if a!='FLYVIS');phase('full_phase.json')
    write(FULL/'independent_audit.json',dict(passed=True,checks=dict(checks),comparison_sha256=sha(FULL/'comparison_report.json'),problem1_acceptance=closure,original_audit_sha256=sha(OWN_FULL/'independent_audit.json')),new=True)


if __name__=='__main__':main(guard())
