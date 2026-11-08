"""Read-only independent audit of saved full trajectories, costs and statistics.

This never runs a tracker, detector, FlyVis network or TrackEval evaluation.
"""
import os,sys,gzip,bisect
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from collections import Counter,defaultdict
from research_singleton_free_20261108.common import *


def overlaps(a,b):
    import numpy as np
    a=np.asarray(a,float).reshape(-1,4);b=np.asarray(b,float).reshape(-1,4)
    wh=np.maximum(0,np.minimum(a[:,None,2:],b[None,:,2:])-np.maximum(a[:,None,:2],b[None,:,:2]))
    inter=wh.prod(-1);aa=np.maximum(0,a[:,2:]-a[:,:2]).prod(-1);bb=np.maximum(0,b[:,2:]-b[:,:2]).prod(-1)
    return inter/np.maximum(aa[:,None]+bb[None,:]-inter,1e-12)


def independent_returns(raw,truth):
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    from datasets.bdd100k_loader import CLASS_NAMES
    outputs={int(f):raw[raw[:,0]==f,1:] for f in np.unique(raw[:,0])};labels={};extras={}
    for frame,objects in truth.items():
        rows=outputs.get(frame,np.empty((0,7)));assigned={};duplicates=defaultdict(set)
        for category in CLASS_NAMES.values():
            ii=[i for i,r in enumerate(rows) if CLASS_NAMES[int(r[6])]==category];gg=[o for o in objects if o[1]==category]
            if not ii or not gg:continue
            iou=overlaps(rows[ii,:4],[o[2] for o in gg]);masked=np.where(iou>=.5,iou,0)
            rr,cc=linear_sum_assignment(-masked)
            for r,c in zip(rr,cc):
                if masked[r,c]>0:assigned[int(rows[ii[r],4])]=gg[c][0]
            for r,i in enumerate(ii):
                c=int(np.argmax(iou[r]))
                if iou[r,c]>=.5:duplicates[gg[c][0]].add(int(rows[i,4]))
        labels[frame]=assigned;extras[frame]=sum(max(0,len(ids)-1) for ids in duplicates.values())
    last={};known={};events=[];totals=Counter();present={};future=defaultdict(list)
    for frame,objects in sorted(truth.items()):
        for identity,_,_ in objects:future[identity].append(frame)
    for frame,rows in sorted(outputs.items()):
        tids=list(map(int,rows[:,4]));assert len(tids)==len(set(tids));present[frame]=set(tids)
        for tid in tids:
            identity=labels.get(frame,{}).get(tid)
            if tid in last and frame-last[tid]>1:
                prior=known.get(tid);kind='unknown' if prior is None or identity is None else ('correct' if prior==identity else 'wrong')
                event=dict(frame=frame,tid=tid,gap=frame-last[tid]-1,prior_identity=prior,identity=identity,kind=kind)
                events.append(event);totals['reactivations']+=1;totals[kind+'_returns']+=1
            last[tid]=frame
            if identity is not None:known[tid]=identity
    for event in events:
        if event['kind']!='correct':event['sustained']='not_correct_return';continue
        target=event['prior_identity'];frames=future[target];start=bisect.bisect_right(frames,event['frame']);following=frames[start:start+2]
        if len(following)<2:kind='censored_insufficient_future'
        elif any(event['tid'] not in present.get(f,set()) for f in following):kind='failed'
        else:
            identities=[labels.get(f,{}).get(event['tid']) for f in following]
            if any(i is not None and i!=target for i in identities):kind='failed'
            elif any(i is None for i in identities):kind='censored_unknown'
            else:kind='success'
        event.update(sustained=kind,future_frames=following);totals['sustained_'+kind]+=1
    totals['duplicate_extra_IDs']=sum(extras.values());totals['annotated_frames_with_duplicates']=sum(v>0 for v in extras.values())
    return dict(totals=dict(totals),events=events,duplicate_extras={str(k):v for k,v in extras.items() if v})


def independent_edges(event,arm,parameters):
    import numpy as np
    from ultralytics.trackers.utils import matching
    C=np.array(event['C'],dtype=event['C_dtype']);native,rr,cc=matching.linear_assignment(C,thresh=.8);native=np.asarray(native,int).reshape(-1,2)
    assert native.tolist()==event['native_first'];rows=event['rows'];classes=event['classes'];plan=event['plan']
    D=C.copy();alpha=parameters['alpha'];beta=parameters['beta'];allowed=[]
    if arm=='U_THRESHOLD':
        for i in rr:
            for j in cc:
                if rows[i]['state']==2 and rows[i]['cls']==classes[j] and C[i,j]<1:D[i,j]-=alpha;allowed.append((int(i),int(j)))
    else:
        proposal=C.copy()
        for i,j in plan['votes']:proposal[i,j]-=alpha
        mapped='G' if arm=='U_GEO' else arm.replace('U_','G_',1)
        for j,col in plan['columns'].items():
            if not col['geometry_valid'] or len(col['rows'])<2:continue
            indices=np.array(col['rows'],int)
            if arm=='U_SHIFT':proposal[indices,int(j)]-=beta;continue
            values=col['scores'].get(mapped)
            if values is None or any(v is None for v in values):continue
            values=np.clip(np.array(values,float),-1,1);proposal[indices,int(j)]-=beta*(values-values.mean())
        for i,j in plan['votes']:
            if i in rr and j in cc and rows[i]['state']==2 and rows[i]['cls']==classes[j] and C[i,j]<1:
                D[i,j]=proposal[i,j];allowed.append((i,j))
    assert np.array_equal(D,np.array(event['corrected'],dtype=C.dtype));assert sorted(allowed)==sorted(map(tuple,event['allowed_free_edges']))
    residual=np.full((len(rr),len(cc)),np.inf,dtype=C.dtype);ri={int(i):k for k,i in enumerate(rr)};ci={int(j):k for k,j in enumerate(cc)}
    for i,j in allowed:residual[ri[i],ci[j]]=D[i,j]
    if np.array_equal(D,C):added=[]
    else:added=[(int(rr[i]),int(cc[j])) for i,j in np.asarray(matching.linear_assignment(residual,thresh=.8)[0],int).reshape(-1,2)]
    assert sorted(map(tuple,event['actual_first']))==sorted([*map(tuple,native),*added])
    assert set(map(tuple,native)).issubset(set(map(tuple,event['actual_first'])))
    return len(added)


def main(obs):
    import numpy as np
    from datasets.bdd100k_loader import CLASS_NAMES
    from evaluation.hota_eval import VALID
    from research_singleton_free_20261108.full_paths import FULL,raw_path,source_arm
    phase('full_phase.json');assert read(FULL/'analyses_complete.json')['passed']
    p=read(FULL/'protocol.json');report=read(FULL/'comparison_report.json');videos=list(p['videos']);arms=list(p['parameters'])
    checks=Counter();return_cache={};totals={a:Counter() for a in arms};components={a:defaultdict(list) for a in arms};metrics={a:[] for a in arms}
    for number,(video,entry) in enumerate(p['videos'].items()):
        for name,h in entry['sources'].items():assert sha(ROOT/name)==h
        for name,h in read(FULL/'completed'/(video+'.json'))['artifacts'].items():assert sha(ROOT/name)==h
        anchors=read(FULL/'runtime_inputs'/(video+'.json'))['anchors'];gt=read(ROOT/f'data/bdd100k/labels/box_track_20/val/{video}.json')
        gt={int(f['frameIndex']):f for f in gt};truth={}
        for frame,clock in anchors.items():
            f=gt[clock['frameIndex']];assert f['name']==clock['name'];objects=[]
            for obj in f.get('labels') or []:
                if obj['category'] not in CLASS_NAMES.values() or (obj.get('attributes') or {}).get('crowd',False):continue
                objects.append((str(obj['id']),obj['category'],[obj['box2d'][k] for k in ['x1','y1','x2','y2']]))
            truth[int(frame)]=objects
        event_cache=set()
        for arm in arms:
            path=raw_path(p,arm,video);signature=(video,sha(path))
            with np.load(path,allow_pickle=False) as z:raw=z['outputs'];assert int(z['frame_count'])==entry['frames']
            assert raw.ndim==2 and raw.shape[1]==8 and np.isfinite(raw).all()
            assert np.all((raw[:,0]>=0)&(raw[:,0]<entry['frames']))
            for frame in np.unique(raw[:,0]):
                tids=raw[raw[:,0]==frame,5];assert len(tids)==len(np.unique(tids))
            if signature not in return_cache:return_cache[signature]=independent_returns(raw,truth);checks['unique_return_trajectories']+=1
            expected=return_cache[signature];saved=read(FULL/'additional'/arm/(video+'.json'))
            for key in ['totals','events','duplicate_extras']:assert saved[key]==expected[key],(video,arm,key)
            totals[arm].update(expected['totals']);checks['return_reports']+=1
            with np.load(ROOT/entry['detections'],allow_pickle=False) as z:detections={k:z[k] for k in z.files}
            actual=source_arm(p,arm);eventpath=FULL/'rollouts'/actual/(video+'.events.json.gz')
            if actual!='B0' and eventpath not in event_cache:
                event_cache.add(eventpath)
                receipt=read(FULL/'rollouts'/actual/(video+'.receipt.json'))
                for name,h in {**receipt['artifacts'],**receipt['feature_files']}.items():assert sha(ROOT/name)==h
                with gzip.open(eventpath,'rt',encoding='utf-8') as f:events=json.load(f)
                for event in events:
                    checks['independent_cost_matrices']+=1;checks['additional_edges']+=independent_edges(event,actual,p['parameters'][actual])
                    frame=event['frame'];rawids=np.flatnonzero((detections['frame_idx']==frame)&(detections['conf']>=.25))
                    assert len(rawids)==len(event['classes'])
                    for i,j in event['actual_first']:
                        tid=str(event['rows'][i]['tid'])
                        if tid in event['assigned']:assert event['assigned'][tid]==int(rawids[j])
                    assert set(map(int,event['assigned'])).issubset(set(event['emitted_ids']))
            metric=read(FULL/'metrics'/arm/(video+'.json'));metrics[arm].append(metric['metrics'])
            assert metric['output_sha256']==sha(FULL/'outputs'/arm/(video+'.json'))
            with gzip.open(FULL/'metrics'/arm/(video+'.details.json.gz'),'rt',encoding='utf-8') as f:detail=json.load(f)[video]
            # Rebuild detection-weighted sufficient statistics from official class details.
            for key in ['TP','FP','FN']:
                value=np.sum([detail[c]['HOTA']['HOTA_'+key] for c in VALID if c in detail],axis=0)
                assert np.allclose(value,metric['components'][key],rtol=0,atol=1e-10);components[arm][key].append(value)
            aa=np.sum([np.array(detail[c]['HOTA']['HOTA_TP'])*detail[c]['HOTA']['AssA'] for c in VALID if c in detail],axis=0)
            assert np.allclose(aa,metric['components']['AA'],rtol=0,atol=1e-10);components[arm]['AA'].append(aa)
        status('independent_full_first_problem_audit',complete=number+1,total=len(videos),checks=dict(checks),guard=obs.read())
    indices=np.random.default_rng(42).integers(0,len(videos),(20000,len(videos)))
    counts=np.array([np.bincount(v,minlength=len(videos)) for v in indices])
    with np.load(FULL/'bootstrap_counts.npz',allow_pickle=False) as z:assert np.array_equal(counts,z['counts'])
    values={};points={}
    for arm in arms:
        c={k:np.array(v) for k,v in components[arm].items()};weights=np.vstack([np.ones((1,len(videos))),counts])
        tp=weights@c['TP'];den=tp+weights@c['FP']+weights@c['FN'];aa=weights@c['AA']
        hota=100*np.mean(np.sqrt((tp/np.maximum(1,den))*(aa/np.maximum(1,tp))),axis=1)
        points[arm]=float(hota[0]);values[arm]=hota[1:];assert abs(points[arm]-report['pooled'][arm]['HOTA'])<1e-9
        for k,v in totals[arm].items():assert v==read(FULL/'additional_report.json')['arms'][arm]['totals'][k]
    accepted={}
    for a,b in p['primary_comparisons']:
        key=a+'-'+b;record=report['comparisons'][key];d=values[a]-values[b]
        assert abs(points[a]-points[b]-record['delta_pooled_HOTA'])<1e-9
        assert np.allclose(np.percentile(d,[2.5,97.5]),record['ci95'],rtol=0,atol=1e-9)
        interval=np.percentile(d,[2.5/11,100-2.5/11]);assert np.allclose(interval,record['ci_corrected'],rtol=0,atol=1e-9)
        harm=dict(AssA_mean=float(np.mean([m['AssA'] for m in metrics[a]])-np.mean([m['AssA'] for m in metrics[b]])),
            IDF1_mean=float(np.mean([m['IDF1'] for m in metrics[a]])-np.mean([m['IDF1'] for m in metrics[b]])),
            IDSw=sum(m['IDSw'] for m in metrics[a])-sum(m['IDSw'] for m in metrics[b]),
            FP=sum(m['CLR_FP'] for m in metrics[a])-sum(m['CLR_FP'] for m in metrics[b]),
            FN=sum(m['CLR_FN'] for m in metrics[a])-sum(m['CLR_FN'] for m in metrics[b]),
            wrong_known_returns=totals[a]['wrong_returns']-totals[b]['wrong_returns'])
        assert all(abs(harm[k]-record['harm'][k])<1e-9 for k in harm)
        safe=harm['AssA_mean']>=0 and harm['IDF1_mean']>=0 and all(harm[k]<=0 for k in ['IDSw','FP','FN','wrong_known_returns'])
        assert safe==record['harm_passed'];accepted[key]=bool(interval[0]>0 and safe);checks['independent_bootstraps']+=1
    assert report['accepted_geometry']==accepted['U_GEO-B0'];assert report['accepted_threshold']==accepted['U_THRESHOLD-B0']
    assert report['accepted_flyvis_system_and_increment']==all(accepted['U_FLYVIS-'+b] for b in arms if b!='U_FLYVIS')
    assert report['accepted_system_variants']==dict(U_GEO=accepted['U_GEO-B0'],U_THRESHOLD=accepted['U_THRESHOLD-B0'],U_FLYVIS=accepted['U_FLYVIS-B0'])
    phase('full_phase.json');write(FULL/'independent_audit.json',dict(passed=True,checks=dict(checks),
        comparison_sha256=sha(FULL/'comparison_report.json'),analyses_sha256=sha(FULL/'analyses_complete.json'),
        accepted=accepted,scope='Saved-output audit only. Independent costs, native-pair locks, raw bindings, sparse identity labels and returns, class components and all11paired-video intervals. Historical cohort limitations retained.'),new=True)


if __name__=='__main__':main(guard())
