"""Independent saved-output audit; no tracker, detector or core simulation."""
import os,sys,gzip
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from collections import Counter,defaultdict
from research_singleton_verify_full_20261108.common import *


def score_from_saved(kind,video,row,box,frame,column,transform,features):
    import numpy as np
    from scipy.spatial import cKDTree,Delaunay
    g=read(ROOT/'runs/z13_local_peak_20261021/geometry.json');points=np.array(g['points'],float)
    old=row['history'][-1];b=np.array(old['box'],float);box=np.array(box,float);c0=np.array(old['crop'],float);ct=np.array(row['current_crop'],float)
    xy0=c0[:2]+points*(c0[2:]-c0[:2])/g['eye_size'];xyt=ct[:2]+points*(ct[2:]-ct[:2])/g['eye_size']
    xx,yy=np.meshgrid(np.arange(4,320,8)*4,np.arange(4,180,8)*4);hull=Delaunay(np.column_stack([xx.ravel(),yy.ravel()]).astype(np.float32))
    valid0=hull.find_simplex(xy0)>=0;validt=hull.find_simplex(xyt)>=0;inside=((xy0>=b[:2])&(xy0<=b[2:])).all(1)
    p=np.flatnonzero(((xyt>=box[:2])&(xyt<=box[2:])).all(1));projected=b[:2]+transform['offset']+(xyt[p]-box[:2])/transform['scale']
    distance,q=cKDTree(xy0).query(projected);spacing=g['spacing']*(c0[2]-c0[0])/g['eye_size']
    keep=((projected>=b[:2])&(projected<=b[2:])).all(1)&inside[q]&(distance<=.6*spacing)&validt[p]&valid0[q]
    p=p[keep];q=q[keep];unique=np.unique(q)
    if len(unique)<3:return None,0
    if kind=='SUPPORT':return 1.,len(unique)
    if kind=='RANDOM':return 2*(int.from_bytes(hashlib.sha256(f"{video}|{frame}|{row['tid']}|{column}|42".encode()).digest()[:8],'little')/2**64)-1,len(unique)
    name='C' if kind in ['FLYVIS','PERM'] else kind;reference_seq=[[h['frame'],h['crop']] for h in row['history']]
    current=features(reference_seq+row['search'],name);reference=features(reference_seq,name)
    x=np.stack([current[p[q==cell]].mean(0) for cell in unique]);y=reference[unique]
    def cosine(a,b,center=True):
        if center:a=a-a.mean(0);b=b-b.mean(0)
        aa=np.sum(a*a);bb=np.sum(b*b)
        return None if min(aa,bb)<1e-12 else float(np.clip(np.sum(a*b)/np.sqrt(aa*bb),-1,1))
    if kind=='DIS':return cosine(x/np.array(transform['scale']),y,False),len(unique)
    if kind!='PERM':return cosine(x,y),len(unique)
    ids=np.flatnonzero(inside&valid0);values=[]
    for rep in range(32):
        seed=int.from_bytes(hashlib.sha256(f"{video}|{row['tid']}|{old['frame']}|{rep}|42".encode()).digest()[:8],'little')
        copied=reference.copy();copied[ids]=reference[np.random.default_rng(seed).permutation(ids)];values.append(cosine(x,copied[unique]))
    return (None if any(v is None for v in values) else float(np.mean(values))),len(unique)


def audit_events(video,arm,events,raw,parameters,feature_sources,checks):
    import numpy as np
    from ultralytics.trackers.utils import matching
    from tracking.z13_readout import HexGeometry
    from tracking.z13_region_assignment import regions
    from tracking.spatial_identity_assignment import geometric_match
    geometry=read(ROOT/'runs/z13_local_peak_20261021/geometry.json');geom=HexGeometry(np.array(geometry['points'],np.float32),geometry['eye_size'])
    cache={}
    def features(sequence,name):
        key=sequence_key(video,sequence);path=FULL/'features'/video/(key+'.'+name+'.npy');relative=path.relative_to(ROOT).as_posix()
        if relative not in cache:
            assert sha(path)==feature_sources[relative];cache[relative]=np.load(path,allow_pickle=False).astype(float)
        return cache[relative]
    for e in events:
        C=np.array(e['C'],dtype=e['C_dtype']);rows=e['rows'];frame=e['frame'];native,rr,cc=matching.linear_assignment(C,thresh=.8)
        native=np.asarray(native,int).reshape(-1,2);assert native.tolist()==e['native_first']
        ids=np.flatnonzero((raw['frame_idx']==frame)&(raw['conf']>=.25));boxes=raw['xyxy'][ids];confidence=raw['conf'][ids];classes=raw['cls'][ids].astype(int)
        assert len(ids)==C.shape[1] and classes.tolist()==e['classes']
        # Independent physical IoU/confidence reconstruction, retaining the
        # actual saved float matrix for exact LAP and threshold verification.
        mu=np.array([r['mean'][:4] for r in rows]);tb=mu.copy();tb[:,:2]-=tb[:,2:]/2;tb[:,2:]+=tb[:,:2];tb=tb.astype(np.float32)
        # _DetSlice forms float32xywh; parse_bboxes appends integer indices
        # and promotes to float64; STrack stores float32ltwh, then xyxy.
        center=(boxes[:,:2]+boxes[:,2:])/2;size=boxes[:,2:]-boxes[:,:2]
        db=np.concatenate([center,size],axis=1).astype(np.float64);db[:,:2]-=db[:,2:]/2;db=db.astype(np.float32);db[:,2:]+=db[:,:2]
        lo=np.maximum(tb[:,None,:2],db[None,:,:2]);hi=np.minimum(tb[:,None,2:],db[None,:,2:]);wh=np.maximum(0,hi-lo);intersection=wh.prod(-1)
        area1=(tb[:,2:]-tb[:,:2]).prod(-1);area2=(db[:,2:]-db[:,:2]).prod(-1)
        iou=intersection/(area1[:,None]+area2[None,:]-intersection+1e-7)
        first_cost=1-iou;physical=1-(1-first_cost)*confidence[None,:]
        error=float(np.max(np.abs(C-physical),initial=0));assert error<1e-7,(video,arm,frame,error)
        checks['physical_cost_matrices']+=1
        allowed=[];expected_decisions={}
        if arm=='THRESHOLD':
            allowed=[(int(i),int(j)) for i in rr for j in cc if rows[i]['state']==2 and rows[i]['cls']==classes[j] and C[i,j]<1]
        else:
            for raw_i in rr:
                i=int(raw_i);row=rows[i]
                if row['state']!=2 or not row['history']:continue
                choice,_,_=regions(geom,row['current_crop'],np.array(row['mean'][:2]),boxes,confidence,classes,row['cls']);choice=choice[1:]
                if len(choice)!=2:continue
                gg=[geometric_match(row['mean'],row['covariance'],row['history'],row['current_crop'],boxes[j],geometry) for j in choice]
                good=[(int(j),g) for j,g in zip(choice,gg) if g['valid']]
                if len(good)!=1:continue
                j,g=good[0]
                if j not in cc or C[i,j]>=1 or row['cls']!=classes[j]:continue
                if arm=='GEO':allowed.append((i,j));continue
                value,support=score_from_saved(arm,video,row,boxes[j],frame,j,g,features);threshold=parameters['threshold']
                accepted=threshold!='off' and value is not None and value>=threshold
                expected_decisions[(i,j)]=dict(score=value,support=support,accepted=accepted)
                if accepted:allowed.append((i,j))
            measured={(d['row'],d['column']):d for d in e['decisions']}
            assert set(measured)==set(expected_decisions)
            for key,wanted in expected_decisions.items():
                actual=measured[key];assert actual['support']==wanted['support'] and actual['accepted']==wanted['accepted']
                assert (actual['score'] is None)==(wanted['score'] is None)
                if actual['score'] is not None:assert abs(actual['score']-wanted['score'])<1e-12
                checks['independent_spatial_decisions']+=1
        assert sorted(allowed)==sorted(map(tuple,e['allowed_free_edges']))
        D=np.full((len(rr),len(cc)),np.inf,dtype=C.dtype);ri={int(i):k for k,i in enumerate(rr)};ci={int(j):k for k,j in enumerate(cc)}
        for i,j in allowed:D[ri[i],ci[j]]=C[i,j]-parameters['alpha']
        added=np.asarray(matching.linear_assignment(D,thresh=.8)[0],int).reshape(-1,2)
        wanted=sorted([*map(tuple,native),*((int(rr[i]),int(cc[j])) for i,j in added)])
        assert sorted(map(tuple,e['actual_first']))==wanted and set(map(tuple,native)).issubset(set(wanted))
        for i,j in e['actual_first']:
            tid=str(rows[i]['tid'])
            if tid in e['assigned']:assert e['assigned'][tid]==int(ids[j])
        checks['independent_residual_matrices']+=1


def main(obs):
    import numpy as np
    from datasets.bdd100k_loader import CLASS_NAMES
    from evaluation.hota_eval import VALID
    from research_singleton_free_20261108.full_audit import independent_returns
    from research_singleton_free_20261108.full_paths import raw_path,source_arm
    configure_paths();phase('full_audit_phase.json');assert read(FULL/'analyses_complete.json')['passed']
    p=read(FULL/'protocol.json');report=read(FULL/'comparison_report.json');arms=list(p['parameters']);videos=list(p['videos']);checks=Counter()
    components={a:{k:[] for k in ['TP','FP','FN','AA']} for a in arms};metrics={a:[] for a in arms};totals={a:Counter() for a in arms}
    for number,(video,entry) in enumerate(p['videos'].items()):
        for name,h in entry['sources'].items():assert sha(ROOT/name)==h
        for name,h in read(FULL/'completed'/(video+'.json'))['artifacts'].items():assert sha(ROOT/name)==h
        clock=read(FULL/'runtime_inputs'/(video+'.json'))['anchors'];gtfile=ROOT/f'data/bdd100k/labels/box_track_20/val/{video}.json'
        gt={int(f['frameIndex']):f for f in read(gtfile)};truth={}
        for frame,anchor in clock.items():
            f=gt[anchor['frameIndex']];assert f['name']==anchor['name'];objects=[]
            for obj in f.get('labels') or []:
                if obj['category'] not in CLASS_NAMES.values() or (obj.get('attributes') or {}).get('crowd',False):continue
                objects.append((str(obj['id']),obj['category'],[obj['box2d'][k] for k in ['x1','y1','x2','y2']]))
            truth[int(frame)]=objects
        with np.load(ROOT/entry['detections'],allow_pickle=False) as z:raw_detections={k:z[k] for k in z.files}
        returns={};event_seen=set()
        for arm in arms:
            rawfile=raw_path(p,arm,video)
            with np.load(rawfile,allow_pickle=False) as z:raw=z['outputs'];assert int(z['frame_count'])==entry['frames']
            assert raw.ndim==2 and raw.shape[1]==8 and np.isfinite(raw).all();signature=sha(rawfile)
            if signature not in returns:returns[signature]=independent_returns(raw,truth);checks['independent_return_trajectories']+=1
            expected=returns[signature];saved=read(FULL/'additional'/arm/(video+'.json'));assert saved['gt_sha256']==sha(gtfile)
            for key in ['totals','events','duplicate_extras']:assert saved[key]==expected[key],(video,arm,key)
            totals[arm].update(expected['totals']);checks['return_reports']+=1
            actual=source_arm(p,arm)
            if actual!='B0' and actual not in event_seen:
                event_seen.add(actual);receipt=read(FULL/'rollouts'/actual/(video+'.receipt.json'));assert receipt['passed']
                for name,h in {**receipt['artifacts'],**receipt['feature_files']}.items():assert sha(ROOT/name)==h
                with gzip.open(FULL/'rollouts'/actual/(video+'.events.json.gz'),'rt',encoding='utf-8') as f:events=json.load(f)
                audit_events(video,actual,events,raw_detections,p['parameters'][actual],receipt['feature_files'],checks)
            m=read(FULL/'metrics'/arm/(video+'.json'));assert m['passed'];metrics[arm].append(m['metrics'])
            assert m['output_sha256']==sha(FULL/'outputs'/arm/(video+'.json'));details=FULL/'metrics'/arm/(video+'.details.json.gz');assert sha(details)==m['details_sha256']
            with gzip.open(details,'rt',encoding='utf-8') as f:d=json.load(f)[video]
            local={}
            for k in ['TP','FP','FN']:
                value=np.sum([d[c]['HOTA']['HOTA_'+k] for c in VALID if c in d],axis=0);assert np.allclose(value,m['components'][k],rtol=0,atol=1e-10);components[arm][k].append(value)
                local[k]=value
            value=np.sum([np.array(d[c]['HOTA']['HOTA_TP'])*d[c]['HOTA']['AssA'] for c in VALID if c in d],axis=0)
            assert np.allclose(value,m['components']['AA'],rtol=0,atol=1e-10);components[arm]['AA'].append(value)
            assa=value/np.maximum(1,local['TP']);deta=local['TP']/np.maximum(1,local['TP']+local['FP']+local['FN'])
            for key,expected in [('HOTA',100*np.sqrt(assa*deta).mean()),('AssA',100*assa.mean()),('DetA',100*deta.mean())]:assert abs(float(expected)-m['metrics'][key])<1e-9
            identity={key:sum(int(d[c]['Identity'][key]) for c in VALID if c in d) for key in ['IDTP','IDFP','IDFN']}
            for key,value in identity.items():assert value==m['components'][key]
            idf1=100*identity['IDTP']/max(1.,identity['IDTP']+.5*identity['IDFP']+.5*identity['IDFN']);assert abs(idf1-m['metrics']['IDF1'])<1e-9
            for key in ['IDSw','CLR_FP','CLR_FN']:assert sum(d[c]['CLEAR'][key] for c in VALID if c in d)==m['metrics'][key]
            checks['independent_video_metric_components']+=1
        status('independent_own_template_full_audit',complete=number+1,total=len(videos),checks=dict(checks),guard=obs.read())
    n=len(videos);indices=np.random.default_rng(42).integers(0,n,(20000,n));counts=np.array([np.bincount(v,minlength=n) for v in indices])
    with np.load(FULL/'bootstrap_counts.npz',allow_pickle=False) as z:assert np.array_equal(counts,z['counts'])
    weights=np.vstack([np.ones((1,n)),counts]);samples={};points={}
    for arm,c in components.items():
        c={k:np.array(v) for k,v in c.items()};tp=weights@c['TP'];aa=weights@c['AA'];den=tp+weights@c['FP']+weights@c['FN']
        values=100*np.mean(np.sqrt((tp/np.maximum(1,den))*(aa/np.maximum(1,tp))),axis=1);points[arm]=float(values[0]);samples[arm]=values[1:]
        assert abs(points[arm]-report['pooled'][arm]['HOTA'])<1e-9
        assert dict(totals[arm])==read(FULL/'additional_report.json')['arms'][arm]['totals']
    accepted={}
    for key,value in report['comparisons'].items():
        a,b=key.split('-');delta=samples[a]-samples[b];ci=np.percentile(delta,[2.5/11,100-2.5/11])
        assert abs(points[a]-points[b]-value['delta_pooled_HOTA'])<1e-9
        assert np.allclose(np.percentile(delta,[2.5,97.5]),value['ci95'],rtol=0,atol=1e-9) and np.allclose(ci,value['ci_corrected'],rtol=0,atol=1e-9)
        harm=dict(AssA_mean=float(np.mean([m['AssA'] for m in metrics[a]])-np.mean([m['AssA'] for m in metrics[b]])),
            IDF1_mean=float(np.mean([m['IDF1'] for m in metrics[a]])-np.mean([m['IDF1'] for m in metrics[b]])),
            IDSw=sum(m['IDSw'] for m in metrics[a])-sum(m['IDSw'] for m in metrics[b]),FP=sum(m['CLR_FP'] for m in metrics[a])-sum(m['CLR_FP'] for m in metrics[b]),
            FN=sum(m['CLR_FN'] for m in metrics[a])-sum(m['CLR_FN'] for m in metrics[b]),wrong_known_returns=totals[a]['wrong_returns']-totals[b]['wrong_returns'])
        assert all(abs(harm[k]-value['harm'][k])<1e-9 for k in harm)
        safe=harm['AssA_mean']>=0 and harm['IDF1_mean']>=0 and all(harm[k]<=0 for k in ['IDSw','FP','FN','wrong_known_returns'])
        assert safe==value['harm_passed'];accepted[key]=bool(ci[0]>0 and safe);checks['independent_bootstraps']+=1
    chosen=p['selected_system'];gain=dict(correct_returns=totals[chosen]['correct_returns']-totals['B0']['correct_returns'],sustained_success=totals[chosen]['sustained_success']-totals['B0']['sustained_success'])
    assert report['identity_gain']==gain and report['accepted_selected_system']==accepted[chosen+'-B0'] and report['accepted_support']==accepted['SUPPORT-B0']
    assert report['problem1_acceptance']==bool(accepted[chosen+'-B0'] and gain['correct_returns']>0 and gain['sustained_success']>0)
    assert report['accepted_flyvis_system_and_increment']==all(accepted['FLYVIS-'+a] for a in arms if a!='FLYVIS')
    phase('full_audit_phase.json');write(FULL/'independent_audit.json',dict(passed=True,checks=dict(checks),comparison_sha256=sha(FULL/'comparison_report.json'),
        additional_sha256=sha(FULL/'additional_report.json'),accepted=accepted,problem1_acceptance=report['problem1_acceptance'],
        scope='Independent saved-data arithmetic, native-cost reconstruction, locked pairs, absolute spatial scores, full sparse identity returns, official components and all registered intervals. No repeated scientific inference.'),new=True)


if __name__=='__main__':main(guard())
