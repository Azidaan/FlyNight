from common import *
import gzip,ast
import numpy as np,lap
from scipy.optimize import linear_sum_assignment
p=verify();t=read(OUT/'trace.json');native=ROOT/'runs/domain_matched_native_20261109';config=read(native/'protocol.json')['native_config'];checks=[];matrices=[];additional=[];maxerr=0.;count=0
def iou(a,b):
    intersection=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    union=max(0,a[2]-a[0])*max(0,a[3]-a[1])+max(0,b[2]-b[0])*max(0,b[3]-b[1])-intersection
    return intersection/union if union>0 else 0.
def label(a,objects):
    out=[]
    for o in objects:
        if o['category']!='car' or (o.get('attributes') or {}).get('crowd',False):continue
        score=iou(a,[o['box2d'][k] for k in ['x1','y1','x2','y2']])
        if score>=.5:out.append(dict(identity=str(o['id']),iou=score))
    return out
for c in t['cases']:
    v=c['video'];gt=read(ROOT/f'data/bdd100k/labels/box_track_20/val/{v}.json');anchors=read(LEGACY/'runtime_inputs'/(v+'.json'))['anchors'];ann={}
    for f,a in anchors.items():
        fr=next(g for i,g in enumerate(gt) if int(g.get('frameIndex',i))==a['frameIndex']);assert fr['name']==a['name'];ann[int(f)]=fr.get('labels') or []
    with gzip.open(native/'rollouts/C34'/(v+'.events.json.gz'),'rt',encoding='utf8') as f:logs=json.load(f)
    with np.load(native/'rollouts/C34'/(v+'.npz'),allow_pickle=False) as z:public=z['outputs']
    with np.load(ROOT/read(native/'protocol.json')['videos'][v]['detections'],allow_pickle=False) as z:raw={k:z[k] for k in z.files}
    for tid,hh in c['histories'].items():
        assignments=[a for a in logs['assignments'] if a['tid']==int(tid)]
        assert len(assignments)==len(hh)
        for a,h in zip(assignments,hh):
            for key in a:assert a[key]==h[key]
            f=a['frame'];rid=a['raw_id'];assert int(raw['frame_idx'][rid])==f;assert h['raw_box']==raw['xyxy'][rid].tolist()
            rows=public[(public[:,0]==f)&(public[:,5]==int(tid))];assert len(rows)==int(a['emitted'])
            if len(rows):assert h['public_box']==rows[0][1:5].tolist()
            for key,box in [('raw_hits',h['raw_box']),('public_hits',h['public_box'])]:
                expected=None if f not in ann or box is None else label(box,ann[f])
                assert [x['identity'] for x in expected or []]==[x['identity'] for x in h[key] or []]
                assert (expected is None)==(h[key] is None)
                for x,y in zip(expected or [],h[key] or []):maxerr=max(maxerr,abs(x['iou']-y['iou']))
            count+=1
        birth=hh[0];assert birth['start_frame']==birth['frame']+1 and not birth['emitted'];assert birth['confidence']>=config['new_track_thresh']
        checks.append(dict(video=v,tid=int(tid),birth_frame=birth['frame'],first_public=next(h['frame'] for h in hh if h['emitted']),birth_matrix_saved=birth['matrix_saved'],first_unique_target=next((h['frame'] for h in hh if h['public_hits'] and len(h['public_hits'])==1 and h['public_hits'][0]['identity']==c['target']),None),origin_rows=len(hh)))
    # Recompute complete ownership census independently, including IDs not selected initially.
    census=[]
    for f in sorted(ann):
        if f>c['endpoint']:continue
        tracks=[]
        for row in public[(public[:,0]==f)&(public[:,7]==2)]:
            hits=label(row[1:5].tolist(),ann[f])
            if any(h['identity']==c['target'] for h in hits):tracks.append(dict(tid=int(row[5]),hits=hits,unique=len(hits)==1,box=row[1:5].tolist()))
        if tracks:census.append(dict(frame=f,tracks=tracks))
    assert len(census)==len(c['target_public_owners'])
    for a,b in zip(census,c['target_public_owners']):
        assert a['frame']==b['frame'] and [x['tid'] for x in a['tracks']]==[x['tid'] for x in b['tracks']]
    extra_ids=sorted({x['tid'] for o in census for x in o['tracks'] if x['unique']}-set(c['ids']))
    for tid in extra_ids:
        aa=[a for a in logs['assignments'] if a['tid']==tid];rr=[]
        for a in aa:
            f=a['frame'];rid=a['raw_id'];assert rid>=0 and int(raw['frame_idx'][rid])==f
            rows=public[(public[:,0]==f)&(public[:,5]==tid)];assert len(rows)==int(a['emitted'])
            rr.append(dict(**a,raw_box=raw['xyxy'][rid].tolist(),confidence=float(raw['conf'][rid]),public_hits=None if f not in ann or not len(rows) else label(rows[0][1:5].tolist(),ann[f])))
        additional.append(dict(video=v,tid=tid,reason='Found in complete same-GT ownership census; follow up without replacing preregistered cases.',assignments=rr))
    for f in ([539,544] if c['ids'][0]==232 else [534,536,540]):
        e=next(e for e in logs['events'] if e['frame']==f);C=np.array(e['C'],np.float32);corr=C.copy()
        for i,j in e['votes']:corr[i,j]-=.2
        assert np.array_equal(corr,np.array(e['corrected'],np.float32))
        _,x,_=lap.lapjv(corr,extend_cost=True,cost_limit=config['match_thresh']);pairs=sorted((i,int(j)) for i,j in enumerate(x) if j>=0);assert pairs==sorted(map(tuple,e['actual_first']))
        n,k=C.shape;aug=np.zeros((n+k,n+k));aug[:n,:k]=corr;aug[:n,k:]=.4;aug[n:,:k]=.4;ii,jj=linear_sum_assignment(aug);objective=sum(float(corr[i,j]) for i,j in pairs)+.4*(n+k-2*len(pairs));err=abs(objective-float(aug[ii,jj].sum()));assert err<1e-9
        selected=[dict(tid=e['rows'][i]['tid'],state=e['rows'][i]['state'],raw_id=e['high_raw_ids'][j],cost=float(corr[i,j])) for i,j in pairs if e['rows'][i]['tid'] in c['ids']]
        selected_columns={j for i,j in pairs if e['rows'][i]['tid'] in c['ids']}
        competitors=[dict(tid=r['tid'],state=r['state'],raw_id=e['high_raw_ids'][j],cost=float(corr[i,j]),original=float(C[i,j])) for i,r in enumerate(e['rows']) if r['tid'] in c['ids'] for j in sorted(selected_columns)]
        matrices.append(dict(video=v,frame=f,selected=selected,competitors=competitors,proposals=[q for q in e['proposals'] if q['old_id'] in c['ids']],independent_objective_error=err,full_assignment_exact=True))
    print('AUDITED',v,flush=True)
assert maxerr<1e-12
verify();save('audit.json',dict(passed=True,rows=count,iou_max_error=maxerr,birth_checks=checks,matrix_checks=matrices,additional_histories=additional,trace_sha256=sha(OUT/'trace.json'),script_sha256=sha(__file__),limits='No tracker update or lifecycle replay. Shared primary annotations and saved output; independent scalar geometry and a separate augmented assignment solver. Annotation-free frames remain unknown.'))
print('PASS',count,'MAX_IOU_ERROR',maxerr,'EXTRA_IDS',[(a['video'],a['tid'],a['assignments'][0]['frame']) for a in additional])
