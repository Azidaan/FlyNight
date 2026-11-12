from common import *
import gzip,collections
import numpy as np
p=verify();native=ROOT/'runs/domain_matched_native_20261109';np_=read(native/'protocol.json');result=[]
def matches(box,labels):
    a=np.asarray(box);out=[]
    for x in labels:
        if x['category']!='car' or (x.get('attributes') or {}).get('crowd',False):continue
        b=np.array([x['box2d'][k] for k in ['x1','y1','x2','y2']]);inter=np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])).prod();union=np.maximum(0,a[2:]-a[:2]).prod()+np.maximum(0,b[2:]-b[:2]).prod()-inter
        if union>0 and inter/union>=.5:out.append(dict(identity=str(x['id']),iou=float(inter/union)))
    return out
for c in p['cases']:
    v=c['video'];gt=read(ROOT/f'data/bdd100k/labels/box_track_20/val/{v}.json');by={int(x.get('frameIndex',i)):x for i,x in enumerate(gt)};ann={}
    for f,a in read(LEGACY/'runtime_inputs'/(v+'.json'))['anchors'].items():
        fr=by[a['frameIndex']];assert fr['name']==a['name'];ann[int(f)]=fr.get('labels') or []
    with gzip.open(native/'rollouts/C34'/(v+'.events.json.gz'),'rt',encoding='utf8') as f:logs=json.load(f)
    with np.load(native/'rollouts/C34'/(v+'.npz'),allow_pickle=False) as z:public=z['outputs']
    with np.load(ROOT/np_['videos'][v]['detections'],allow_pickle=False) as z:raw={k:z[k] for k in ['frame_idx','xyxy','conf','cls']}
    events={e['frame']:e for e in logs['events']};histories={};owners=[]
    for tid in c['ids']:
        history=[]
        for a in logs['assignments']:
            if a['tid']!=tid:continue
            f=a['frame'];rid=a['raw_id'];assert rid>=0 and int(raw['frame_idx'][rid])==f
            rows=public[(public[:,0]==f)&(public[:,5]==tid)];assert len(rows)==int(a['emitted'])
            if len(rows):assert np.float32(rows[0][6])==np.float32(raw['conf'][rid]) and int(rows[0][7])==int(raw['cls'][rid])
            rh=None if f not in ann else matches(raw['xyxy'][rid],ann[f]) if int(raw['cls'][rid])==2 else []
            ph=None if f not in ann or not len(rows) else matches(rows[0][1:5],ann[f]) if int(rows[0][7])==2 else []
            e=events.get(f);first=[];proposals=[];dupes=[]
            if e:
                first=[dict(tid=e['rows'][i]['tid'],state=e['rows'][i]['state'],raw_id=e['high_raw_ids'][j],original=e['C'][i][j],corrected=e['corrected'][i][j]) for i,j in e['actual_first'] if e['rows'][i]['tid']==tid or e['high_raw_ids'][j]==rid]
                proposals=[q for q in e['proposals'] if q['old_id']==tid or q['raw_id']==rid]
                dupes=[d for d in e['duplicates'] if tid in [d['tracked_id'],d['lost_id']]]
            history.append(dict(**a,raw_box=raw['xyxy'][rid].tolist(),confidence=float(raw['conf'][rid]),raw_hits=rh,public_hits=ph,public_box=None if not len(rows) else rows[0][1:5].tolist(),matrix_saved=e is not None,actual_first=first,proposals=proposals,duplicates=dupes))
        histories[str(tid)]=history
    for f in sorted(ann):
        if f>c['endpoint']:continue
        tracks=[]
        for a in public[(public[:,0]==f)&(public[:,7]==2)]:
            hh=matches(a[1:5],ann[f])
            if any(h['identity']==c['target'] for h in hh):tracks.append(dict(tid=int(a[5]),hits=hh,unique=len(hh)==1,box=a[1:5].tolist()))
        if tracks:owners.append(dict(frame=f,tracks=tracks,matrix_saved=f in events))
    births={tid:hh[0] for tid,hh in histories.items()}
    result.append(dict(**c,births=births,histories=histories,target_public_owners=owners,saved_event_frames=sorted(events),all_assignment_frames=sorted({a['frame'] for a in logs['assignments']}),annotation_frames=sorted(ann)))
verify();save('trace.json',dict(passed=True,cases=result,protocol_sha256=sha(OUT/'protocol.json')))
for c in result:
    print(c['video'])
    for tid,hh in c['histories'].items():print('ID',tid,'birth',hh[0]['frame'],'birth_matrix',hh[0]['matrix_saved'],'annotated',[(h['frame'],h['public_hits'],h['raw_hits']) for h in hh if h['raw_hits'] is not None])
    print('TARGET_OWNERS',[(x['frame'],[y['tid'] for y in x['tracks'] if y['unique']]) for x in c['target_public_owners']])
