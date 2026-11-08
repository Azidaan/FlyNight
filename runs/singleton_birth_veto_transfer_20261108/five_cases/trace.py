"""Read frozen arrays; never run tracker/inference. No classification fit."""
from pathlib import Path
from collections import defaultdict
import hashlib,json
import numpy as np
P=Path(__file__).resolve().parent;D=P.parent;ROOT=D.parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
f=read(D/'frozen.json');complete=read(D/'complete.json');assert complete['frozen_sha256']==sha(D/'frozen.json')
def check(p):
    assert sha(p)==f['inputs'][p.relative_to(ROOT).as_posix()]
    return p
def iou(a,b):
    a,b=np.asarray(a,float),np.asarray(b,float)
    i=np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])).prod()
    return float(i/((a[2:]-a[:2]).prod()+(b[2:]-b[:2]).prod()-i))
def g(a):
    a=np.array(a);return np.r_[(a[2:]+a[:2])/2,np.log(a[2:]-a[:2])]
def box(p):
    h=np.exp(p[2:])/2;return np.r_[p[:2]-h,p[:2]+h]
def one_max(ds):
    best=max(ds.values(),default=0)
    opts=[k for k in ds if ds[k]==best and best>0]
    return opts[0] if len(opts)==1 else None
cases=read(D/'result.json')['changed'];cases=[x for x in cases if x['kind']!='unknown']
assert len(cases)==5
rows=[]
for case in cases:
    vid=case['video'];t,old,new=map(int,case['key'].split('|'))
    p=D/'videos'/f'{vid}.json';assert sha(p)==complete['outputs'][vid];saved=read(p)['candidates'][case['key']]
    inv=ROOT/'runs/bbox_confirmation_inventory_20261108/native';js=read(check(inv/f'{vid}.json'))
    with np.load(check(inv/f'{vid}.npz')) as z:n=z['outputs']
    with np.load(check(ROOT/f['videos'][vid]['raw'])) as z:rf,b,cf,cl=[z[k] for k in ('frame_idx','xyxy','conf','cls')]
    event,=[x for x in js['events'] if x['frame']==t];r,=[x for x in event['rows'] if x['old_tid']==old]
    c,=[x for x in r['candidates'] if x['new_tid']==new];cls=r['row_data']['cls']
    hist={h['frame']:g(h['box']) for h in r['row_data']['history'][-4:]};ts=sorted(hist);last=ts[-1]
    vel=np.median([(hist[j]-hist[i])/(j-i) for i,j in zip(ts,ts[1:])],axis=0)
    assert last<t-1==c['birth']['frame']
    pred=[box(hist[last]+vel*(u-last)) for u in (t-1,t)]
    cb=[c['birth']['box'],c['box']];own_i=[iou(cb[i],pred[i]) for i in (0,1)]
    raw=[];owners=[];alts=[]
    for i,u in enumerate((t-1,t)):
        ids=np.flatnonzero(rf==u);high=ids[cf[ids]>=.25];low=ids[(cf[ids]>.1)&(cf[ids]<.25)]
        own={}
        for track in n[n[:,0]==u]:
            rid=int((high if track[6]>=.25 else low)[int(track[8])]);assert cf[rid]==track[6] and cl[rid]==track[7] and rid not in own
            own[rid]=int(track[5])
        raw.append([int(x) for x in ids[(cf[ids]>=.1)&(cl[ids]==cls)]])
        alts.append([x for x in raw[i] if iou(b[x],cb[i])<.5 and iou(b[x],pred[i])>own_i[i]])
        owners.append(own)
    assert alts[0]==saved['alternatives'] and alts[0] and owners[1][c['raw_id']]==new
    ds=(g(cb[1])[:2]-g(cb[0])[:2]);dot=float(ds@vel[:2]);err=float(np.linalg.norm(ds-vel[:2]))
    paths=[]
    for a in alts[0]:
        id0=owners[0].get(a)
        if id0 is not None:
            a1=next((x for x in raw[1] if owners[1].get(x)==id0),None);method='same_public_id'
        else:
            a1=one_max({x:iou(b[a],b[x]) for x in raw[1]});method='reciprocal_iou'
            if a1 is not None and one_max({x:iou(b[a1],b[x]) for x in raw[0]})!=a:a1=None
        if a1==c['raw_id'] or (a1 is not None and owners[1].get(a1)==new):a1=None
        paths.append(dict(raw_birth=a,public_birth=id0,raw_now=a1,method=method,birth_iou=iou(b[a],pred[0]),confidence=float(cf[a]),current_iou=iou(b[a1],pred[1]) if a1 is not None else None,current_public=owners[1].get(a1),both_warn=a1 in alts[1] if a1 is not None else False,motion_error=float(np.linalg.norm((g(b[a1].astype(float))[:2]-g(b[a].astype(float))[:2])-vel[:2])) if a1 is not None else None))
    flags=dict(now_warn=bool(alts[1]),owned_at_birth=any(p['public_birth'] is not None for p in paths),same_owner_warn=any(p['method']=='same_public_id' and p['both_warn'] for p in paths),opposed=(dot<0) if np.linalg.norm(ds)>0 and np.linalg.norm(vel[:2])>0 else None,linked_both=any(p['both_warn'] for p in paths),motion_better=any(p['motion_error']<err for p in paths if p['raw_now'] is not None) if any(p['raw_now'] is not None for p in paths) else None)
    labels=read(check(ROOT/'runs/bbox_singleton_pilot_20261108/labels'/f'{vid}.json'))['rows'][case['key']]
    assert ('correct' if labels['positive'] else 'foreign')==case['kind'] and labels['known']
    rows.append(dict(video=vid,t=t,old=old,new=new,kind=case['kind'],geometry=saved['score'],cls=cls,confidence=c['confidence'],source_frames=ts,own_iou=own_i,motion_old=vel[:2].tolist(),motion_new=ds.tolist(),motion_error=err,alternatives_birth=paths,alternatives_now=[dict(index=x,owner=owners[1].get(x),iou=iou(b[x],pred[1])) for x in alts[1]],flags=flags,labels_for_evaluation_only=labels))
assert len(rows)==5
with (P/'evidence.json').open('x',encoding='utf-8') as out:json.dump(dict(protocol_sha=sha(P/'protocol.md'),source_sha=sha(Path(__file__)),source_result_sha=sha(D/'result.json'),rows=rows),out,indent=2,allow_nan=False)
for r in rows:
    print(r['old'],r['new'],r['kind'],r['flags'],'iou',r['own_iou'],'err',r['motion_error'],'paths',r['alternatives_birth'],'now',r['alternatives_now'])
