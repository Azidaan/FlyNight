"""Saved-data diagnostic. Later observations are appended separately from signals."""
import json,hashlib
from pathlib import Path
from collections import Counter
import numpy as np
from scipy.optimize import linear_sum_assignment
P=Path(__file__).resolve().parent
ROOT=P.parents[2]
D=ROOT/'runs/singleton_working_domain_20261108'
O=P.parent/'recovery1'
used={}
def bind(p):
    used[p.relative_to(ROOT).as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest()
    return p
def j(p):return json.loads(bind(p).read_text(encoding='utf-8-sig'))
def val(b):
    a=np.asarray(b,dtype=float);return np.r_[(a[:2]+a[2:])/2,np.log(a[2:]-a[:2])]
def box(v):return np.r_[v[:2]-np.exp(v[2:])/2,v[:2]+np.exp(v[2:])/2]
def iou(a,b):
    a,b=np.asarray(a),np.asarray(b);i=np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])).prod();return float(i/((a[2:]-a[:2]).prod()+(b[2:]-b[:2]).prod()-i))
def subset(outs,f):return outs[outs[:,0]==f]
def keep(z):return dict(id=int(z[5]),cls=int(z[7]),box=z[1:5].tolist())
cases=j(O/'audit.json')['revocations']['H']; pairs=j(O/'pairs.json');base=j(O/'result.json'); assert used[(O/'pairs.json').relative_to(ROOT).as_posix()]==base['pairs_sha256']
names=['pedestrian','rider','car','truck','bus','train','motorcycle','bicycle']
results=[]
for vid in sorted({a['video'] for a in pairs if a['selected']['H']}):
    video=[a for a in pairs if a['video']==vid and a['selected']['H']]
    cache=ROOT/'runs/train_cache'/vid
    with np.load(bind(cache/'detections/raw_video.npz')) as z:rf,rb,rc,rs=[z[k] for k in ['frame_idx','xyxy','cls','conf']]
    with np.load(bind(D/'native'/f'{vid}.npz')) as z:o=z['outputs']
    e={a['frame']:a for a in j(D/'native'/f'{vid}.json')['events']}
    labels=j(D/'labels'/f'{vid}.json')['rows']
    case_video=any(c['video']==vid and not c['baseline'] for c in cases)
    if case_video:
        gt={int(m['video_frame_index']):f for m,f in zip(j(cache/'alignment.json')['matches'],j(ROOT/f'data/bdd100k/labels/box_track_20/train/{vid}.json'))}
    for a in video:
        t,old,new=[a[k] for k in ('frame','old_tid','new_tid')]
        r,=[r for r in e[t]['rows'] if r['old_tid']==old];c,=[c for c in r['candidates'] if c['new_tid']==new];cls=r['row_data']['cls']
        hist={x['frame']:val(x['box']) for x in r['row_data']['history']};ts=sorted(hist);last=ts[-1];vel=np.median([(hist[y]-hist[x])/(y-x) for x,y in zip(ts,ts[1:])],axis=0)
        before=o[(o[:,5]==old)&(o[:,0]<t),0].astype(int); gaps=[(int(x),int(y)) for x,y in zip(before,before[1:]) if y>x+1]
        assert last==int(before[-1]) and last<t-1 and old not in subset(o,t)[:,5]
        b1=c['birth']['box'];b2=c['box'];comp=[]
        for f,b in [(t-1,b1),(t,b2)]:
            pred=box(hist[last]+(f-last)*vel);own=iou(b,pred)
            ii=np.flatnonzero((rf==f)&(rc==cls)&(rs>=.1))
            alt=[dict(raw_index=int(i),conf=float(rs[i]),iou=iou(rb[i],pred)) for i in ii if iou(rb[i],b)<.5 and iou(rb[i],pred)>own]
            others=[int(z[5]) for z in subset(o,f) if int(z[5]) not in (old,new) and int(z[7])==cls and iou(z[1:5],b)<.5 and iou(z[1:5],pred)>own]
            comp.append(dict(frame=f,pred_iou=own,raw=alt,public=others))
        dot=float(vel[:2]@(val(b2)[:2]-val(b1)[:2]))
        flags=dict(gap12=any(x>=t-12 for x,y in gaps),gap_ever=bool(gaps),raw_both=all(x['raw'] for x in comp),public_both=bool(set(comp[0]['public'])&set(comp[1]['public'])),opposed=dot<0)
        when=next((v['revoked_at'] for v in cases if v['video']==vid and v['frame']==t and v['new_tid']==new),None)
        row=dict(video=vid,frame=t,old=old,new=new,kind=a['kind'],baseline=a['baseline'],revocation=when,H=a['scores']['H']['score'],native_cost=c['cost'],forward_iou=c['forward_iou'],det_conf=c['confidence'],past_gaps=gaps,old_public_count=len(before),alternatives=comp,motion_dot=dot,flags=flags)
        if when is not None and not a['baseline']:
            old_o=o[o[:,5]==old];new_o=o[o[:,5]==new];ot=set(map(int,old_o[:,0]));nt=set(map(int,new_o[:,0]))
            row['after']=dict(old_reappears=min(x for x in ot if x>=t),first_joint=min(x for x in ot&nt if x>=t),frames_new_visible_to_collision=sorted(x for x in nt if t<=x<=when),at_collision=[keep(z) for z in subset(o,when) if int(z[5]) in (old,new)],joint_frames_until_end=sorted(ot&nt))
            joint=row['after']['at_collision'];assert len(joint)==2 and row['after']['first_joint']==when
            row['after']['iou_at_collision']=iou(joint[0]['box'],joint[1]['box'])
            row['fragment_label']=labels[f'{t}|{old}|{new}']
            row['after']['annotated']=[]
            for f in sorted(x for x in gt if last-12<=x<=when+12):
                obs=subset(o,f); objs=[l for l in gt[f].get('labels') or [] if l['category'] in names and not (l.get('attributes') or {}).get('crowd',False)]
                m=np.zeros((len(obs),len(objs)))
                for i,z in enumerate(obs):
                    for k,l in enumerate(objs):
                        if names[int(z[7])]==l['category']:
                            v=iou(z[1:5],[l['box2d'][s] for s in ('x1','y1','x2','y2')]);m[i,k]=v if v>=.5 else 0
                ii,jj=linear_sum_assignment(-m); hits={int(obs[i,5]):objs[k]['id'] for i,k in zip(ii,jj) if m[i,k]>0}
                row['after']['annotated'].append(dict(frame=f,visible=[int(i) for i in obs[:,5] if int(i) in (old,new)],assigned={i:hits[i] for i in (old,new) if i in hits}))
        results.append(row)
def group(r):
    if r['revocation'] and not r['baseline']:return 'new_conflict'
    if r['kind']=='correct':return 'previous_correct' if r['baseline'] else 'restored_correct'
    return 'other_unknown'
summary={}
for k in results[0]['flags']:
    summary[k]=dict(Counter(group(r) for r in results if r['flags'][k]))
g=Counter(map(group,results));assert g==dict(new_conflict=4,restored_correct=7,previous_correct=6,other_unknown=32)
for k,v in base['input_sha256'].items():
    if k in used:assert used[k]==v
with (P/'evidence.json').open('x',encoding='utf-8') as f:json.dump(dict(inputs=used,scope=dict(g),flags=summary,pairs=results),f,indent=2,allow_nan=False)
print(summary)
for r in results:
    if 'after' in r:print(r['old'],r['new'],r['H'],r['flags'],r['after'],r['fragment_label'])
