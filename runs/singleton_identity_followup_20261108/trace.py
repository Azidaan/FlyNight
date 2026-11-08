from pathlib import Path
from collections import Counter
import json,hashlib,numpy as np
P=Path(__file__).resolve().parent;ROOT=P.parents[1];D=ROOT/'runs/singleton_id_fate_117_20261108/recovery1';INV=ROOT/'runs/bbox_confirmation_inventory_20261108/native'
def j(p):return json.loads(p.read_text(encoding='utf-8'))
sources={}
def bind(p):sources[p.relative_to(ROOT).as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest();return p
def overlap(a,b):
    a=np.asarray(a,float).reshape(-1,4);b=np.asarray(b,float);s=np.maximum(0,np.minimum(a[:,2:],b[2:])-np.maximum(a[:,:2],b[:2]));i=s.prod(-1);return i/np.maximum(1e-12,(a[:,2:]-a[:,:2]).prod(1)+(b[2:]-b[:2]).prod()-i)
summ=j(bind(D/'summary.json'));wanted=set(tuple(a) for a in summ['arms']['B']['improved_vs_B0']);evs=j(bind(D/'events.json'));assert hashlib.sha256((D/'events.json').read_bytes()).hexdigest()==summ['events_sha256']
f=j(bind(ROOT/'runs/singleton_birth_veto_transfer_20261108/frozen.json'));out=[]
names=['pedestrian','rider','car','truck','bus','train','motorcycle','bicycle']
for e in evs:
    vid=e['video']
    if (vid,e['key']) not in wanted:continue
    t,old,new=map(int,e['key'].split('|'))
    with np.load(bind(INV/f'{vid}.npz')) as z:n=z['outputs']
    with np.load(bind(ROOT/f['videos'][vid]['raw'])) as z:rf,b,cl,cf=[z[k] for k in ('frame_idx','xyxy','cls','conf')]
    events={x['frame']:x for x in j(bind(INV/f'{vid}.json'))['events']}
    gt={int(a['video_frame_index']):r for a,r in zip(j(bind(ROOT/f'runs/train_cache/{vid}/alignment.json'))['matches'],j(bind(ROOT/f'data/bdd100k/labels/box_track_20/train/{vid}.json')))}
    obs=e['arms']['B']['observations'];bad=next((r for r in obs if r['status']!='saved' or r['hijacked']),None);change=next((r for r in obs if r['status']=='changed_id'),None)
    r=dict(video=vid,edge=e['key'],target=e['target'],first_bad=bad,first_known_change=change,revocation=e['revocations']['B'])
    if bad:
        u=bad['frame'];g,=[x for x in gt[u]['labels'] if str(x['id'])==e['target']];box=[g['box2d'][k] for k in ('x1','y1','x2','y2')];idx=np.flatnonzero(rf==u);vals=overlap(b[idx],box);pub=n[n[:,0]==u];pubv=overlap(pub[:,1:5],box)
        r['first_bad_evidence']=dict(gt_box=box,gt_class=g['category'],gt_attributes=g.get('attributes'),raw_hits=[dict(raw_id=int(idx[i]),iou=float(vals[i]),conf=float(cf[idx[i]]),cls=int(cl[idx[i]])) for i in np.flatnonzero(vals>=.5)],public_hits=[dict(native_id=int(pub[i,5]),iou=float(pubv[i]),conf=float(pub[i,6]),cls=int(pub[i,7])) for i in np.flatnonzero(pubv>=.5)],best_raw_same_class=max((float(vals[i]) for i in range(len(idx)) if names[int(cl[idx[i]])]==g['category']),default=None))
    if change:
        successor,=change['target_native'];rows=n[n[:,5]==successor];first=int(rows[0,0]);previous=n[(n[:,5]==new)&(n[:,0]<first),0].astype(int);simultaneous=bool(np.any((n[:,5]==new)&(n[:,0]==first)));context=events.get(first)
        row=next((rr for rr in context['rows'] if rr['old_tid']==new),None) if context else None
        r['next_fragment']=dict(native=successor,first_public=first,prior_fragment_present=simultaneous,last_prior_public=int(previous[-1]) if len(previous) else None,age_from_last_public=first-int(previous[-1]) if len(previous) else None,exported_old_row=row,first_context_rows=[rr['old_tid'] for rr in context['rows']] if context else [],next_exists_before_link=first<=t)
    out.append(r)
assert len(out)==6 and all(r['revocation'] is None for r in out)
assert all(hashlib.sha256((ROOT/k).read_bytes()).hexdigest()==s for k,s in sources.items())
with (P/'evidence.json').open('x',encoding='utf-8') as g:json.dump(dict(sources=sources,protocol_sha256=hashlib.sha256((P/'protocol.md').read_bytes()).hexdigest(),rows=out),g,indent=2,ensure_ascii=False)
for r in out:
    print(r['edge'],'first bad',r['first_bad']['frame'] if r['first_bad'] else None,r.get('first_bad_evidence'), 'next',r.get('next_fragment'))
