"""Sparse-ground-truth public-ID outcomes and exact return detection provenance."""
import os,sys,json,hashlib,subprocess
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
from collections import defaultdict,Counter
from datetime import datetime
P=Path(__file__).resolve().parent;ROOT=P.parents[1];D=P.parent/'singleton_birth_veto_transfer_20261108';T=P.parent/'singleton_two_frames_46_20261108';INV=P.parent/'bbox_confirmation_inventory_20261108/native';DEV=P.parent/'singleton_box_history_pairs_20261108/conflict_review'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,x):
    with p.open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
def freeze():
    old=read(D/'frozen.json');end=read(D/'complete.json');assert sha(D/'frozen.json')==end['frozen_sha256']
    files=[P/'protocol.md',P/'analyze.py',P/'verify.py',D/'result.json',D/'frozen.json',D/'complete.json',T/'result.json',T/'decisions.json']
    for v in old['videos']:
        p=D/'videos'/f'{v}.json';assert sha(p)==end['outputs'][v]
        files.extend([p,INV/f'{v}.npz',INV/f'{v}.json',ROOT/old['videos'][v]['raw'],ROOT/f'runs/bbox_singleton_pilot_20261108/labels/{v}.json',ROOT/f'runs/train_cache/{v}/alignment.json',ROOT/f'data/bdd100k/labels/box_track_20/train/{v}.json'])
    save(P/'frozen.json',dict(time=datetime.now().astimezone().isoformat(),head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),videos=old['videos'],files={p.relative_to(ROOT).as_posix():sha(p) for p in files}))
def main():
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    f=read(P/'frozen.json');assert all(sha(ROOT/k)==v for k,v in f['files'].items())
    names=['pedestrian','rider','car','truck','bus','train','motorcycle','bicycle']
    def overlap(a,b):
        a=np.asarray(a,float).reshape(-1,4);b=np.asarray(b,float).reshape(-1,4)
        s=np.maximum(0,np.minimum(a[:,None,2:],b[None,:,2:])-np.maximum(a[:,None,:2],b[None,:,:2]));i=s.prod(-1)
        return i/np.maximum(1e-12,(a[:,2:]-a[:,:2]).prod(1)[:,None]+(b[:,2:]-b[:,:2]).prod(1)[None,:]-i)
    two=read(T/'decisions.json');allrows=[];checks=Counter()
    for vid,meta in f['videos'].items():
        prev=read(D/'videos'/f'{vid}.json');links=prev['arms']['H']['links']
        if not links:continue
        cand=prev['candidates'];data=read(INV/f'{vid}.json');ev={e['frame']:e for e in data['events']}
        labs=read(ROOT/f'runs/bbox_singleton_pilot_20261108/labels/{vid}.json');edge_labels=labs['rows']
        with np.load(INV/f'{vid}.npz') as z:n=z['outputs']
        with np.load(ROOT/meta['raw']) as z:rf,rb,rs,rc=[z[k] for k in ('frame_idx','xyxy','conf','cls')]
        idx=np.argsort(rf,kind='stable');sorted_frames=rf[idx]
        def provenance(frame,row):
            lo,hi=np.searchsorted(sorted_frames,[frame,frame+1]);ids=idx[lo:hi];domain=ids[rs[ids]>=.25] if row[6]>=.25 else ids[(rs[ids]>.1)&(rs[ids]<.25)]
            rid=int(domain[int(row[8])]);assert rs[rid]==row[6] and rc[rid]==row[7]
            return dict(raw_index=rid,confidence=float(row[6]),cls=int(row[7]),box=row[1:5].tolist(),raw_box=rb[rid].tolist())
        counts=np.bincount(n[:,0].astype(int),minlength=data['frames']);offset=np.r_[0,counts.cumsum()]
        def output(t):return n[offset[t]:offset[t+1]]
        gt={};assigned={};multiplicity={}
        align=read(ROOT/f'runs/train_cache/{vid}/alignment.json')['matches'];annotations=read(ROOT/f'data/bdd100k/labels/box_track_20/train/{vid}.json');assert len(align)==len(annotations)
        for m,a in zip(align,annotations):
            t=int(m['video_frame_index']);assert t not in gt
            objs=[l for l in a.get('labels') or [] if l['category'] in names and not (l.get('attributes') or {}).get('crowd',False)]
            gt[t]={str(l['id']):l for l in objs};matches={};dups=defaultdict(list);out=output(t)
            for cls in set(map(int,out[:,7])):
                ii=np.flatnonzero(out[:,7]==cls);jj=[i for i,o in enumerate(objs) if o['category']==names[cls]]
                if not jj:continue
                ious=overlap(out[ii,1:5],[[objs[j]['box2d'][k] for k in ('x1','y1','x2','y2')] for j in jj]);masked=np.where(ious>=.5,ious,0);ri,ci=linear_sum_assignment(-masked)
                for r,c in zip(ri,ci):
                    if masked[r,c]>0:matches[int(out[ii[r],5])]=str(objs[jj[c]]['id'])
                for r,c in zip(*np.where(ious>=.5)):dups[str(objs[jj[c]]['id'])].append(int(out[ii[r],5]))
            assigned[t]=matches;multiplicity[t]=dups
        for a in labs['new_needed_frame_assignments']:
            t=a['frame'];assert [[int(r[5]),assigned[t].get(int(r[5]))] for r in output(t)]==a['matches'];checks['saved_annotation_frames']+=1
        arms={**prev['arms'],'B':two[vid]['arm'],'B0':dict(links={},revocations={})}
        maps={}
        for arm,log in arms.items():
            at=defaultdict(list);rv=defaultdict(list);alias={};points={}
            for k,root in log['links'].items():at[int(k.split('|')[0])].append((k,root))
            for k,r in log['revocations'].items():rv[r['frame']].append(k)
            for t in range(data['frames']):
                for k in rv[t]:
                    native_new=int(k.split('|')[2]);assert alias[native_new]==log['links'][k];del alias[native_new]
                for k,root in at[t]:alias[int(k.split('|')[2])]=root
                if t in gt:points[t]=alias.copy()
                ids=[alias.get(int(i),int(i)) for i in output(t)[:,5]];assert len(ids)==len(set(ids))
            maps[arm]=points
        for k,root in links.items():
            t,old,new=map(int,k.split('|'));r=next(r for r in ev[t]['rows'] if r['old_tid']==old);c=next(c for c in r['candidates'] if c['new_tid']==new)
            label=edge_labels[k];anchors=label['target_anchors'];target=label['target'];assert (target is not None)==bool(anchors)
            if target is not None:assert all(i==target and assigned[fr].get(old)==target for fr,i in anchors)
            anchor=max((fr for fr,i in anchors),default=None)
            byarm={};future=sorted(fr for fr in gt if fr>t and target in gt[fr]) if target else []
            for arm in arms:
                wanted=maps[arm][anchor].get(old,old) if anchor is not None else None
                observations=[]
                for fr in future:
                    matches=assigned[fr];mp=maps[arm][fr];target_ids=[i for i,g in matches.items() if g==target];actual=[mp.get(i,i) for i in target_ids]
                    owner=[i for i in output(fr)[:,5].astype(int) if mp.get(int(i),int(i))==wanted]
                    owner_gt=matches.get(owner[0]) if owner else None
                    status='saved' if wanted in actual else 'changed_id' if actual else 'unmatched_target'
                    observations.append(dict(frame=fr,status=status,target_native=target_ids,target_public=actual,number_owner_native=owner[0].item() if owner and isinstance(owner[0],np.generic) else owner[0] if owner else None,number_owner_gt=owner_gt,hijacked=owner_gt is not None and owner_gt!=target,duplicate_native=multiplicity[fr].get(target,[])))
                first=observations[:2]
                state='unknown_prior' if target is None else 'insufficient_future' if len(first)<2 else 'hijacked' if any(u['hijacked'] for u in first) else 'saved_both' if all(u['status']=='saved' for u in first) else 'changed_id' if any(u['status']=='changed_id' for u in first) else 'unmatched_target'
                byarm[arm]=dict(wanted=wanted,state=state,observations=observations,all_counts=dict(Counter(u['status'] for u in observations)),root_matches_transfer=(wanted==root) if wanted is not None else None)
            old_times=n[(n[:,5]==old)&(n[:,0]>t),0].astype(int);new_times=set(n[(n[:,5]==new)&(n[:,0]>=t),0].astype(int));ret=int(old_times[0]) if len(old_times) else None;joint=next((int(x) for x in old_times if x in new_times),None)
            returned=None
            if ret is not None:
                rr,=output(ret)[output(ret)[:,5]==old];returned=dict(frame=ret,delay=ret-t,**provenance(ret,rr),annotated=ret in gt,gt_identity=assigned.get(ret,{}).get(old),nearby_annotations=[dict(frame=fr,identity=assigned[fr].get(old),public_present=bool(np.any(output(fr)[:,5]==old))) for fr in sorted(gt) if ret-6<=fr<=ret+12])
            # Link warning detections to public owners exactly at birth/confirmation.
            warning=[]
            for u in (t-1,t):
                h={o['frame']:np.r_[(np.array(o['box'][:2])+o['box'][2:])/2,np.log(np.array(o['box'][2:])-o['box'][:2])] for o in r['row_data']['history'][-4:]};ts=sorted(h);last=ts[-1]
                vel=np.median([(h[b]-h[a])/(b-a) for a,b in zip(ts,ts[1:])],axis=0);pred=h[last]+(u-last)*vel;hh=np.exp(pred[2:])/2;pb=np.r_[pred[:2]-hh,pred[:2]+hh];b=c['birth']['box'] if u==t-1 else c['box'];own=overlap([b],[pb])[0,0];ii=np.flatnonzero((rf==u)&(rs>=.1)&(rc==r['row_data']['cls']));alts=[int(i) for i in ii if overlap([rb[i]],[b])[0,0]<.5 and overlap([rb[i]],[pb])[0,0]>own]
                owners={provenance(u,o)['raw_index']:int(o[5]) for o in output(u)}
                warning.append(dict(frame=u,alternatives=[dict(raw_index=i,public_native=owners.get(i)) for i in alts]))
            assert warning[0]['alternatives'] and prev['candidates'][k]['veto'] or not warning[0]['alternatives'] and not prev['candidates'][k]['veto']
            # Exact warning->return requires the returning native track to own that earlier detection; Lost old isn't public then.
            warning_owners={a['public_native'] for w in warning for a in w['alternatives'] if a['public_native'] is not None}
            lineage='no_return' if ret is None else 'no_warning' if not any(w['alternatives'] for w in warning) else 'exact_native_owner' if old in warning_owners else 'unknown_unowned_warning' if any(a['public_native'] is None for w in warning for a in w['alternatives']) else 'warning_owned_by_other_track'
            allrows.append(dict(video=vid,split=meta['split'],key=k,target=target,anchor=anchor,original_kind=prev['candidates'][k]['kind'],arms=byarm,return_info=returned,first_joint=joint,revocations={arm:log['revocations'].get(k) for arm,log in arms.items()},warning=warning,lineage=lineage,new_last_public=int(n[n[:,5]==new,0].max())))
    assert len(allrows)==117 and sum(r['revocations']['H'] is not None for r in allrows)==21
    assert all(sha(ROOT/k)==v for k,v in f['files'].items())
    save(P/'events.json',allrows);summary={}
    for arm in ('B0','H','V','B'):
        states=Counter(r['arms'][arm]['state'] for r in allrows);better=[];worse=[]
        for r in allrows:
            b0=r['arms']['B0']['state'];a=r['arms'][arm]['state']
            if b0 not in ('unknown_prior','insufficient_future') and a not in ('unknown_prior','insufficient_future'):
                if a=='saved_both' and b0!='saved_both':better.append([r['video'],r['key']])
                if a!='saved_both' and b0=='saved_both':worse.append([r['video'],r['key']])
        summary[arm]=dict(states=states,improved_vs_B0=better,worsened_vs_B0=worse)
    save(P/'summary.json',dict(arms=summary,events=117,target_known=sum(r['target'] is not None for r in allrows),returns=sum(r['return_info'] is not None for r in allrows),lineage=Counter(r['lineage'] for r in allrows),checks=checks,events_sha256=sha(P/'events.json'),frozen_sha256=sha(P/'frozen.json')))
    print(summary)
if __name__=='__main__':
    if 'freeze' in sys.argv:freeze()
    else:
        sys.path.insert(0,str(ROOT/'runs/flyvis_repairs_20261106/unification_20261107'));from system_memory_guard import guard_entrypoint
        guard_entrypoint(timeout=180);main()
