"""Explain saved rejections; reconstruct state without recomputing assignments."""
from pathlib import Path
from collections import Counter,defaultdict
import json,numpy as np,hashlib
P=Path(__file__).resolve().parent;R=P.parents[1];D=R/'runs/singleton_birth_veto_transfer_20261108';T=R/'runs/singleton_two_frames_46_20261108'
def j(p):return json.loads(p.read_text(encoding='utf-8'))
records=[];counts=Counter()
for v,dd in j(T/'decisions.json').items():
    d=j(D/'videos'/f'{v}.json');a=d['candidates'];log=dd['arm']
    ev={e['frame']:e for e in j(R/f'runs/bbox_confirmation_inventory_20261108/native/{v}.json')['events']}
    with np.load(R/f'runs/bbox_confirmation_inventory_20261108/native/{v}.npz') as z:n=z['outputs']
    frameids=defaultdict(set)
    for t,i in n[:,[0,5]].astype(int):frameids[int(t)].add(int(i))
    points=defaultdict(list);links=defaultdict(list);rev=defaultdict(list)
    for k,row in a.items():
        if row['score'] is not None and row['score']<=1 and not dd['confirmation_alternatives_on_birth_warnings'].get(k):points[row['t']].append(k)
    for k,root in log['links'].items():links[int(k.split('|')[0])].append((k,root))
    for k,r in log['revocations'].items():rev[r['frame']].append(k)
    alias={};owner={};seen=set();last={};recorded={}
    for t in range(int(n[:,0].max())+1):
        for k in rev[t]:del alias[int(k.split('|')[2])]
        cur=frameids[t];emitted={alias.get(i,i) for i in cur}
        for k in points[t]:
            x=a[k];old,new=x['old'],x['new'];root=alias.get(old,old);why=[]
            if new not in cur-seen or new not in ev[t]['newly_confirmed']:why.append('not_first_public')
            if old not in seen:why.append('old_never_public')
            if old in cur:why.append('old_currently_public')
            if owner.get(root)!=old:why.append('canonical_owner_changed')
            if root in emitted:why.append('canonical_currently_public')
            if old==new:why.append('self')
            accepted=k in log['links']
            if accepted:assert not why
            elif not why:why.append('global_assignment')
            counts.update(why)
            records.append(dict(video=v,key=k,kind=x['kind'],accepted=accepted,reasons=why,root=root,canonical_owner=owner.get(root),owner_last_frame=last.get(owner.get(root)),root_public_native=[i for i in cur if alias.get(i,i)==root],old_last_public=last.get(old)))
        for k,root in links[t]:alias[int(k.split('|')[2])]=root
        for i in cur:owner[alias.get(i,i)]=i;last[i]=t
        seen.update(cur)
assert sum(r['accepted'] for r in records)==95
with (P/'ownership.json').open('x',encoding='utf-8') as f:json.dump(dict(reasons=counts,rows=records),f,indent=2)
print(counts)
for r in records:
    if not r['accepted']:print(r)
