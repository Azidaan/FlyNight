"""Counterfactual public-ID bookkeeping on frozen outputs; no tracker replay."""
from pathlib import Path
import hashlib,json
from collections import defaultdict,Counter
import numpy as np
p=Path(__file__).resolve().parent;root=p.parents[2];h=root/'runs/singleton_box_history_pairs_20261108/recovery1'
def read(x):return json.loads(x.read_text(encoding='utf-8'))
d=read(p/'evidence.json');assert len(d['pairs'])==49
assert read(h/'audit.json')['score_and_selection_agree']  # Every H-admissible edge was actually selected.
def label(r):return (r['video'],r['frame'],r['old'],r['new'])
def subset(r):return 'new_conflicts' if r['revocation'] and not r['baseline'] else ('restored' if not r['baseline'] else 'old_correct') if r['kind']=='correct' else 'other_unknown'
res={}
for mode in ('both','birth_only'):
    yes=[];no=[];rev=[];roots={}
    for vid in sorted(set(r['video'] for r in d['pairs'])):
        rows=[r for r in d['pairs'] if r['video']==vid]
        f=root/f'runs/singleton_working_domain_20261108/native/{vid}.npz'
        assert hashlib.sha256(f.read_bytes()).hexdigest()==d['inputs'][f.relative_to(root).as_posix()]
        with np.load(f) as z:out=z['outputs']
        frames=defaultdict(set)
        for t,i in out[:,[0,5]].astype(int):frames[t].add(i)
        alias={};owner={};seen=set()
        for t,ids in sorted(frames.items()):
            groups=defaultdict(set)
            for i in ids:groups[alias[i][0] if i in alias else i].add(i)
            for old,group in groups.items():
                if len(group)<2:continue
                assert old in group
                for i in group-{old}:
                    _,r=alias.pop(i);rev.append(dict(pair=label(r),kept=int(old),revoked=int(i),at=int(t)))
            possible=[r for r in rows if r['frame']==t];emitted={alias[i][0] if i in alias else i for i in ids}
            # No competing admissible edges in these saved decisions; verify rather than assuming.
            assert len({r['new'] for r in possible})==len(possible) and len({r['old'] for r in possible})==len(possible)
            for r in possible:
                a=r['alternatives'];veto=bool(a[0]['raw']) and (mode=='birth_only' or bool(a[1]['raw']))
                if veto:no.append(r);continue
                i,n=r['old'],r['new'];old=alias[i][0] if i in alias else i
                assert n in ids-seen and i in seen-ids and owner.get(old)==i and old not in emitted
                alias[n]=(old,r);emitted.add(old);yes.append(r);roots[label(r)]=int(old)
            newids=[alias[i][0] if i in alias else i for i in ids];assert len(newids)==len(set(newids))
            for i in ids:owner[alias[i][0] if i in alias else i]=i
            seen.update(ids)
    assert len(yes)+len(no)==49
    res[mode]=dict(vetoed_by_group=dict(Counter(subset(r) for r in no)),remaining_by_group=dict(Counter(subset(r) for r in yes)),vetoed=[label(r) for r in no],remaining_revocations=rev,canonical_ids_of_restored=[(label(r),roots[label(r)]) for r in yes if subset(r)=='restored'])
with (p/'veto_check.json').open('x',encoding='utf-8') as f:json.dump(res,f,indent=2)
for k,v in res.items():print(k,v['vetoed_by_group'],v['remaining_by_group'],v['remaining_revocations'])
