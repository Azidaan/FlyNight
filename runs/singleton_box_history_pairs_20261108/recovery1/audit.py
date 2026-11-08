"""Independent scalar geometry and collision accounting from saved ID streams."""
from pathlib import Path
from collections import Counter,defaultdict
from statistics import median
from math import log,exp,isclose
import hashlib,json,numpy as np
p=Path(__file__).resolve().parent
d=p.parents[1]/'singleton_working_domain_20261108'
def read(path):return json.loads(path.read_text(encoding='utf-8'))
rows=read(p/'pairs.json');report=read(p/'result.json')
assert hashlib.sha256((p/'pairs.json').read_bytes()).hexdigest()==report['pairs_sha256']
def values(b):
    x,y,X,Y=b
    return [(X+x)/2,(Y+y)/2,log(X-x),log(Y-y)]
rev=defaultdict(list);checked=0
for video in set(k.stem for k in (d/'outcomes').glob('*.json')):
    rr=[r for r in rows if r['video']==video]
    raw={e['frame']:e for e in read(d/'native'/f'{video}.json')['events']}
    for r in rr:
        t=r['frame']; track,=[x for x in raw[t]['rows'] if x['old_tid']==r['old_tid']];cand,=[c for c in track['candidates'] if c['new_tid']==r['new_tid']]
        hist={h['frame']:values(h['box']) for h in track['row_data']['history'][-4:]};times=sorted(hist);last=times[-1]
        for arm in ('H','S','D'):
            scores=[]
            for f,b in [(t-1,cand['birth']['box']),(t,cand['box'])]:
                if arm=='S' and f!=t:continue
                pred=[]
                for i in range(4):
                    v=median((hist[z][i]-hist[a][i])/(z-a) for a,z in zip(times,times[1:])) if arm=='H' else 0
                    pred.append(hist[last][i]+v*(f-last))
                obs=values(b);s=[2*abs(obs[i]-pred[i])/exp(pred[i+2]) for i in range(2)]+[abs(obs[i]-pred[i])/log(2) for i in (2,3)];scores.extend(s)
            assert isclose(max(scores),r['scores'][arm]['score'],rel_tol=1e-9,abs_tol=1e-9)
            assert (max(scores)<=1)==r['selected'][arm];checked+=1
    with np.load(d/'native'/f'{video}.npz') as z:out=z['outputs']
    frames=defaultdict(set)
    for t,i in out[:,[0,5]].astype(int):frames[t].add(i)
    for arm in ('H','baseline'):
        links={};owner={};born={};at=defaultdict(list)
        for r in rr:
            if (r['baseline'] if arm=='baseline' else r['selected'][arm]):at[r['frame']].append(r)
        for t,ids in sorted(frames.items()):
            groups=defaultdict(list)
            for i in ids:groups[links[i][0] if i in links else i].append(i)
            for root,group in groups.items():
                if len(group)<2:continue
                keep=root if root in group else owner.get(root)
                if keep not in group:keep=min(group,key=lambda i:(born.get(i,float('inf')),i))
                for i in group:
                    if i!=keep:
                        _,r=links.pop(i);rev[arm].append({k:r[k] for k in ('video','frame','old_tid','new_tid','kind','baseline')}|{'revoked_at':int(t)})
            for r in at[t]:
                old,new=r['old_tid'],r['new_tid'];links[new]=(links[old][0] if old in links else old,r)
            assert len({links[i][0] if i in links else i for i in ids})==len(ids)
            for i in ids:owner[links[i][0] if i in links else i]=i;born.setdefault(i,t)
assert checked==279 and len(rev['H'])==report['arms']['H']['revoked']
value={'passed':True,'scalar_scores_checked':checked,'revocations':rev,'score_and_selection_agree':True}
with (p/'audit.json').open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False)
print(json.dumps(value))
