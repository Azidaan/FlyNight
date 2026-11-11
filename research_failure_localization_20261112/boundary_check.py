"""Resolve the newly found exact-boundary ambiguity; threshold remains 0.8.
No runtime changes. Compare native float32 correction with real-number-intended
float64 1-0.2 at one element, preserving all other entries.
"""
from common import *
import numpy as np,lap
p=verify();s=read(OUT/'stage2.json');records=[]
def solve(c):
    _,x,y=lap.lapjv(c,extend_cost=True,cost_limit=.8)
    return sorted((i,int(j)) for i,j in enumerate(x) if j>=0)
for r in s['records']:
    if r['kind']!='own' or r['diagnostic_cost']<=.8:continue
    e=next(e for e in events(r['video'])['events'] if e['frame']==r['frame'])
    q=next(q for q in e['proposals'] if q['old_id']==r['old_id'] and q['raw_id']==r['raw_id']);ij=(q['row'],q['column'])
    a=np.array(e['corrected'],np.float32);a[ij]=np.float32(e['C'][ij[0]][ij[1]])-np.float32(.2)
    assert solve(a)==list(map(tuple,r['diagnostic_assignment']))
    b=a.astype(float);assert solve(b)==solve(a);b[ij]=float(e['C'][ij[0]][ij[1]])-.2
    after=solve(b)
    records.append(dict(video=r['video'],old_id=r['old_id'],frame=r['frame'],threshold=.8,float32_cost=float(a[ij]),float64_cost=float(b[ij]),before_selected=ij in solve(a),exact_boundary_selected=ij in after,
       before=solve(a),after=after,only_one_cost_changed=int(np.count_nonzero(a.astype(float)!=b))))
save('boundary_check.json',dict(records=records,passed=True,script_sha256=sha(__file__),scope='One cost arithmetic precision diagnostic only; no threshold or alpha change, no tracker. Equality is a possible optimum tie with unmatched, not automatically mandated.'))
print(json.dumps(records))
