from common import *
import numpy as np,lap
from scipy.optimize import linear_sum_assignment
from collections import Counter
p=verify();assert read(OUT/'stage1.json')['passed'];out=[];maxerr=0.
def solve(c):
    _,x,y=lap.lapjv(c,extend_cost=True,cost_limit=.8)
    return sorted((i,int(j)) for i,j in enumerate(x) if j>=0)
def objective(c,m):return float(sum(float(c[i,j]) for i,j in m)+.4*(sum(c.shape)-2*len(m)))
def independent(c,m):
    n,k=c.shape;a=np.zeros((n+k,n+k));a[:n,:k]=c;a[:n,k:]=.4;a[n:,:k]=.4
    i,j=linear_sum_assignment(a);v=float(a[i,j].sum());err=abs(v-objective(c,m));assert err<1e-9,(err,v,objective(c,m));return err
for v in sorted({x['video'] for x in p['selected']}):
    d=events(v);lookup={e['frame']:e for e in d['events']}
    for r in [x for x in p['selected'] if x['video']==v]:
        e=lookup[r['frame']];q=next(q for q in e['proposals'] if q['old_id']==r['old_id'] and q['raw_id']==r['raw_id']);ij=(q['row'],q['column'])
        base=np.array(e['C'],np.float32);old=np.array(e['corrected'],np.float32);reconstructed=base.copy()
        for i,j in e['votes']:reconstructed[i,j]-=.2
        assert np.array_equal(old,reconstructed)
        before=solve(old);assert before==sorted(map(tuple,e['actual_first']))
        assert solve(old.copy())==before
        maxerr=max(maxerr,independent(old,before))
        altered=old.copy();altered[ij]=base[ij]
        if r['kind']=='own':altered[ij]-=.2
        mask=np.ones(base.shape,bool);mask[ij]=False;assert np.array_equal(old[mask],altered[mask])
        after=solve(altered);maxerr=max(maxerr,independent(altered,after))
        def describe(matches):return [dict(row=i,column=j,tid=e['rows'][i]['tid'],raw_id=e['high_raw_ids'][j],cost=float(altered[i,j])) for i,j in matches]
        out.append(dict(video=v,frame=r['frame'],old_id=r['old_id'],raw_id=r['raw_id'],kind=r['kind'],original_reason=r['reason'],
          previous_vote=list(ij) in e['votes'],diagnostic_vote=r['kind']=='own',gate_bypassed_for_diagnostic=not r['scored'],
          previous_cost=float(old[ij]),diagnostic_cost=float(altered[ij]),old_selected=ij in before,new_selected=ij in after,
          previous_assignment=before,diagnostic_assignment=after,added=describe(sorted(set(after)-set(before))),removed=describe(sorted(set(before)-set(after))),
          cost_changed=int(np.count_nonzero(old!=altered)),objective_before=objective(old,before),objective_after=objective(altered,after)))
    print('STAGE2',v,len(out),flush=True)
verify();save('stage2.json',dict(passed=True,records=out,identity_controls=29,independent_objective_max_error=maxerr,
 threshold=.8,alpha=.2,lap_version=lap.__version__,script_sha256=sha(__file__),
 summary=dict(Counter((r['kind']+'_'+str(r['new_selected'])) for r in out)),
 scope='Single fixed matrix only. Counterfactual permissions are diagnostic labels, not deployable predictions. Public lifecycle and future trajectories were not executed. Removing a bonus does not forbid a foreign edge.'))
journal('Второй этап: 29 исходных назначений воспроизведены; на копии каждой полной матрицы изменено только разрешение выбранной размеченной связи. Порог 0,8 и поправка 0,2 сохранены. Источник research_failure_localization_20261112/stage2.json; выводы ограничены одним назначением, без жизненного цикла трекера.')
print(json.dumps([{k:r[k] for k in ['old_id','frame','kind','original_reason','old_selected','new_selected','diagnostic_cost','added','removed']} for r in out],ensure_ascii=False))
