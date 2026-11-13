"""Independent scalar descriptor distances and augmented SciPy assignment objective."""
from common import *
os.environ['CUDA_VISIBLE_DEVICES']=''
import math,numpy as np
from scipy.optimize import linear_sum_assignment
p=read(OUT/'protocol.json'); result=read(OUT/'result.json'); runtime=read(PREV/'runtime.json'); observations=read(OUT/'descriptor_plan.json')['observations']; old=np.load(PREV/'descriptors.npz',allow_pickle=False); new=np.load(OUT/'descriptors.npz',allow_pickle=False)
def distance(a,b):
    c=math.fsum(float(x)*float(y) for x,y in zip(a,b))/math.sqrt(math.fsum(float(x)**2 for x in a)*math.fsum(float(y)**2 for y in b))
    return (1-max(-1,min(1,c)))/2
maxcost=maxobjective=0.; summaries=[]
previous=read(PREV/'result.json')['records']
for record in result['records']:
    e=next(e for e in runtime['events'] if e['frame']==record['frame']); ids=[r['tid'] for r in e['rows']]; rawids=[c['raw_id'] for c in e['candidates']]; actual=np.array(record['cost']); expected=np.array(e['corrected'],float)
    for i,row in enumerate(e['rows']):
        if record['policy']=='native': continue
        refs=record['references'][str(row['tid'])]
        # Independently validate selector decisions before using recorded provenance.
        hist=sorted(row['history'],key=lambda h:h['frame']); pol=record['policy']
        if row['tid'] not in [2179,2445] or pol=='latest': assert refs==[dict(frame=row['reference']['frame'],kind='filtered')]
        elif pol=='gallery_raw': assert [r['frame'] for r in refs]==[h['frame'] for h in hist] and all(r['kind']=='raw' for r in refs)
        elif pol in ['earliest_raw','earliest_filtered']: assert len(refs)==1 and refs[0]['frame']==hist[0]['frame'] and refs[0]['kind']==pol.split('_')[1]
        elif pol=='latest_raw': assert len(refs)==1 and refs[0]['frame']==hist[-1]['frame'] and refs[0]['kind']=='raw'
        elif pol.startswith('freeze2445_'):
            assert len(refs)==1 and refs[0]['frame']==(min(int(pol.rsplit('_',1)[1]),e['frame']-1) if row['tid']==2445 else row['reference']['frame'])
        elif pol=='swap_latest_negative': assert refs[0]['frame']==next(r for r in e['rows'] if r['tid']==(2445 if row['tid']==2179 else 2179))['reference']['frame'] and refs[0]['kind']=='swapped_filtered'
        assert all(r['frame']<e['frame'] for r in refs)
        vectors=[]
        for ref in refs:
            if 'raw_id' in ref:
                obs=next(o for o in observations if o['tid']==row['tid'] and o['frame']==ref['frame'] and o['raw_id']==ref['raw_id']); slot=obs[ref['kind']+'_slot']; vectors.append(new[slot[0]][slot[1]])
            else:
                other=row if ref['kind']!='swapped_filtered' else next(r for r in e['rows'] if r['tid']==(2445 if row['tid']==2179 else 2179)); slot=other['reference']['slot']; vectors.append(old[slot[0]][slot[1]])
        for j,c in enumerate(e['candidates']):
            slot=c['slot']; ap=min(distance(a,old[slot[0]][slot[1]]) for a in vectors); expected[i,j]+=record['alpha']*(ap-e['base'][i][j])
    maxcost=max(maxcost,float(abs(actual-expected).max())); assert maxcost<1e-6,maxcost
    if record['policy'] in ['native','latest']:
        name='baseline' if record['policy']=='native' else 'appearance_'+str(record['alpha'])
        prior=next(x for x in previous if x['frame']==e['frame'] and x['variant']==name)
        assert np.array_equal(actual,np.array(prior['cost'])) and sorted(map(tuple,record['pairs']))==sorted(map(tuple,prior['pairs']))
    pairs=set(map(tuple,record['pairs'])); assert len({a for a,b in pairs})==len(pairs)==len({b for a,b in pairs})
    n,k=actual.shape; aug=np.zeros((n+k,n+k)); aug[:n,:k]=actual; aug[:n,k:]=aug[n:,:k]=.4; ii,jj=linear_sum_assignment(aug); optimal=math.fsum(aug[i,j] for i,j in zip(ii,jj)); own=math.fsum(actual[ids.index(t),rawids.index(r)] for t,r in pairs)+.4*(n+k-2*len(pairs)); maxobjective=max(maxobjective,abs(optimal-own)); assert maxobjective<1e-9
    base=next(x for x in result['records'] if x['frame']==e['frame'] and x['policy']=='native' and x['alpha']==record['alpha']); good={x['tid'] for x in record['evaluation'] if x['outcome']=='correct'}; bad={x['tid'] for x in record['evaluation'] if x['outcome']=='wrong'}; oldgood={x['tid'] for x in base['evaluation'] if x['outcome']=='correct'}; oldbad={x['tid'] for x in base['evaluation'] if x['outcome']=='wrong'}
    summaries.append(dict(frame=e['frame'],alpha=record['alpha'],policy=record['policy'],correct=len(good),wrong=len(bad),lost_correct=sorted(oldgood-good),new_wrong=sorted(bad-oldbad),target2179=record['targets']['2179'],probe_costs=record['probe_costs'],changed_vs_native=sorted(pairs-set(map(tuple,base['pairs'])))))
for n,h in p['inputs'].items(): assert sha(ROOT/n)==h,n
save('audit.json',dict(passed=True,records=len(summaries),maximum_cost_error=maxcost,maximum_objective_error=maxobjective,summary=summaries,baseline_and_prior_appearance_exact=True,inputs_unchanged=True,scope='Independent scalar cost and SciPy assignment objective; label accounting uses previously verified exact-anchor evaluation helper, not a new independent GT audit.'))
print('HISTORY_AUDIT_PASSED',len(summaries),maxcost,maxobjective,flush=True)
