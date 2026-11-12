"""Independent scalar weight formulas, solver objective and exact-ID accounting."""
from common import *
os.environ['CUDA_VISIBLE_DEVICES']=''
import math,numpy as np
from scipy.optimize import linear_sum_assignment
p=read(OUT/'protocol.json');r=read(OUT/'result.json');runtime=read(OUT/'runtime.json');plan=read(OUT/'evaluation_plan.json');clock=read(ROOT/plan['clock'])['anchors'];lab=read(ROOT/plan['labels']);by={int(x.get('frameIndex',i)):x for i,x in enumerate(lab)}
with np.load(ROOT/plan['detector_file'],allow_pickle=False) as z:raw_boxes=z['xyxy'].copy()
def labels(frame,box):
    if str(frame) not in clock:return None
    anchor=clock[str(frame)];objects=by[anchor['frameIndex']];assert objects['name']==anchor['name'];out=[]
    for item in objects.get('labels') or []:
        if not item.get('box2d') or (item.get('attributes') or {}).get('crowd',False):continue
        other=[item['box2d'][k] for k in ['x1','y1','x2','y2']];inter=math.prod(max(0,min(box[k+2],other[k+2])-max(box[k],other[k])) for k in [0,1]);area=math.prod(box[k+2]-box[k] for k in [0,1])+math.prod(other[k+2]-other[k] for k in [0,1])-inter
        if area>0 and inter/area>=.5:out.append(str(item['id']))
    return out
def cosine(a,b):return max(-1,min(1,math.fsum(float(x)*float(y) for x,y in zip(a,b))/math.sqrt(math.fsum(float(x)**2 for x in a)*math.fsum(float(y)**2 for y in b))))
maximum=objective_error=0.;summary=[]
with np.load(OUT/'descriptors.npz',allow_pickle=False) as bank:
    for e in runtime['events']:
        ids=[x['tid'] for x in e['rows']];rawids=[x['raw_id'] for x in e['candidates']];expected={}
        for tid in ids:
            identities=[];invalid=False
            for a in plan['past_anchored_assignments']:
                if not e['frame']-12<=a['frame']<e['frame']:continue
                rid=a['assigned'].get(str(tid))
                if rid is None:continue
                h=labels(a['frame'],raw_boxes[int(rid)]);invalid|=h is None or len(h)!=1
                if h and len(h)==1:identities.append(h[0])
            expected[tid]=identities[0] if identities and not invalid and len(set(identities))==1 else None
        rows=[x for x in r['records'] if x['frame']==e['frame']];base=np.asarray(e['base'],float);original=np.asarray(e['corrected'],float);baseline=next(x for x in rows if x['variant']=='baseline');oldpairs=set(map(tuple,baseline['pairs']))
        for record in rows:
            v=next(x for x in p['variants'] if x['name']==record['variant']);actual=np.array(record['cost']);target=original.copy();permutation=np.roll(np.arange(len(rawids)),1)
            for i,row in enumerate(e['rows']):
                for j,c in enumerate(e['candidates']):
                    if v['kind']=='confidence':
                        recovered=(1-base[i,j])/c['confidence'];target[i,j]=original[i,j]+1-recovered*c['confidence']**v['confidence_power']-base[i,j]
                    elif v['kind'] not in ['identity']:
                        a=row['reference']['slot'];b=e['candidates'][int(permutation[j]) if v['kind']=='permuted' else j]['slot'];appearance=(1-cosine(bank[a[0]][a[1]],bank[b[0]][b[1]]))/2;target[i,j]=original[i,j]+v['appearance_weight']*(appearance-base[i,j])
            maximum=max(maximum,float(abs(actual-target).max()));assert maximum<1e-6
            pairs=set(map(tuple,record['pairs']));assert len({x[0] for x in pairs})==len(pairs)==len({x[1] for x in pairs})
            n,k=actual.shape;aug=np.zeros((n+k,n+k));aug[:n,:k]=actual;aug[:n,k:]=.4;aug[n:,:k]=.4;ii,jj=linear_sum_assignment(aug);optimal=math.fsum(aug[a,b] for a,b in zip(ii,jj));own=math.fsum(actual[ids.index(t),rawids.index(raw)] for t,raw in pairs)+.4*(n+k-2*len(pairs));objective_error=max(objective_error,abs(optimal-own));assert objective_error<1e-9
            for x in record['evaluation']:
                hits=labels(e['frame'],e['candidates'][rawids.index(x['raw_id'])]['box']);identity=expected[x['tid']];outcome='unknown' if identity is None or hits is None or len(hits)!=1 else 'correct' if hits[0]==identity else 'wrong';assert x['outcome']==outcome and x['expected_identity']==identity and x['current_hits']==hits
            assert set(map(tuple,record['added']))==pairs-oldpairs and set(map(tuple,record['removed']))==oldpairs-pairs
            oldgood={x['tid'] for x in baseline['evaluation'] if x['outcome']=='correct'};oldbad={x['tid'] for x in baseline['evaluation'] if x['outcome']=='wrong'};good={x['tid'] for x in record['evaluation'] if x['outcome']=='correct'};bad={x['tid'] for x in record['evaluation'] if x['outcome']=='wrong'};assert record['lost_correct_tids']==sorted(oldgood-good) and record['new_wrong_tids']==sorted(bad-oldbad)
            summary.append(dict(frame=e['frame'],variant=record['variant'],correct=len(good),wrong=len(bad),unknown=sum(x['outcome']=='unknown' for x in record['evaluation']),lost_correct=len(oldgood-good),new_wrong=len(bad-oldbad),changed=bool(record['added'] or record['removed']),target2179=record['target2179'],raw109699_owner=record['raw109699_owner']))
assert sha(OUT/'active_settings.json')==sha(OUT/'original_settings.json')==r['original_settings_sha256']
for n,h in p['inputs'].items():assert sha(ROOT/n)==h,n
save('audit.json',dict(passed=True,records=len(summary),maximum_cost_error=maximum,maximum_solver_objective_error=objective_error,independent='Scalar fsum cosine, formula from archived base/conf, SciPy dummy.4 assignmentobjective, separate exactclock/GTidentity implementation; no worker/evaluation helper imports.',shared='Frozen model descriptors, raw detection metadata, GT files and clock sources.',all_settings_restored=True,summary=summary))
print('INDEPENDENT_WEIGHT_AUDIT_PASSED',len(summary),maximum,objective_error,flush=True)
