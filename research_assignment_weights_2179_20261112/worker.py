from common import *
os.environ['CUDA_VISIBLE_DEVICES']=''
import math,numpy as np,lap
from evaluate import evaluation_tables,classify
p=read(OUT/'protocol.json');f=read(OUT/'feature_result.json');assert f['passed'] and sha(OUT/'descriptors.npz')==f['bank_sha256']
for n,h in p['inputs'].items():assert sha(ROOT/n)==h,n
runtime=read(OUT/'runtime.json');bank=np.load(OUT/'descriptors.npz',allow_pickle=False);tables=evaluation_tables();original_settings=(OUT/'original_settings.json').read_bytes();records=[]
def solve(a):
    _,x,_=lap.lapjv(a,extend_cost=True,cost_limit=.8);return [(i,int(j)) for i,j in enumerate(x) if j>=0]
def vector(slot):return bank[slot[0]][slot[1]].astype(float)
def predicted(r):
    m=r['mean'];return [m[0]-m[2]/2,m[1]-m[3]/2,m[0]+m[2]/2,m[1]+m[3]/2]
def typed_iou(a,b):
    a=np.asarray(a,np.float32);b=np.asarray(b,np.float32);v=np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2]));x=np.float32(v[0]*v[1]);u=np.float32(np.float32((a[2]-a[0])*(a[3]-a[1]))+np.float32((b[2]-b[0])*(b[3]-b[1]))-x);return np.float32(x/np.float32(u+np.float32(1e-7)))
try:
    for e,table in zip(runtime['events'],tables):
        base=np.asarray(e['base'],np.float32);saved=np.asarray(e['corrected'],np.float32);frozen_base=base.copy();frozen_saved=saved.copy();ids=[r['tid'] for r in e['rows']];rawids=[c['raw_id'] for c in e['candidates']];confidence=np.asarray([c['confidence'] for c in e['candidates']],np.float32)
        complete=all(r['reference'] is not None for r in e['rows']);assert complete,'Do not drop incomplete competitor rows'
        a=np.stack([vector(r['reference']['slot']) for r in e['rows']]);b=np.stack([vector(c['slot']) for c in e['candidates']]);a/=np.linalg.norm(a,axis=1)[:,None];b/=np.linalg.norm(b,axis=1)[:,None];appearance=(1-np.clip(a@b.T,-1,1))/2
        iou=np.array([[typed_iou(predicted(r),c['box']) for c in e['candidates']] for r in e['rows']],np.float32);native=np.float32(1-np.float32(np.float32(1-np.float32(1-iou))*confidence));assert np.array_equal(native,base),(e['frame'],abs(native-base).max())
        before=solve(saved);assert sorted(before)==sorted(map(tuple,e['actual_first']));baseline_pairs=[(ids[i],rawids[j]) for i,j in before];baseline_eval=classify(table,baseline_pairs);good={r['tid'] for r in baseline_eval if r['outcome']=='correct'};wrong={r['tid'] for r in baseline_eval if r['outcome']=='wrong'}
        for v in p['variants']:
            settings=dict(variant=v['name'],appearance_weight=v['appearance_weight'],confidence_power=v['confidence_power']);(OUT/'active_settings.json').write_bytes((json.dumps(settings,sort_keys=True)+'\n').encode())
            if v['kind']=='identity':cost=saved.copy()
            elif v['kind']=='confidence':
                sim=np.float32(1-np.float32(1-iou));newbase=np.float32(1-np.float32(sim*np.power(confidence,np.float32(v['confidence_power']))));cost=np.float32(saved+np.float32(newbase-base))
            else:
                ap=appearance
                if v['kind']=='permuted':
                    perm=np.roll(np.arange(len(b)),1);ap=(1-np.clip(a@b[perm].T,-1,1))/2
                cost=np.asarray(saved+v['appearance_weight']*(ap-base),np.float32)
            assert cost.shape==saved.shape and cost.dtype==np.float32 and np.isfinite(cost).all()
            if v['name'] in ['baseline','identity']:assert np.array_equal(cost,saved)
            pairs=[(ids[i],rawids[j]) for i,j in solve(cost)];ev=classify(table,pairs);oldset=set(baseline_pairs);added=set(pairs)-oldset;removed=oldset-set(pairs);newgood={r['tid'] for r in ev if r['outcome']=='correct'};newwrong={r['tid'] for r in ev if r['outcome']=='wrong'};unknown_added=[x for x in ev if (x['tid'],x['raw_id']) in added and x['outcome']=='unknown'];worse=bool(good-newgood or newwrong-wrong)
            records.append(dict(frame=e['frame'],variant=v['name'],settings=settings,cost=cost.tolist(),pairs=[list(x) for x in pairs],added=[list(x) for x in sorted(added)],removed=[list(x) for x in sorted(removed)],evaluation=ev,baseline_correct_tids=sorted(good),correct_tids=sorted(newgood),new_wrong_tids=sorted(newwrong-wrong),lost_correct_tids=sorted(good-newgood),unknown_added=unknown_added,confirmed_worse=worse,target2179=[x for x in ev if x['tid']==2179],raw109699_owner=[t for t,r in pairs if r==109699],appearance_2179_to_all=appearance[ids.index(2179)].tolist() if 2179 in ids else None))
            (OUT/'active_settings.json').write_bytes(original_settings);assert (OUT/'active_settings.json').read_bytes()==original_settings
        assert np.array_equal(base,frozen_base) and np.array_equal(saved,frozen_saved)
finally:
    (OUT/'active_settings.json').write_bytes(original_settings)
for n,h in p['inputs'].items():assert sha(ROOT/n)==h,n
save('result.json',dict(passed=True,records=records,evaluation_tables=tables,bank_sha256=f['bank_sha256'],original_settings_sha256=sha(OUT/'original_settings.json'),restored_settings_sha256=sha(OUT/'active_settings.json'),all_variants_rolled_back=True,production_files_changed=False,training_calls=0,core_calls=0,tracker_calls=0,scope='Within-case fixed-snapshot first-assignment intervention. Archived core deltas fixed; no state/lifecycle rerun. ExpectedpersistentIDs are not labels of unannotateddescriptorpixels.'))
print('WEIGHT_PROBE_COMPLETE',len(records),flush=True)
