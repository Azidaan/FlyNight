from common import *
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np, lap, importlib.util
p=read(OUT/'protocol.json')
for n,h in p['inputs'].items(): assert sha(ROOT/n)==h,n
assert read(OUT/'feature_result.json')['bank_sha256']==sha(OUT/'descriptors.npz')
runtime=read(PREV/'runtime.json'); plan=read(OUT/'descriptor_plan.json'); bank=np.load(OUT/'descriptors.npz',allow_pickle=False); oldbank=np.load(PREV/'descriptors.npz',allow_pickle=False)
# Evaluation-only helper is isolated from feature/reference selection.
spec=importlib.util.spec_from_file_location('old_evaluation',PREV/'evaluate.py'); ev=importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)
ev.OUT=PREV; tables=ev.evaluation_tables()
def norm(x): return x.astype(float)/np.linalg.norm(x.astype(float))
def oldvec(slot): return norm(oldbank[slot[0]][slot[1]])
def newvec(slot): return norm(bank[slot[0]][slot[1]])
def select(e,row,policy):
    if row['tid'] not in [2179,2445] or policy in ['latest','native']: return [oldvec(row['reference']['slot'])], [dict(frame=row['reference']['frame'],kind='filtered')]
    obs=[o for o in plan['observations'] if o['tid']==row['tid'] and any(o['frame']==h['frame'] and o['raw_id']==h['raw_id'] for h in row['history'])]
    assert obs and all(o['frame']<e['frame'] for o in obs)
    obs.sort(key=lambda o:o['frame'])
    kind='raw'
    if policy=='latest_raw': chosen=obs[-1:]
    elif policy=='earliest_filtered': chosen=obs[:1]; kind='filtered'
    elif policy=='earliest_raw': chosen=obs[:1]
    elif policy=='gallery_raw': chosen=obs
    elif policy.startswith('freeze2445_'):
        if row['tid']!=2445: return [oldvec(row['reference']['slot'])],[dict(frame=row['reference']['frame'],kind='filtered')]
        frame=min(int(policy.rsplit('_',1)[1]),e['frame']-1)
        o=next(o for o in plan['observations'] if o['tid']==2445 and o['frame']==frame)
        chosen=[o];kind='filtered'
    elif policy=='swap_latest_negative':
        other=next(r for r in e['rows'] if r['tid']==(2445 if row['tid']==2179 else 2179))
        return [oldvec(other['reference']['slot'])],[dict(frame=other['reference']['frame'],kind='swapped_filtered')]
    else: raise ValueError(policy)
    return [newvec(o[kind+'_slot']) for o in chosen],[dict(frame=o['frame'],raw_id=o['raw_id'],kind=kind) for o in chosen]
records=[]
for e,table in zip(runtime['events'],tables):
    ids=[r['tid'] for r in e['rows']]; rawids=[c['raw_id'] for c in e['candidates']]; base=np.asarray(e['base'],np.float32); saved=np.asarray(e['corrected'],np.float32); candidates=np.stack([oldvec(c['slot']) for c in e['candidates']])
    for alpha in p['alphas']:
        for policy in p['policies']:
            ap=[]; refs={}
            for row in e['rows']:
                vectors,witness=select(e,row,policy); refs[str(row['tid'])]=witness
                ap.append(np.min((1-np.clip(np.stack(vectors)@candidates.T,-1,1))/2,axis=0))
            appearance=np.stack(ap); cost=saved.copy() if policy=='native' else np.asarray(saved+alpha*(appearance-base),np.float32)
            _,x,_=lap.lapjv(cost,extend_cost=True,cost_limit=.8); pairs=[(ids[i],rawids[j]) for i,j in enumerate(x) if j>=0]
            evaluation=ev.classify(table,pairs)
            if policy=='native': assert sorted((ids[i],rawids[j]) for i,j in e['actual_first'])==sorted(pairs)
            records.append(dict(frame=e['frame'],alpha=alpha,policy=policy,cost=cost.tolist(),pairs=pairs,evaluation=evaluation,references=refs,targets={str(t):[(a,b) for a,b in pairs if a==t] for t in [2179,2445]},probe_costs={str(raw):{str(t):dict(cost=float(cost[ids.index(t),rawids.index(raw)]),appearance=float(appearance[ids.index(t),rawids.index(raw)])) for t in [2179,2445]} for raw in [109699,110260] if raw in rawids}))
for n,h in p['inputs'].items(): assert sha(ROOT/n)==h,n
save('result.json',dict(passed=True,records=records,evaluation_tables=tables,core_calls=0,tracker_calls=0,production_changes=0,scope=p['scope']))
print('HISTORY_ASSIGNMENT_COMPLETE',len(records),flush=True)
