"""Deployment preflight on ready matrices; no new trajectories or signal fits."""
import os,sys,json,hashlib,gzip,subprocess
from pathlib import Path
from datetime import datetime,timezone,timedelta
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
TRAIN=ROOT/'research_domain_matched_reader_20261109';OLD=ROOT/'runs/full_template_native_20261109'
MSK=timezone(timedelta(hours=3));STOP=datetime(2026,11,9,23,54,tzinfo=MSK)
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'runs/flyvis_repairs_20261106/unification_20261107'))
from system_memory_guard import guard_entrypoint
guard_entrypoint(timeout=min(600,(STOP-datetime.now(MSK)).total_seconds()))
import psutil
psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,obj):
    with (OUT/name).open('x',encoding='utf8') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
mode=sys.argv[1]
if mode=='register':
    assert read(TRAIN/'summary.json')['gate'] and read(TRAIN/'audit.json')['passed'] and read(TRAIN/'final_checks.json')['passed']
    files=[Path(__file__),TRAIN/'deployment.json',TRAIN/'summary.json',TRAIN/'audit.json',TRAIN/'final_checks.json',
        ROOT/'tracking/channel_identity_head.py',OLD/'native_runtime.py',OLD/'boundary_audit.json',
        ROOT/'.venv/Lib/site-packages/lap/__init__.py',ROOT/'.venv/Lib/site-packages/lap/_lapjv_wp.py']
    files+=sorted((OLD/'rollouts/C34').glob('*.events.json.gz'))
    save('delivery_protocol.json',dict(registered=datetime.now(MSK).isoformat(),head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip(),sources={p.relative_to(ROOT).as_posix():sha(p) for p in files},
        question='Does the newly trained deployed C34 head admit original-cost1edges that float32delivery still rejects? Old negative boundary finding applies only to the OLD head.',
        action='Before a native pilot, compare exactly the new frozen decisions under unchanged float32 subtraction versus float64 subtraction with the SAME alpha0.2 and threshold0.8. Same closed oldC34histories and full saved global matrices; no labels influence decisions.',
        no_change='Not a threshold search, not a new pose/readout, not a full trajectory result. If deliveryprecision matters, any actual fix must have identical precision for ordinary-input and geometry controls and a full empty-vote zero-control. Preserve a float32newC34 arm to isolate training-domain benefit from numerical delivery.',
        inputs='Reuse all213readyoldC34geometry/channel vectors. No images,maps,network,optimizer or video. Other proposals outside the domain unchanged.',deadline=STOP.isoformat()))
    print('DELIVERY_REGISTERED',flush=True);raise SystemExit(0)
assert mode=='run';p=read(OUT/'delivery_protocol.json')
for name,digest in p['sources'].items():assert sha(ROOT/name)==digest,name
import numpy as np
from ultralytics.trackers.utils import matching
from tracking.channel_identity_head import decision_function
model=read(TRAIN/'deployment.json');stats=model['geometry_normalization'];records=[];frames=[];scored=0;empty_equal=0
def assignment(cost):return np.asarray(matching.linear_assignment(cost,thresh=.8)[0]).reshape(-1,2).tolist()
for path in sorted((OLD/'rollouts/C34').glob('*.events.json.gz')):
    video=path.name.split('.events')[0]
    with gzip.open(path,'rt',encoding='utf8') as f:events=json.load(f)['events']
    for event in events:
        rows=[r for r in event['proposals'] if r['reason'] in ['head_accept','head_refuse']]
        if not rows:continue
        c=np.array(event['C'],np.float32);a=c.copy();b=c.astype(float)
        assert assignment(c)==assignment(c.astype(float))==event['native_first'];empty_equal+=1
        accepted=[]
        for row in rows:
            g=(np.array(row['geometry'])-stats['mean'])/stats['scale']
            if row.get('evidence_valid',False):z=float(decision_function(model['models']['C34'],np.r_[g,row['channels']][None])[0])
            else:z=float(decision_function(model['models']['G7'],g[None])[0])
            i,j=row['row'],row['column'];scored+=1
            if z>0:a[i,j]-=.2;b[i,j]-=.2;accepted.append([i,j])
            records.append(dict(video=video,frame=event['frame'],old_id=row['old_id'],raw_id=row['raw_id'],new_logit=z,accepted=z>0,
                original_cost=float(c[i,j]),float32_corrected=float(a[i,j]),float64_corrected=float(b[i,j]),
                accepted_but_above_float32_limit=bool(z>0 and float(a[i,j])>.8),accepted_within_float64_limit=bool(z>0 and float(b[i,j])<=.8)))
        aa=assignment(a);bb=assignment(b)
        frames.append(dict(video=video,frame=event['frame'],votes=accepted,float32_matches=aa,float64_matches=bb,changed=aa!=bb))
assert scored==213
for name,digest in p['sources'].items():assert sha(ROOT/name)==digest,name
summary=dict(scored=scored,accepted=sum(r['accepted'] for r in records),
    accepted_above_float32_limit=sum(r['accepted_but_above_float32_limit'] for r in records),
    accepted_cost_exactly_one=sum(r['accepted'] and r['original_cost']==1 for r in records),
    full_global_frames=len(frames),empty_cast_frames_exact=empty_equal,
    global_assignments_changed=sum(f['changed'] for f in frames))
save('delivery_report.json',dict(passed=True,summary=summary,records=records,frames=frames,
    not_actual_new_IDs=True,interpretation='Counterfactual delivery on closed old histories only; no new trajectory or benefit claim. Fix precision only with matched native controls and explicit sensitivity.'))
print('DELIVERY',json.dumps(summary),flush=True)
