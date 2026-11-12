"""Execute only after parent supplies exact registered protocol hash."""
import os,sys
os.environ['CUDA_VISIBLE_DEVICES']='0'
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
from common import *
import argparse,time
parser=argparse.ArgumentParser();parser.add_argument('--go-protocol-sha256',required=True)
# guard_entrypoint consumes its private marker before this parser runs.
sys.path.insert(0,str(ROOT/'runs/flyvis_repairs_20261106/unification_20261107'))
from system_memory_guard import guard_entrypoint
def status(stage,**extra):(OUT/'status.json').write_text(json.dumps(dict(stage=stage,updated=now(),**extra)),encoding='utf-8')
status('starting_guard')
remaining=(datetime.fromisoformat('2026-11-13T07:35:00+03:00')-datetime.now(ZoneInfo('Europe/Moscow'))).total_seconds();assert remaining>0
observation=guard_entrypoint(timeout=remaining)
args=parser.parse_args();assert sha(OUT/'protocol.json')==args.go_protocol_sha256
p=verify()
import numpy as np,cv2,torch
from tracking.gpu_lock import GpuLock
from tracking.masa_r50 import build,preprocess
cv2.setNumThreads(2);torch.set_num_threads(2)
jobs={};case_slots={}
def add(v,frame,box):
    key=v+'|'+str(frame);tasks=jobs.setdefault(key,dict(video=v,frame=frame,boxes=[]))
    # Equality in source coordinates; float32conversion recorded separately.
    if box not in tasks['boxes']:tasks['boxes'].append(box)
    return key,tasks['boxes'].index(box)
for c in p['cases']:
    cid=str(c['old_id']);v=c['video'];slots=dict(reference=add(v,c['reference_frame'],c['reference_box']),candidates=[])
    for r in c['candidates']:slots['candidates'].append(dict(raw_id=r['raw_id'],slot=add(v,c['current_frame'],r['box'])))
    slots['desired']=add(v,c['current_frame'],c['desired']['box']);slots['oracle']=[dict(identity=o['identity'],slot=add(v,c['current_frame'],o['box'])) for o in c['oracle_current_same_identity']]
    case_slots[cid]=slots
arrays={};frame_receipts=[];began=time.perf_counter();guard_checked=False
status('waiting_for_gpu',unique_frames=len(jobs))
with GpuLock('Local\\DissertationSingleGpu'),torch.inference_mode():
    model=build(device='cuda');assert not model.training and all(x.device.type=='cuda' for x in model.parameters())
    assert not torch.is_autocast_enabled();model_state={n:hashlib.sha256(a.detach().cpu().numpy().tobytes()).hexdigest() for n,a in model.state_dict().items()}
    for v in sorted(p['videos']):
        wanted={j['frame']:k for k,j in jobs.items() if j['video']==v};cap=cv2.VideoCapture(str(ROOT/p['videos'][v]['video_path']));assert cap.isOpened()
        try:
            for frame in range(max(wanted)+1):
                ok,image=cap.read();assert ok,(v,frame)
                if frame not in wanted:continue
                key=wanted[frame];job=jobs[key];digest=hashlib.sha256(image.tobytes()).hexdigest();assert digest==p['expected_raw_frame_hashes'][key]
                status('encoding',video=v,frame=frame,completed=len(frame_receipts),total=len(jobs))
                x,scale=preprocess(image,device='cuda');assert x.device.type=='cuda'
                # CPU preprocessing path checked without a second model instance.
                cpu_x,cpu_scale=preprocess(image,device='cpu');assert cpu_x.device.type=='cpu' and scale==cpu_scale and torch.equal(x.cpu(),cpu_x)
                feats=model.features(x);values=[]
                for start in range(0,len(job['boxes']),p['batch_size']):
                    bb=np.asarray(job['boxes'][start:start+p['batch_size']],np.float32)
                    duplicate=not guard_checked
                    if duplicate:bb=np.vstack([bb,bb[:1]])
                    bt=torch.from_numpy(bb).to('cuda');assert bt.device.type=='cuda'
                    e=torch.nn.functional.normalize(model.embed_boxes(feats,bt,scale),dim=1)
                    assert e.device.type=='cuda' and torch.isfinite(e).all()
                    if duplicate:assert torch.equal(e[0],e[-1]);e=e[:-1];guard_checked=True
                    values.append(e.cpu().numpy().copy())
                z=np.concatenate(values);assert z.shape==(len(job['boxes']),256) and np.isfinite(z).all();arrays[key]=z
                frame_receipts.append(dict(key=key,pixel_sha256=digest,source_boxes=job['boxes'],float32_boxes=np.asarray(job['boxes'],np.float32).tolist(),count=len(z),scale=scale,feature_calls=1))
                del feats,x,cpu_x,e,z
        finally:cap.release()
    assert guard_checked
    assert all(hashlib.sha256(a.detach().cpu().numpy().tobytes()).hexdigest()==model_state[n] for n,a in model.state_dict().items())
    gpu=dict(name=torch.cuda.get_device_name(),peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved())
del model
def vector(slot):return arrays[slot[0]][slot[1]].astype(float)
records=[]
for c in p['cases']:
    s=case_slots[str(c['old_id'])];ref=vector(s['reference']);ref/=np.linalg.norm(ref);rows=[]
    for raw in s['candidates']:
        r=next(r for r in c['candidates'] if r['raw_id']==raw['raw_id']);z=vector(raw['slot']);score=float(ref@(z/np.linalg.norm(z)));ids=[h['identity'] for h in r['hits']]
        rows.append(dict(**r,cosine=score,own=ids==[c['identity']],ambiguous=len(ids)>1,annotation_unmatched=len(ids)==0))
    for r in rows:r['rank']=1+sum(t['cosine']>r['cosine'] for t in rows)
    own=[r['cosine'] for r in rows if r['own']];foreign=[r['cosine'] for r in rows if len(r['hits'])==1 and not r['own']]
    z=vector(s['desired']);desired_score=float(ref@(z/np.linalg.norm(z)))
    oracles=[]
    for oracle in s['oracle']:
        z=vector(oracle['slot']);oracles.append(dict(identity=oracle['identity'],cosine=float(ref@(z/np.linalg.norm(z))),oracle=True))
    records.append(dict(old_id=c['old_id'],label=c['label'],identity=c['identity'],operational_candidates=rows,own_available=bool(own),best_own_cosine=max(own) if own else None,best_foreign_cosine=max(foreign) if foreign else None,own_minus_foreign_margin=max(own)-max(foreign) if own and foreign else None,desired=dict(**c['desired'],cosine=desired_score,rank_in_operational=1+sum(r['cosine']>desired_score for r in rows)),oracle=oracles))
verify()
with (OUT/'descriptors.npz').open('xb') as f:np.savez_compressed(f,**arrays)
save('result.json',dict(passed=True,records=records,slots=case_slots,frames=frame_receipts,descriptors_sha256=sha(OUT/'descriptors.npz'),model_state=model_state,gpu=gpu,duplicate_roi_exact=True,cpu_gpu_preprocess_exact=True,supervisor=observation.read(),seconds=time.perf_counter()-began,new_fits=0,tracker_calls=0,feature_calls=len(frame_receipts)))
status('complete',completed=len(frame_receipts),total=len(jobs));print('COMPLETE',len(records),len(jobs),flush=True)
