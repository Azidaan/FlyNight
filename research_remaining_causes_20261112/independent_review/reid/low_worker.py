"""Single GPU owner: five current frames, all saved raw-car boxes."""
import os,sys
os.environ['CUDA_VISIBLE_DEVICES']='0';os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
from common import *
import argparse,time
sys.path.insert(0,str(ROOT/'runs/flyvis_repairs_20261106/unification_20261107'))
from system_memory_guard import guard_entrypoint
def status(stage,**kw):(OUT/'low_status.json').write_text(json.dumps(dict(stage=stage,updated=now(),**kw)),encoding='utf8')
status('starting_guard');remaining=(datetime.fromisoformat('2026-11-13T07:35:00+03:00')-datetime.now(ZoneInfo('Europe/Moscow'))).total_seconds();assert remaining>0
observation=guard_entrypoint(timeout=remaining)
parser=argparse.ArgumentParser();parser.add_argument('--go-protocol-sha256',required=True);args=parser.parse_args();assert sha(OUT/'low_protocol.json')==args.go_protocol_sha256
def verifylow():
    p=read(OUT/'low_protocol.json')
    for n,h in p['inputs'].items():assert sha(ROOT/n)==h,n
    verify();return p
p=verifylow();oldr=read(OUT/'result.json')
import numpy as np,cv2,torch
from tracking.gpu_lock import GpuLock
from tracking.masa_r50 import build,preprocess
torch.set_num_threads(2);cv2.setNumThreads(2);refs={};previous={}
with np.load(OUT/'descriptors.npz') as old:
    for c in p['cases']:
        s=oldr['slots'][str(c['old_id'])];refs[str(c['old_id'])]=old[s['reference'][0]][s['reference'][1]].copy()
        for slot in s['candidates']:previous[(c['video'],c['current_frame'],slot['raw_id'])]=old[slot['slot'][0]][slot['slot'][1]].copy()
arrays={};receipts=[];records=[];max_descriptor_error=0.;max_score_error=0.;duplicate_checked=False;began=time.perf_counter()
status('waiting_for_gpu')
with GpuLock('Local\\DissertationSingleGpu'),torch.inference_mode():
    model=build('cuda');assert not model.training and all(x.device.type=='cuda' for x in model.parameters()) and not torch.is_autocast_enabled()
    assert all(hashlib.sha256(a.detach().cpu().numpy().tobytes()).hexdigest()==oldr['model_state'][n] for n,a in model.state_dict().items())
    for c in p['cases']:
        cid=str(c['old_id']);v=c['video'];f=c['current_frame'];cap=cv2.VideoCapture(str(ROOT/p['videos'][v]['video_path']));assert cap.isOpened()
        try:
            for fi in range(f+1):ok,image=cap.read();assert ok,(v,fi)
        finally:cap.release()
        digest=hashlib.sha256(image.tobytes()).hexdigest();assert digest==p['expected_raw_frame_hashes'][v+'|'+str(f)];status('encoding',old_id=c['old_id'],completed=len(receipts),total=5)
        x,scale=preprocess(image,'cuda');xc,sc=preprocess(image,'cpu');assert x.device.type=='cuda' and xc.device.type=='cpu' and scale==sc and torch.equal(x.cpu(),xc)
        features=model.features(x);values=[]
        for start in range(0,len(c['candidates']),p['batch_size']):
            boxes=np.asarray([r['box'] for r in c['candidates'][start:start+p['batch_size']]],np.float32);dup=not duplicate_checked
            if dup:boxes=np.vstack([boxes,boxes[:1]])
            bt=torch.from_numpy(boxes).to('cuda');e=torch.nn.functional.normalize(model.embed_boxes(features,bt,scale),dim=1);assert e.device.type=='cuda' and torch.isfinite(e).all()
            if dup:assert torch.equal(e[0],e[-1]);e=e[:-1];duplicate_checked=True
            values.append(e.cpu().numpy().copy())
        current=np.concatenate(values);arrays[cid]=current;ref=refs[cid].astype(float);ref/=np.linalg.norm(ref);rows=[]
        oldcase=next(r for r in oldr['records'] if r['old_id']==c['old_id'])
        for i,raw in enumerate(c['candidates']):
            z=current[i].astype(float);value=float(ref@(z/np.linalg.norm(z)));identities=[h['identity'] for h in raw['hits']];row=dict(**raw,cosine=value,own=identities==[c['identity']],annotation_unmatched=not identities,ambiguous=len(identities)>1);rows.append(row)
            key=(v,f,raw['raw_id'])
            if key in previous:
                error=float(abs(current[i]-previous[key]).max());max_descriptor_error=max(max_descriptor_error,error);assert error<1e-6,(key,error)
                ov=next(r['cosine'] for r in oldcase['operational_candidates'] if r['raw_id']==raw['raw_id']);err=abs(value-ov);max_score_error=max(max_score_error,err);assert err<1e-6,(key,err)
        for row in rows:
            row['rank_all_raw']=1+sum(z['cosine']>row['cosine'] for z in rows)
            row['rank_previous_diagnostic']=1+sum(z['cosine']>row['cosine'] for z in rows if z['previous_diagnostic_ge_point1'])
        own=[r for r in rows if r['own']];foreign=[r for r in rows if len(r['hits'])==1 and not r['own']];other=[r for r in rows if not r['own']]
        records.append(dict(old_id=c['old_id'],identity=c['identity'],candidates=rows,own_available_all_raw=bool(own),best_own_raw_id=max(own,key=lambda r:r['cosine'])['raw_id'] if own else None,best_own_cosine=max(r['cosine'] for r in own) if own else None,best_labeled_foreign_cosine=max(r['cosine'] for r in foreign) if foreign else None,own_minus_all_other=max(r['cosine'] for r in own)-max(r['cosine'] for r in other) if own and other else None))
        receipts.append(dict(video=v,frame=f,pixel_sha256=digest,roi_count=len(rows),feature_calls=1));del features,x,xc,e
    assert all(hashlib.sha256(a.detach().cpu().numpy().tobytes()).hexdigest()==oldr['model_state'][n] for n,a in model.state_dict().items())
verifylow()
with (OUT/'low_descriptors.npz').open('xb') as f:np.savez_compressed(f,**arrays)
save('low_result.json',dict(passed=True,records=records,frames=receipts,descriptors_sha256=sha(OUT/'low_descriptors.npz'),max_previous_descriptor_error=max_descriptor_error,max_previous_score_error=max_score_error,duplicate_roi_exact=duplicate_checked,seconds=time.perf_counter()-began,new_fits=0,new_tracker_calls=0,new_confidence_gate=False))
status('complete',completed=5,total=5);print('LOW_COMPLETE',sum(len(r['candidates']) for r in records),flush=True)
