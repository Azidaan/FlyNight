"""Supervised single GPU owner for the frozen A frame/ROI ledger."""
import os,sys,json,hashlib,argparse,time
os.environ['CUDA_VISIBLE_DEVICES']='0';os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent;SHARED=OUT.parent/'shared';sys.path.insert(0,str(ROOT))
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(n,x):
 with (SHARED/n).open('x',encoding='utf8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
def status(stage,**kw):(SHARED/'extraction_status.json').write_text(json.dumps(dict(stage=stage,updated=datetime.now(ZoneInfo('Europe/Moscow')).isoformat(),**kw)),encoding='utf8')
status('starting_guard');sys.path.insert(0,str(ROOT/'runs/flyvis_repairs_20261106/unification_20261107'))
from system_memory_guard import guard_entrypoint
remaining=(datetime.fromisoformat('2026-11-13T07:35:00+03:00')-datetime.now(ZoneInfo('Europe/Moscow'))).total_seconds();assert remaining>0;observation=guard_entrypoint(timeout=remaining)
parser=argparse.ArgumentParser();parser.add_argument('--go-protocol-sha256',required=True);args=parser.parse_args();assert sha(SHARED/'extraction_protocol.json')==args.go_protocol_sha256
def verify():
 p=read(SHARED/'extraction_protocol.json')
 for n,h in p['inputs'].items():assert sha(ROOT/n)==h,n
 return p
p=verify();plan=read(ROOT/p['plan']);runtime=read(ROOT/p['runtime']);old=read(ROOT/p['original_state_source'])
import numpy as np,cv2,torch
from tracking.gpu_lock import GpuLock
from tracking.masa_r50 import build,preprocess
torch.set_num_threads(2);cv2.setNumThreads(2);arrays={};receipts=[];duplicate=False;began=time.perf_counter();status('waiting_for_gpu',frames=p['frames'],rois=p['rois'])
with GpuLock('Local\\DissertationSingleGpu'),torch.inference_mode():
 model=build('cuda');assert not model.training and not torch.is_autocast_enabled() and all(a.device.type=='cuda' for a in model.parameters())
 assert all(hashlib.sha256(a.detach().cpu().numpy().tobytes()).hexdigest()==old['model_state'][n] for n,a in model.state_dict().items())
 for video in sorted({f['video'] for f in plan['frames'].values()}):
  assert video not in runtime['excluded_videos'];wanted={f['frame']:k for k,f in plan['frames'].items() if f['video']==video};entry=plan['frames'][next(iter(wanted.values()))];cap=cv2.VideoCapture(str(ROOT/entry['video_path']));assert cap.isOpened()
  try:
   for fi in range(max(wanted)+1):
    ok,image=cap.read();assert ok,(video,fi)
    if fi not in wanted:continue
    key=wanted[fi];boxes=plan['frames'][key]['boxes'];status('encoding',video=video,frame=fi,completed=len(receipts),total=p['frames']);digest=hashlib.sha256(image.tobytes()).hexdigest()
    x,scale=preprocess(image,'cuda');xc,sc=preprocess(image,'cpu');assert x.device.type=='cuda' and xc.device.type=='cpu' and scale==sc and torch.equal(x.cpu(),xc);feats=model.features(x);pieces=[]
    for start in range(0,len(boxes),p['batch_size']):
     bb=np.asarray(boxes[start:start+p['batch_size']],np.float32);dup=not duplicate
     if dup:bb=np.vstack([bb,bb[:1]])
     bt=torch.from_numpy(bb).to('cuda');z=torch.nn.functional.normalize(model.embed_boxes(feats,bt,scale),dim=1);assert z.device.type=='cuda' and torch.isfinite(z).all()
     if dup:assert torch.equal(z[0],z[-1]);z=z[:-1];duplicate=True
     pieces.append(z.cpu().numpy().copy())
    array=np.concatenate(pieces);assert array.shape==(len(boxes),256) and np.isfinite(array).all();arrays[key]=array
    receipts.append(dict(key=key,pixel_sha256=digest,roi_count=len(boxes),source_boxes=boxes,float32_boxes=np.asarray(boxes,np.float32).tolist(),scale=scale,feature_calls=1));del feats,x,xc,z
  finally:cap.release()
 assert duplicate and len(arrays)==p['frames']
 assert all(hashlib.sha256(a.detach().cpu().numpy().tobytes()).hexdigest()==old['model_state'][n] for n,a in model.state_dict().items())
gpu=dict(name=torch.cuda.get_device_name(),peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved());verify()
with (SHARED/'descriptors_A.npz').open('xb') as f:np.savez_compressed(f,**arrays)
save('extraction_result.json',dict(passed=True,bank_sha256=sha(SHARED/'descriptors_A.npz'),frames=receipts,feature_calls=len(receipts),rois=sum(len(a) for a in arrays.values()),duplicate_first_roi_exact=True,cpu_gpu_preprocess_exact=True,weights_unchanged=True,old_repair_embeddings_reused=0,source_checksums_verified=True,gpu=gpu,seconds=time.perf_counter()-began,supervisor=observation.read(),neural_readouts=0,tracker_calls=0,scope='Descriptorpreparationonly, no newrepairoutcomescomputed.'))
status('complete',completed=len(receipts),total=p['frames']);print('A_BANK_COMPLETE',len(receipts),sum(len(a) for a in arrays.values()),flush=True)
