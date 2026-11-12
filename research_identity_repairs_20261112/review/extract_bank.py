"""Single supervised GPU owner; exact frozen targets are copied, never encoded."""
import os,sys,json,hashlib,argparse,time
os.environ['CUDA_VISIBLE_DEVICES']='0';os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(n,d):
 with (OUT/n).open('x',encoding='utf8') as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False)
def status(stage,**kw):(OUT/'bank_status.json').write_text(json.dumps(dict(stage=stage,updated=datetime.now(ZoneInfo('Europe/Moscow')).isoformat(),**kw)),encoding='utf8')
status('starting_guard');sys.path.insert(0,str(ROOT/'runs/flyvis_repairs_20261106/unification_20261107'))
from system_memory_guard import guard_entrypoint
remaining=(datetime.fromisoformat('2026-11-13T07:35:00+03:00')-datetime.now(ZoneInfo('Europe/Moscow'))).total_seconds();assert remaining>0
observation=guard_entrypoint(timeout=remaining)
parser=argparse.ArgumentParser();parser.add_argument('--go-protocol-sha256',required=True);args=parser.parse_args();assert sha(OUT/'bank_protocol.json')==args.go_protocol_sha256
def verify():
 p=read(OUT/'bank_protocol.json')
 for n,h in p['inputs'].items():assert sha(ROOT/n)==h,n
 return p
p=verify()
import numpy as np,cv2,torch
from tracking.gpu_lock import GpuLock
from tracking.masa_r50 import build,preprocess
torch.set_num_threads(2);cv2.setNumThreads(2);old=read(ROOT/p['original_result_path']);arrays={k:np.full((len(f['boxes']),256),np.nan,np.float32) for k,f in p['frames'].items()};reused={}
with np.load(ROOT/p['original_descriptor_path']) as z:
 for item in p['target_reuse']:
  dest,source=item['target_slot'],item['source_slot'];arrays[dest[0]][dest[1]]=z[source[0]][source[1]];reused[(dest[0],dest[1])]=source
began=time.perf_counter();receipts=[];guard_checked=False;features_called=0
status('waiting_for_gpu',frames=len(arrays))
with GpuLock('Local\\DissertationSingleGpu'),torch.inference_mode():
 model=build('cuda');assert not model.training and not torch.is_autocast_enabled() and all(a.device.type=='cuda' for a in model.parameters())
 assert all(hashlib.sha256(a.detach().cpu().numpy().tobytes()).hexdigest()==old['model_state'][n] for n,a in model.state_dict().items())
 for video in sorted({f['video'] for f in p['frames'].values()}):
  wanted={f['frame']:k for k,f in p['frames'].items() if f['video']==video};entry=p['frames'][next(iter(wanted.values()))];cap=cv2.VideoCapture(str(ROOT/entry['video_path']));assert cap.isOpened()
  try:
   for fi in range(max(wanted)+1):
    ok,image=cap.read();assert ok,(video,fi)
    if fi not in wanted:continue
    key=wanted[fi];frame=p['frames'][key];digest=hashlib.sha256(image.tobytes()).hexdigest();previous=next((r['pixel_sha256'] for r in old['frames'] if r['key']==key),None)
    if previous:assert digest==previous
    missing=[i for i,row in enumerate(arrays[key]) if not np.isfinite(row).all()];status('encoding',video=video,frame=fi,completed=len(receipts),total=len(arrays),new_roi=len(missing))
    if missing:
     x,scale=preprocess(image,'cuda');xc,sc=preprocess(image,'cpu');assert x.device.type=='cuda' and xc.device.type=='cpu' and scale==sc and torch.equal(x.cpu(),xc);feats=model.features(x);features_called+=1
     for start in range(0,len(missing),p['batch_size']):
      ids=missing[start:start+p['batch_size']];boxes=np.asarray([frame['boxes'][i] for i in ids],np.float32);dup=not guard_checked
      if dup:boxes=np.vstack([boxes,boxes[:1]])
      bt=torch.from_numpy(boxes).to('cuda');e=torch.nn.functional.normalize(model.embed_boxes(feats,bt,scale),dim=1);assert e.device.type=='cuda' and torch.isfinite(e).all()
      if dup:assert torch.equal(e[0],e[-1]);e=e[:-1];guard_checked=True
      arrays[key][ids]=e.cpu().numpy()
     del feats,x,xc,e
    assert np.isfinite(arrays[key]).all();receipts.append(dict(key=key,pixel_sha256=digest,roi_count=len(arrays[key]),encoded_indices=missing,reused_indices=[i for i in range(len(arrays[key])) if (key,i) in reused],feature_calls=int(bool(missing))))
  finally:cap.release()
 assert guard_checked
 assert all(hashlib.sha256(a.detach().cpu().numpy().tobytes()).hexdigest()==old['model_state'][n] for n,a in model.state_dict().items())
gpu=dict(name=torch.cuda.get_device_name(),peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved());verify()
with (OUT/'reference_bank.npz').open('xb') as f:np.savez_compressed(f,**arrays)
save('bank_result.json',dict(passed=True,bank_sha256=sha(OUT/'reference_bank.npz'),frames=receipts,target_reuse=p['target_reuse'],all_reference_rows=sum(len(a) for a in arrays.values()),feature_calls=features_called,duplicate_new_roi_exact=guard_checked,target_vectors_reencoded=0,cpu_gpu_preprocess_exact=True,source_checksums_verified=True,gpu=gpu,seconds=time.perf_counter()-began,supervisor=observation.read(),current_descriptor_path=p['current_descriptor_path'],current_ledger=p['current_ledger']))
status('complete',completed=len(receipts),total=len(arrays));print('REFERENCE_BANK_COMPLETE',len(receipts),sum(len(a) for a in arrays.values()),flush=True)
