"""Новые изображения и две фиксированные версии FlyVis; без изменения весов на диске."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['Y1_EXECUTION_DEVICE']='cpu'
from pathlib import Path
import argparse,hashlib,json,os,shutil,sys,time
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import flyvis_y1_latency_fp64_20261104 as W
OUT=ROOT/'runs/flyvis_offset_20261104'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,obj):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.next');t.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');t.replace(p)
def positions(case):
 v=np.zeros((79,2),np.int32);v[(31 if case['start']=='still' else 1):55,case['axis']]=case['speed'];v[55:,case['axis']]=-case['speed'];return np.cumsum(v,axis=0),v
def prepare():
 p=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'));W.positions=positions;cv2.setNumThreads(2)
 centers=np.array([(int(13*(u+v/2)),13*v) for u in range(-15,16) for v in range(max(-15,-15-u),min(15,15-u)+1)],np.int64)
 mask=(centers.astype(float)**2).sum(1)<=(195*.5)**2;np.savez_compressed(OUT/'geometry.npz',centers=centers,mask=mask)
 for arm in ('stock','fast'):
  d=OUT/arm;d.mkdir(exist_ok=True);(d/'inputs').mkdir(exist_ok=True)
  if not (d/'geometry.npz').exists():os.link(OUT/'geometry.npz',d/'geometry.npz')
  if not (d/'protocol.json').exists():shutil.copy2(OUT/'protocol.json',d/'protocol.json')
 paths={};sources=[]
 for i,video in enumerate(p['videos']):
  source=ROOT/'data/bdd100k/videos/train'/(video+'.mov');cap=cv2.VideoCapture(str(source));ok,frame=cap.read();fps=cap.get(cv2.CAP_PROP_FPS);cap.release();assert ok
  paths[i]=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY);sources.append(dict(video=video,source_sha256=sha(source),frame_sha256=hashlib.sha256(paths[i].tobytes()).hexdigest(),fps=fps))
 write(OUT/'source_images.json',dict(sources=sources))
 started=time.monotonic()
 for index,case in enumerate(p['cases']):
  dest=OUT/'inputs'/(case['id']+'.npz');marker=dest.with_suffix('.json');dest.parent.mkdir(exist_ok=True)
  if marker.exists():assert sha(dest)==json.loads(marker.read_text())['sha256']
  else:
   frames=W.frames_for(paths[case['scene']],case);eyes=W.cpu_eye(frames,centers)
   np.savez_compressed(dest,frames=frames,eye_reference=eyes)
   write(marker,dict(sha256=sha(dest),frame_sha256=[hashlib.sha256(g.tobytes()).hexdigest() for g in frames]))
  for arm in ('stock','fast'):
   for source in (dest,marker):
    target=OUT/arm/'inputs'/source.name
    if not target.exists():os.link(source,target)
    assert sha(target)==sha(source)
  free=shutil.disk_usage(OUT).free;assert free>=5_000_000_000
  if index==0:
   required=2*dest.stat().st_size*(len(p['cases'])-1)+5_000_000_000
   assert free>required;write(OUT/'disk_gate.json',dict(required=required,free=free,passed=True))
  write(OUT/'status.json',dict(stage='prepare',done=index+1,total=len(p['cases']),elapsed_seconds=time.monotonic()-started,free=free))
 for arm in ('stock','fast'):write(OUT/arm/'inputs_completed.json',dict(complete=True,cases=len(p['cases']),protocol_sha256=sha(OUT/arm/'protocol.json')))
 write(OUT/'prepare_result.json',dict(complete=True,cases=len(p['cases']),elapsed_seconds=time.monotonic()-started,source_images_sha256=sha(OUT/'source_images.json')))
def infer(arm):
 import torch
 from tracking.flyvis_motion import FlyVisMotionEstimator
 W.OUT=OUT/arm
 if arm=='fast':
  old=FlyVisMotionEstimator.__init__
  def patched(self,*args,**kwargs):
   old(self,*args,**kwargs);before={k:v.detach().cpu().clone() for k,v in self.network.state_dict().items()}
   with torch.no_grad():self.network.nodes_time_const.mul_(.5)
   changes=[k for k,v in self.network.state_dict().items() if not torch.equal(before[k],v.cpu())];assert changes==['nodes_time_const']
  FlyVisMotionEstimator.__init__=patched
 W.infer()
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','stock','fast']);args=ap.parse_args();prepare() if args.stage=='prepare' else infer(args.stage)
