"""Three-frame CPU execution through real FlyVis and the tracker, no training."""
from pathlib import Path
import os,sys,json,hashlib,time
os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
OUT=Path(__file__).resolve().parent
from tracking.windows_memory_job import MemoryJob
job=MemoryJob(4*1024**3)
import torch,numpy as np,cv2
torch.set_default_device('cpu');torch.set_num_threads(2);cv2.setNumThreads(1)
torch.cuda.is_available=lambda:False
torch.cuda.device_count=lambda:0
def forbidden(*args,**kwargs):raise RuntimeError('CPU-only verification forbids CUDA initialization')
torch.cuda._lazy_init=forbidden
from tracking.flyvis_motion import FlyVisMotionEstimator
from tracking.flyvis_tracker import FlyVisTracker
def params(model):
    h=hashlib.sha256()
    for name,p in model.named_parameters():h.update(name.encode());h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()
start=time.monotonic()
model=FlyVisMotionEstimator(device='cpu',source_fps=30.)
before=(params(model.network),params(model.decoder))
tracker=FlyVisTracker();rows=[]
for i in range(3):
    frame=np.full((180,320),127,np.uint8);frame[50:90,40:70]=210;frame[90:130,220:250]=50
    field=model.observe(frame)
    boxes=np.asarray([[40,50,70,90],[220,90,250,130]],dtype=np.float32)
    tracks=tracker.update(boxes,np.asarray([.95,.95],np.float32),np.asarray([2,2]),motion_field=field)
    assert tracks.shape[0]==2 and np.isfinite(tracks).all()
    assert np.isfinite(field.vectors_xy).all()
    rows.append(dict(frame=i,ids=tracks[:,4].tolist(),track_shape=list(tracks.shape),
        baseline_shape=list(model.baseline_activity.shape),
        activity_shape=None if model.last_activity is None else list(model.last_activity.shape)))
assert rows[0]['ids']==rows[1]['ids']==rows[2]['ids']
assert before==(params(model.network),params(model.decoder))
assert not torch.cuda.is_initialized()
report=dict(passed=True,frames=3,model='flow/0000/000',mode=model.mode,fade_in_seconds=model.fade_in_seconds,
    source_fps=model.source_fps,geometry_version=model.geometry_version,parameters_unchanged=True,
    cuda_initialized=False,rows=rows,seconds=time.monotonic()-start,job=job.read(),
    scope='Real FlyVis eye/core/decoder -> motion field -> tracker. Fixed synthetic detections; detector inference and HOTA are not tested.')
(OUT/'pipeline_smoke.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report),flush=True)
