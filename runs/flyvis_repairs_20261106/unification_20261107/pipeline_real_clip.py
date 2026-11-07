"""Real val frames -> YOLO26 -> caches -> full/tiled FlyVis -> trackers -> JSON."""
from pathlib import Path
import sys,os,json,hashlib,time
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from system_memory_guard import guard_entrypoint
job=guard_entrypoint();out=job.run_dir
passport='--passport' in sys.argv
if passport:sys.argv.remove('--passport')
reuse=len(sys.argv)>1
if reuse:
    out=Path(sys.argv[1]).resolve()
    assert out.parent==job.run_dir.parent and out.is_dir()
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch,cv2,numpy as np
torch.set_default_device('cpu');torch.set_num_threads(2);cv2.setNumThreads(1)
torch.cuda.is_available=lambda:False;torch.cuda.device_count=lambda:0
def forbidden(*a,**k):raise RuntimeError('CPU-only acceptance forbids CUDA')
torch.cuda._lazy_init=forbidden
from scripts import run_bdd_video_rate_experiment as runner

source=runner.DATA_ROOT/'videos/val'/f'{runner.DEFAULT_VIDEO_ID}.mov'
weights=ROOT/'weights/baseline_seed42/best.pt'
protocol=None
if passport:
    assert not reuse,'A fingerprinted acceptance run must build fresh caches'
    from run_fingerprint import begin,finish
    protocol=begin(out,source,weights,dict(frames=3,imgsz=960,conf=.01,iou=.7,device='cpu',
        model='flow/0000/000',sources=['flyvis_full','flyvis_tiled'],max_dt=1/50,geometry='legacy_v1',
        temporal_resampling='linear',flow_aggregation='mean',clip_codec='MJPG',torch_threads=2))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()
before=sha(weights);start=time.monotonic()
clip=out/'three_real_frames.avi';writer=None
cap=cv2.VideoCapture(str(source));assert cap.isOpened()
fps=cap.get(cv2.CAP_PROP_FPS);assert np.isfinite(fps) and fps>0
try:
    if not reuse:
        for i in range(3):
            ok,frame=cap.read();assert ok
            if writer is None:
                writer=cv2.VideoWriter(str(clip),cv2.VideoWriter_fourcc(*'MJPG'),fps,(frame.shape[1],frame.shape[0]))
                assert writer.isOpened()
            writer.write(frame)
finally:
    cap.release()
    if writer:writer.release()
detpath=out/'detections.npz'
kwargs=dict(imgsz=960,conf=.01,iou=.7,device='cpu',force=False)
detmeta=runner.build_detection_cache(clip,weights,detpath,**kwargs)
assert detmeta['frame_count']==3
before_cache=sha(detpath)
runner.build_detection_cache(clip,weights,detpath,**kwargs)
assert sha(detpath)==before_cache
detections=runner.load_detection_cache(detpath)
assert len(detections['conf'])>0 and np.isfinite(detections['xyxy']).all()
meta=[dict(videoName=runner.DEFAULT_VIDEO_ID,name=f'frame-{i}.jpg',frameIndex=i) for i in range(3)]
matches=[dict(video_frame_index=i) for i in range(3)]
results={}
for source_name in ('flyvis_full','flyvis_tiled'):
    kw=dict(fps=fps,max_dt=1/50,force=False,device='cpu')
    runner.build_motion_cache(source_name,clip,out,**kw)
    motion_path=out/(source_name+'.npz');cache_sha=sha(motion_path)
    runner.build_motion_cache(source_name,clip,out,**kw)
    assert sha(motion_path)==cache_sha
    with np.load(motion_path) as motion:assert np.isfinite(motion['vectors']).all()
for name in ('bytetrack','flyvis_full_calibrated','flyvis_tiled_calibrated'):
    rows,stats=runner.run_tracker(name,detections,meta,matches,out)
    assert len(rows)==3
    assert sum(len(r['labels']) for r in rows)>0
    target=out/(name+'.json');target.write_text(json.dumps(rows,allow_nan=False),encoding='utf-8')
    assert json.loads(target.read_text())==rows
    results[name]=dict(frames=len(rows),labels=[len(r['labels']) for r in rows],stats=stats)
assert sha(weights)==before and not torch.cuda.is_initialized()
report=dict(passed=True,source_video=str(source),source_video_sha256=sha(source),clip_sha256=sha(clip),
            weights_sha256=before,weights_unchanged=True,frames=3,fps=fps,device='cpu',
            results=results,seconds=time.monotonic()-start,
            scope='Execution and cache reuse on a three-frame reencoded val clip; no full-video HOTA or benefit claim.')
report['cache_directory']=str(out)
(job.run_dir/'pipeline_real_clip.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
if protocol is not None:finish(job.run_dir,protocol)
print(json.dumps(report),flush=True)
