"""Native CPU old/new parity and reset for classical optical flow."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys,importlib.util,json,hashlib
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import cv2,numpy as np
from tracking.classical_motion import OpenCVMotionEstimator
cv2.setNumThreads(1)
spec=importlib.util.spec_from_file_location('original_classical_motion',OUT/'original/tracking/classical_motion.py')
original=importlib.util.module_from_spec(spec);sys.modules[spec.name]=original;spec.loader.exec_module(original)
first=np.random.default_rng(309).integers(0,256,(128,192),dtype=np.uint8)
first=cv2.GaussianBlur(first,(0,0),1.)
frames=[cv2.warpAffine(first,np.array([[1,0,k*4],[0,1,k*2]],np.float32),(192,128),borderMode=cv2.BORDER_REFLECT) for k in range(3)]
def run(estimator,inputs):
    estimator.reset()
    fields=[estimator.observe(image) for image in inputs]
    return np.stack([f.points_xy for f in fields]),np.stack([f.vectors_xy for f in fields])
checks=[]
arrays={}
for method in ('farneback','dis'):
    for scale in (1.,.25):
        for kind in ('uint8_gray','float32_gray','uint8_bgr'):
            inputs=[cv2.cvtColor(f,cv2.COLOR_GRAY2BGR) if kind=='uint8_bgr' else f.astype(np.float32) if kind=='float32_gray' else f for f in frames]
            old=original.OpenCVMotionEstimator(method,processing_scale=scale)
            new=OpenCVMotionEstimator(method,processing_scale=scale,geometry_version='legacy_v1')
            oldp,oldv=run(old,inputs);newp,newv=run(new,inputs)
            assert np.array_equal(oldp,newp) and np.array_equal(oldv,newv)
            resetp,resetv=run(new,inputs)
            assert np.array_equal(resetp,newp) and np.array_equal(resetv,newv)
            new.geometry_version='resize_v2';v2p,v2v=run(new,inputs)
            assert np.array_equal(v2v,oldv)
            np.testing.assert_array_equal(v2p,oldp+(.5/scale-.5))
            checks.append(dict(method=method,scale=scale,input=kind,legacy_points_exact=True,
                               legacy_vectors_exact=True,reset_exact=True,v2_vectors_exact=True,
                               v2_point_offset=.5/scale-.5))
            key=method+'_'+str(scale)+'_'+kind
            arrays[key+'_points']=v2p;arrays[key+'_vectors']=v2v
np.savez_compressed(OUT/'native_outputs.npz',**arrays)
torch=sys.modules.get('torch')
cuda_initialized=bool(torch.cuda.is_initialized()) if torch is not None else False
assert not cuda_initialized
result=dict(passed=True,opencv=cv2.__version__,checks=checks,cuda_initialized=cuda_initialized,
            scope='CPU, fixed synthetic translations; no tracker-quality claim',
            source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
            ('tracking/classical_motion.py','tracking/flyvis_tracker.py','scripts/run_bdd_video_rate_experiment.py',
             'scripts/run_flyvis_tracker.py','scripts/evaluate_motion_sources.py','scripts/build_train_caches.py','scripts/build_dis_retina.py')})
(OUT/'native_report.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
