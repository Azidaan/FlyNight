"""Coordinate-ramp evidence for the configured OpenCV resize transform."""
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import cv2,numpy as np
from tracking.classical_motion import OpenCVMotionEstimator
rows=[]
for width,height,scale in [(1280,720,.25),(1281,721,.25),(101,99,.25),(64,64,1.),(32,32,.99)]:
    estimator=OpenCVMotionEstimator('farneback',processing_scale=scale)
    yy,xx=np.indices((height,width),dtype=np.float32)
    xmap=cv2.resize(xx,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA)
    ymap=cv2.resize(yy,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA)
    ph,pw=xmap.shape
    gx,gy=np.meshgrid(np.arange(4,pw,8),np.arange(4,ph,8))
    independent_points=np.column_stack([xmap[gy,gx].ravel(),ymap[gy,gx].ravel()])
    raw=np.ones((ph,pw,2),np.float32)
    field=estimator._sample(raw,width,height)
    inverse=1. if (pw,ph)==(width,height) else 1./scale
    candidate_points=(np.column_stack([gx.ravel(),gy.ravel()])+.5)*inverse-.5
    rows.append(dict(source_size=[width,height],scale=scale,processed_size=[pw,ph],
                     current_points_max_abs_error=float(np.max(np.abs(field.points_xy-independent_points))),
                     candidate_points_max_abs_error=float(np.max(np.abs(candidate_points-independent_points))),
                     current_unit_flow_vector=field.vectors_xy[0].tolist(),
                     inverse_scale_from_cv2_call=inverse))
report=dict(opencv=cv2.__version__,rows=rows,
            source='https://raw.githubusercontent.com/opencv/opencv/4.13.0/modules/imgproc/src/resize.cpp',
            source_lines='3957-3968: fx/fy retained for implicit dsize; 3984-3988: same-size copy',
            source_sha256=hashlib.sha256((ROOT/'tracking/classical_motion.py').read_bytes()).hexdigest())
(OUT/'geometry_before_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
