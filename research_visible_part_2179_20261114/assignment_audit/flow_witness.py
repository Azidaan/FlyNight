"""Local optical-flow witness for the two visually distinct cars; no ID classifier."""
import os,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
os.environ['TEMP']=os.environ['TMP']=str(ROOT/'.cache/tmp')
import cv2,numpy as np
cv2.setNumThreads(1)
# Regions selected from the unannotated images before this computation. They
# intentionally avoid the moving pedestrians and the other vehicle's roof.
regions={'dark_rear_window':[920,267,990,309],'red_roof':[1145,251,1215,269]}
cap=cv2.VideoCapture(str(ROOT/'data/bdd100k/videos/val/b1ca2e5d-84cf9134.mov')); frames={}
for i in range(1166):
 ok,img=cap.read();assert ok
 if 1159<=i<=1165:frames[i]=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cap.release();result={}
for name,(x1,y1,x2,y2) in regions.items():
 mask=np.zeros_like(frames[1159]);mask[y1:y2,x1:x2]=255
 p=cv2.goodFeaturesToTrack(frames[1159],maxCorners=40,qualityLevel=.01,minDistance=3,mask=mask)
 if p is None:result[name]={'count':0};continue
 first=p.copy();alive=np.ones(len(p),bool)
 for f in range(1159,1165):
  q,st,_=cv2.calcOpticalFlowPyrLK(frames[f],frames[f+1],p,None,winSize=(15,15),maxLevel=2,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01));back,st2,_=cv2.calcOpticalFlowPyrLK(frames[f+1],frames[f],q,None,winSize=(15,15),maxLevel=2,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01));alive &= st[:,0].astype(bool)&st2[:,0].astype(bool)&(np.linalg.norm((back-p)[:,0],axis=1)<1);p=q
 delta=(p-first)[:,0];result[name]=dict(region=[x1,y1,x2,y2],initial_count=len(first),surviving=int(alive.sum()),median_delta=np.median(delta[alive],axis=0).tolist() if alive.any() else None,start=first[:,0][alive].tolist(),end=p[:,0][alive].tolist(),all_surviving_delta=delta[alive].tolist())
(OUT/'flow_witness.json').write_text(json.dumps(dict(regions=regions,results=result,limit='Local pixel motion only; manually selected regions, not a general segmentation or ground-truth identity proof.'),indent=2)+'\n')
print(json.dumps(result))
