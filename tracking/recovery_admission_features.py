"""Причинные признаки допуска: текущая область и последнее реальное наблюдение."""
import cv2
import numpy as np
from tracking.recovery_detection_support import box, iou

FEATURE_NAMES=['class_'+str(i) for i in range(8)]+[
    'score','lost_age_30','cx_w','cy_h','width_w','height_h',
    'support_iou','support_conf','base_overlap','observations_100','past_available',
    'x1_change_w','y1_change_h','x2_change_w','y2_change_h','past_age_30']

def crop(image,bb):
    h,w=image.shape[:2];x1,y1,x2,y2=np.asarray(bb,dtype=float)
    if not np.isfinite([x1,y1,x2,y2]).all():return np.zeros((384,128,3),np.uint8)
    a,b=max(0,int(np.floor(x1))),max(0,int(np.floor(y1)))
    c,d=min(w,int(np.ceil(x2))),min(h,int(np.ceil(y2)))
    if c<=a or d<=b:return np.zeros((384,128,3),np.uint8)
    return cv2.resize(image[b:d,a:c],(128,384),interpolation=cv2.INTER_LINEAR)

class CausalAdmissionMemory:
    def __init__(self):self.history={}
    def describe(self,frame,candidate,track,base,xyxy,confidence,classes,image):
        bb=box(candidate);h,w=image.shape[:2];cls=int(track.cls)
        support=[(iou(bb,np.asarray(b)),float(s)) for b,s,k in zip(xyxy,confidence,classes) if s>.1 and int(k)==cls]
        overlap,score=max(support,default=(0.,0.))
        old=self.history.get(int(track.track_id));onehot=[float(i==cls) for i in range(8)]
        previous=np.zeros((384,128,3),np.uint8)
        delta=np.zeros(4);count=0;age=0
        if old:
            previous_frame,past_bb,previous,count=old
            assert previous_frame<frame,'Область текущего/будущего кадра попала в память'
            delta=(bb-past_bb)/np.array([w,h,w,h]);age=frame-previous_frame
        values=onehot+[float(track.score),float(frame+1-track.end_frame)/30,
            float((bb[0]+bb[2])/(2*w)),float((bb[1]+bb[3])/(2*h)),
            float((bb[2]-bb[0])/w),float((bb[3]-bb[1])/h),overlap,score,
            max((iou(bb,box(b)) for b in base),default=0.),min(count,100)/100,float(old is not None),
            *delta.tolist(),age/30]
        result=np.asarray(values,np.float32);assert len(result)==len(FEATURE_NAMES) and np.isfinite(result).all()
        return result,crop(image,bb),previous,(-1 if not old else previous_frame)
    def remember(self,frame,base,image,live_ids):
        for row in base:
            ident=int(row[4]);old=self.history.get(ident);count=0 if old is None else old[3]
            self.history[ident]=(frame,np.array(row[:4],copy=True),crop(image,row[:4]),count+1)
        self.history={i:v for i,v in self.history.items() if i in live_ids}
