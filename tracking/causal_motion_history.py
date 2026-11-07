"""История реальных изменений с причинной компенсацией переноса камеры.

Это перенос, а не полная компенсация вращения/перспективы. При ненадёжной
оценке след сбрасывается; разметка и рамки объектов в оценке камеры не участвуют.
"""
import cv2
import numpy as np

class CausalMotionHistory:
    def __init__(self,width=320,height=180,horizon_frames=60,threshold=.1,min_response=.1):
        self.width=int(width);self.height=int(height);self.horizon=float(horizon_frames)
        self.threshold=float(threshold);self.min_response=float(min_response)
        self.previous=None;self.frame=None;self.recency=np.zeros((height,width),np.float32)
        self.gray=self.recency.copy();self.valid=self.recency.copy();self.camera=np.zeros(2,np.float64);self.epoch=0
        self.window=cv2.createHanningWindow((width,height),cv2.CV_32F)

    def update(self,image,frame):
        if self.frame is not None and frame<=self.frame:raise ValueError('Только последовательно увиденные кадры')
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) if image.ndim==3 else image
        current=cv2.resize(gray,(self.width,self.height),interpolation=cv2.INTER_AREA).astype(np.float32)/255.
        if self.previous is None:
            self.valid.fill(0);response=0.;shift=(0.,0.);reliable=False
        else:
            shift,response=cv2.phaseCorrelate(self.previous.copy(),current.copy(),self.window)
            reliable=np.isfinite(shift).all() and np.isfinite(response) and response>=self.min_response and abs(shift[0])<=self.width/4 and abs(shift[1])<=self.height/4
            if reliable:
                dx,dy=shift;warp=np.array([[1.,0.,dx],[0.,1.,dy]],np.float32)
                before=cv2.warpAffine(self.previous,warp,(self.width,self.height),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
                support=cv2.warpAffine(np.ones_like(self.previous),warp,(self.width,self.height),flags=cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT)
                self.recency=cv2.warpAffine(self.recency,warp,(self.width,self.height),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
                self.recency=np.maximum(self.recency-(frame-self.frame)/self.horizon,0)
                changed=(np.abs(current-before)>self.threshold)&(support>.99)
                self.recency[changed]=1.;self.recency[support<.99]=0
                self.valid=support;self.camera+=np.array(shift)
            else:
                self.recency.fill(0);self.valid.fill(0);self.epoch+=1
        self.gray=current;self.previous=current.copy();self.frame=int(frame)
        return dict(shift=np.asarray(shift,np.float64),response=float(response),reliable=bool(reliable),epoch=self.epoch,camera=self.camera.copy())

    def crop(self,box_xyxy,image_shape,size=64,context=3.):
        h,w=image_shape[:2];box=np.asarray(box_xyxy,np.float32)
        if box.shape!=(4,) or not np.isfinite(box).all() or np.any(box[2:]<=box[:2]):raise ValueError('Неверная реальная область')
        center=(box[:2]+box[2:])/2*np.array([self.width/w,self.height/h])
        extent=(box[2:]-box[:2])*context*np.array([self.width/w,self.height/h])
        scale=size/np.maximum(extent,1e-3)
        transform=np.array([[scale[0],0,size/2-center[0]*scale[0]],[0,scale[1],size/2-center[1]*scale[1]]],np.float32)
        planes=[cv2.warpAffine(x,transform,(size,size),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT) for x in [self.gray,self.recency,self.valid]]
        return np.rint(np.clip(np.stack(planes),0,1)*255).astype(np.uint8)
