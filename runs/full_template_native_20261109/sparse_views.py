"""Restore small causal retinal views; never rebuild a whole processed video."""
from collections import OrderedDict
import pickle
import numpy as np
from common import ROOT,OUT,LEGACY,read,sha


class SparseFrames:
    def __init__(self,video,*,force_initial=False):
        import cv2
        self.cv2=cv2;self.video=video;self.entry=read(LEGACY/'protocol.json')['videos'][video]
        self.path=ROOT/self.entry['video_path'];assert sha(self.path)==self.entry['sources'][self.entry['video_path']]
        self.marker=read(LEGACY/'processed'/(video+'.json'));self.shape=tuple(self.marker['shape']);self.force_initial=force_initial
        self.states=[];self.sources={self.entry['video_path']:sha(self.path)}
        for p in sorted((LEGACY/'processed'/(video+'.checkpoints')).glob('*.json')):
            m=read(p);assert m['video_sha256']==self.sources[self.entry['video_path']]
            self.states.append((int(m['next_frame']),p,m))
        self.cache=OrderedDict();self.cursor=None;self.retina=None;self.cap=None;self.resets=0;self.decoded=0

    def close(self):
        if self.cap is not None:self.cap.release();self.cap=None

    def _reset(self,frame):
        self.close();valid=[] if self.force_initial else [s for s in self.states if s[0]<=frame]
        if valid:
            start,meta,m=valid[-1];path=meta.parent/m['state_file'];assert sha(path)==m['state_sha256']
            state=pickle.loads(path.read_bytes());self.retina=state['retina']
            self.sources[path.relative_to(ROOT).as_posix()]=m['state_sha256'];self.sources[meta.relative_to(ROOT).as_posix()]=sha(meta)
        else:
            from tracking.retina import make_retina
            start=0;self.retina=make_retina('lamina',self.entry['fps'],blur_downscale=4)
        self.cap=self.cv2.VideoCapture(str(self.path));assert self.cap.isOpened()
        if start:assert self.cap.set(self.cv2.CAP_PROP_POS_FRAMES,start)
        self.cursor=start;self.resets+=1

    def get(self,frame,now):
        frame=int(frame);now=int(now);assert 0<=frame<=now<self.shape[0]
        if frame in self.cache:
            self.cache.move_to_end(frame);return self.cache[frame]
        starts=[s[0] for s in self.states if s[0]<=frame] if not self.force_initial else []
        best=max(starts,default=0)
        if self.cursor is None or self.cursor>frame or self.cursor<best:self._reset(frame)
        assert self.cursor<=frame
        while self.cursor<=frame:
            index=self.cursor;ok,image=self.cap.read();assert ok and image.shape[:2]==self.shape[1:]
            value=self.retina.process(self.cv2.cvtColor(image,self.cv2.COLOR_BGR2GRAY))
            assert str(value.dtype)==self.marker['native_dtypes'][index],(index,value.dtype,self.marker['native_dtypes'][index])
            self.cache[index]=np.array(value,copy=True)
            while len(self.cache)>64:self.cache.popitem(last=False)
            self.cursor+=1;self.decoded+=1
        return self.cache[frame]


class SparseViews:
    def __init__(self,video,*,force_initial=False):
        import torch
        self.torch=torch;self.video=video;self.frames=SparseFrames(video,force_initial=force_initial)
        self.points=np.array(read(ROOT/'runs/z13_local_peak_20261021/geometry.json')['points'],int)
        self.px=torch.from_numpy(self.points[:,0].copy());self.py=torch.from_numpy(self.points[:,1].copy())
        self.kernel=torch.ones((1,1,13,13),dtype=torch.float32,device='cpu');self.bias=torch.zeros(1,dtype=torch.float32,device='cpu')
        self.folder=OUT/'retinal'/video;self.folder.mkdir(parents=True,exist_ok=True);self.cache={};self.files={}

    @staticmethod
    def key(frame,crop):
        import hashlib,json
        return hashlib.sha256(json.dumps([int(frame),list(crop)],separators=(',',':')).encode()).hexdigest()[:32]

    def render(self,frame,crop,now):
        import cv2
        gray=self.frames.get(frame,now);x1,y1,x2,y2=map(int,crop)
        assert 0<=x1<x2<=gray.shape[1] and 0<=y1<y2<=gray.shape[0]
        # Resize native float32/float64 first, matching the archived path.
        image=cv2.resize(gray[y1:y2,x1:x2],(391,391),interpolation=cv2.INTER_AREA).astype(np.float32)/255.
        x=self.torch.from_numpy(image).reshape(1,1,391,391)
        with self.torch.inference_mode():
            y=self.torch.nn.functional.conv2d(self.torch.nn.functional.pad(x,(6,6,6,6)),self.kernel,self.bias)/169
        return y[0,0,self.py,self.px].cpu().numpy().astype(np.float16)

    def get(self,frame,crop,now,*,force_restore=False):
        key=self.key(frame,crop)
        if not force_restore and key in self.cache:return self.cache[key]
        own=self.folder/(key+'.npy');legacy=LEGACY/'retinal'/self.video/(key+'.npy')
        path=own if own.exists() else legacy
        if not force_restore and path.exists():value=np.load(path,allow_pickle=False)
        else:
            value=self.render(frame,crop,now)
            if force_restore:return value
            assert not own.exists()
            with own.open('xb') as f:np.save(f,value,allow_pickle=False)
            path=own
        assert value.shape==(721,) and value.dtype==np.float16 and np.isfinite(value).all()
        self.files[path.relative_to(ROOT).as_posix()]=sha(path)
        if len(self.cache)>=1000:self.cache.clear()
        self.cache[key]=value;return value

    def close(self):self.frames.close()
