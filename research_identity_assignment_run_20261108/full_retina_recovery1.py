"""The fixed input box filter, implemented without importing FlyVis or its core."""
import numpy as np
from research_identity_assignment_run_20261108.common import *

FULL=RUN/'full'


class Receptors:
    def __init__(self,video):
        import torch
        self.torch=torch;self.video=video;self.points=np.array(read(ROOT/'runs/z13_local_peak_20261021/geometry.json')['points'],int)
        self.kernel=torch.ones((1,1,13,13),dtype=torch.float32,device='cpu');self.bias=torch.zeros(1,dtype=torch.float32,device='cpu')
        self.px=torch.from_numpy(self.points[:,0].copy());self.py=torch.from_numpy(self.points[:,1].copy())
        p=read(FULL/'processed'/(video+'.json'));self.shape=tuple(p['shape'])
        self.gray=np.memmap(FULL/'processed'/(video+'.float32'),mode='r',dtype=np.float32,shape=self.shape)
        self.folder=FULL/'retinal'/video;self.folder.mkdir(parents=True,exist_ok=True)
        self.cache={}

    def render(self,gray,crop):
        import cv2
        t=self.torch;x1,y1,x2,y2=crop
        image=cv2.resize(gray[y1:y2,x1:x2],(391,391),interpolation=cv2.INTER_AREA).astype(np.float32)/255.0
        x=t.from_numpy(image).reshape(1,1,391,391)
        with t.inference_mode():
            y=t.nn.functional.conv2d(t.nn.functional.pad(x,(6,6,6,6)),self.kernel,self.bias)/169
        return y[0,0,self.py,self.px].cpu().numpy().astype(np.float16)

    def get(self,frame,crop,now):
        assert 0<=frame<=now<self.shape[0]
        key=hashlib.sha256(json.dumps([frame,crop],separators=(',',':')).encode()).hexdigest()[:32]
        if key in self.cache:return self.cache[key]
        path=self.folder/(key+'.npy')
        if path.exists():value=np.load(path,allow_pickle=False)
        else:
            value=self.render(self.gray[frame],crop)
            with path.open('xb') as f:np.save(f,value,allow_pickle=False)
        # Bound the transient RAM cache; completed views remain on disk.
        if len(self.cache)>=1000:self.cache.clear()
        self.cache[key]=value;return value


def prepare_video(video,obs):
    import cv2
    from tracking.retina import make_retina
    p=read(FULL/'protocol.json');entry=p['videos'][video];folder=FULL/'processed';folder.mkdir(parents=True,exist_ok=True)
    marker=folder/(video+'.json');path=folder/(video+'.float32')
    if marker.exists():assert path.stat().st_size==read(marker)['bytes'];return
    source=ROOT/entry['video_path'];assert sha(source)==entry['sources'][entry['video_path']]
    assert not path.exists(),'Partial processed video retained; recover explicitly.'
    shape=(entry['frames'],720,1280);data=np.memmap(path,mode='w+',dtype=np.float32,shape=shape)
    retina=make_retina('lamina',entry['fps'],blur_downscale=4);cap=cv2.VideoCapture(str(source));assert cap.isOpened()
    for frame in range(shape[0]):
        ok,image=cap.read();assert ok and image.shape[:2]==(720,1280)
        data[frame]=retina.process(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY))
        if frame%100==0:
            data.flush()
            status('full_retinal_preparation',video=video,frame=frame,total=shape[0],guard=obs.read())
    ok,_=cap.read();assert not ok;cap.release();data.flush();del data
    assert sha(source)==entry['sources'][entry['video_path']]
    write(marker,dict(passed=True,shape=shape,bytes=path.stat().st_size,sha256=sha(path),
        input_video_sha256=entry['sources'][entry['video_path']],causal_access='Readers assert every requested frame <= current tracker frame.'),new=True)
