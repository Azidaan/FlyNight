"""Preserve Retina's native per-frame dtype and bound input IO memory."""
import os,pickle
from collections import OrderedDict
import numpy as np
from research_identity_assignment_run_20261108.common import *
from research_identity_assignment_run_20261108.full_retina import Receptors as OriginalReceptors
from research_identity_assignment_run_20261108.full_planner import Representations as OriginalRepresentations

FULL=RUN/'full_precision'


class FrameStore:
    def __init__(self,path,shape,dtypes):self.path=Path(path);self.shape=tuple(shape);self.dtypes=dtypes;self.cache=OrderedDict()
    def __getitem__(self,frame):
        frame=int(frame)
        if frame in self.cache:self.cache.move_to_end(frame);return self.cache[frame]
        size=self.shape[1]*self.shape[2];value=np.memmap(self.path,mode='r',dtype=np.float64,offset=frame*size*8,shape=self.shape[1:])
        if self.dtypes[frame]=='float32':value=np.asarray(value,dtype=np.float32)
        else:assert self.dtypes[frame]=='float64'
        self.cache[frame]=value
        if len(self.cache)>4:self.cache.popitem(last=False)
        return value


class Receptors(OriginalReceptors):
    def __init__(self,video):
        import torch
        self.torch=torch;self.video=video;self.points=np.array(read(ROOT/'runs/z13_local_peak_20261021/geometry.json')['points'],int)
        self.kernel=torch.ones((1,1,13,13),dtype=torch.float32,device='cpu');self.bias=torch.zeros(1,dtype=torch.float32,device='cpu')
        self.px=torch.from_numpy(self.points[:,0].copy());self.py=torch.from_numpy(self.points[:,1].copy())
        marker=read(FULL/'processed'/(video+'.json'));self.shape=tuple(marker['shape'])
        self.gray=FrameStore(FULL/'processed'/(video+'.float64'),self.shape,marker['native_dtypes'])
        self.folder=FULL/'retinal'/video;self.folder.mkdir(parents=True,exist_ok=True);self.cache={}


class Representations(OriginalRepresentations):
    def __init__(self,video,kind,backbone=None,*,archive=None):
        self.video=video;self.kind=kind;self.backbone=backbone;self.archive=archive;self.cache={};self.files={}
        self.renderer=None if archive else Receptors(video)
        self.folder=FULL/'features'/video;self.folder.mkdir(parents=True,exist_ok=True)
        norm=read(ROOT/f'runs/z13_target_center_dev_20261018/{kind}/normalization.json')
        self.mean=np.array(norm['mean'])[:34];self.sd=np.array(norm['sd'])[:34]


def prepare_video(video,obs):
    import cv2
    from tracking.retina import make_retina
    p=read(FULL/'protocol.json');entry=p['videos'][video];source=ROOT/entry['video_path'];assert sha(source)==entry['sources'][entry['video_path']]
    folder=FULL/'processed';folder.mkdir(parents=True,exist_ok=True);path=folder/(video+'.float64');marker=folder/(video+'.json')
    if marker.exists():assert path.stat().st_size==read(marker)['bytes'];return
    checkpoints=folder/(video+'.checkpoints');checkpoints.mkdir(exist_ok=True)
    candidates=sorted(checkpoints.glob('*.json'));dtypes=[];start=0;hasher=hashlib.sha256()
    if candidates:
        state_meta=read(candidates[-1]);state_path=checkpoints/state_meta['state_file'];assert sha(state_path)==state_meta['state_sha256']
        state=pickle.loads(state_path.read_bytes());retina=state['retina'];start=state_meta['next_frame'];dtypes=state['native_dtypes']
        assert state_meta['video_sha256']==entry['sources'][entry['video_path']]
        with path.open('rb') as f:
            remaining=state_meta['bytes']
            while remaining:
                data=f.read(min(1024*1024,remaining));assert data;hasher.update(data);remaining-=len(data)
        assert hasher.hexdigest()==state_meta['prefix_sha256']
        if path.stat().st_size>state_meta['bytes']:
            tail=checkpoints/('uncommitted_after_'+str(start)+'.tail')
            with path.open('rb') as f,tail.open('xb') as out:
                f.seek(state_meta['bytes'])
                for data in iter(lambda:f.read(1024*1024),b''):out.write(data)
            with path.open('r+b') as f:f.truncate(state_meta['bytes'])
        stream=path.open('ab')
    else:
        assert not path.exists(),'Uncheckpointed partial input must be preserved explicitly.'
        retina=make_retina('lamina',entry['fps'],blur_downscale=4);stream=path.open('xb')
    cap=cv2.VideoCapture(str(source));assert cap.isOpened()
    for _ in range(start):ok,_image=cap.read();assert ok
    shape=(entry['frames'],720,1280)
    with stream:
        for frame in range(start,entry['frames']):
            ok,image=cap.read();assert ok and image.shape[:2]==(720,1280)
            processed=retina.process(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY));native=str(processed.dtype);assert native in ['float32','float64']
            dtypes.append(native);data=np.asarray(processed,dtype=np.float64,order='C').tobytes();assert len(data)==720*1280*8
            stream.write(data);hasher.update(data)
            if (frame+1)%100==0 or frame+1==entry['frames']:
                stream.flush();os.fsync(stream.fileno())
                name=f'{frame+1:06d}';state_path=checkpoints/(name+'.pkl')
                with state_path.open('xb') as f:pickle.dump(dict(retina=retina,native_dtypes=dtypes),f,protocol=5)
                write(checkpoints/(name+'.json'),dict(next_frame=frame+1,bytes=stream.tell(),state_file=state_path.name,
                    state_sha256=sha(state_path),prefix_sha256=hasher.hexdigest(),video_sha256=entry['sources'][entry['video_path']]),new=True)
                status('full_native_precision_preparation',video=video,frame=frame+1,total=entry['frames'],guard=obs.read())
    ok,_=cap.read();assert not ok;cap.release();assert sha(source)==entry['sources'][entry['video_path']]
    assert path.stat().st_size==entry['frames']*720*1280*8 and sha(path)==hasher.hexdigest()
    write(marker,dict(passed=True,shape=shape,bytes=path.stat().st_size,sha256=hasher.hexdigest(),native_dtypes=dtypes,
        storage='float64 lossless container; each frame restored to its native dtype before cv2.resize.',
        input_video_sha256=entry['sources'][entry['video_path']],causal_access='FrameStore is read only through get(frame,crop,now), which asserts frame<=now.'),new=True)
