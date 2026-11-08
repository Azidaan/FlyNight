"""Lazy exact feature reuse; recover only inputs/maps missing for new causal histories."""
import shutil
import numpy as np
from research_singleton_free_20261108.common import *
from research_identity_assignment_run_20261108.export_states import sequence_key
from research_identity_assignment_run_20261108.full_retina import Receptors as InputFilter

FULL=RUN/'full';OLD=BASE/'full_precision'


class Receptors(InputFilter):
    def __init__(self,video,obs):
        import torch
        self.torch=torch;self.video=video;self.obs=obs;self._gray=None
        points=np.array(read(ROOT/'runs/z13_local_peak_20261021/geometry.json')['points'],int)
        self.px=torch.from_numpy(points[:,0].copy());self.py=torch.from_numpy(points[:,1].copy())
        self.kernel=torch.ones((1,1,13,13),dtype=torch.float32,device='cpu');self.bias=torch.zeros(1,dtype=torch.float32,device='cpu')
        p=read(FULL/'protocol.json');self.shape=(p['videos'][video]['frames'],720,1280)
        self.folder=FULL/'retinal'/video;self.folder.mkdir(parents=True,exist_ok=True);self.cache={};self.reused={}

    @property
    def gray(self):
        if self._gray is None:
            import research_identity_assignment_run_20261108.full_precision_io as io
            folder=OLD if (OLD/'processed'/(self.video+'.float64')).exists() else FULL
            if folder==FULL:
                original_full,original_status=io.FULL,io.status;io.FULL=FULL;io.status=status
                try:io.prepare_video(self.video,self.obs)
                finally:io.FULL,io.status=original_full,original_status
            meta=read(folder/'processed'/(self.video+'.json'));path=folder/'processed'/(self.video+'.float64')
            assert sha(path)==meta['sha256'];self._gray=io.FrameStore(path,meta['shape'],meta['native_dtypes'])
        return self._gray

    def get(self,frame,crop,now):
        assert 0<=frame<=now<self.shape[0]
        key=hashlib.sha256(json.dumps([frame,crop],separators=(',',':')).encode()).hexdigest()[:32]
        if key in self.cache:return self.cache[key]
        path=self.folder/(key+'.npy');old=OLD/'retinal'/self.video/path.name
        if path.exists():value=np.load(path,allow_pickle=False)
        elif old.exists():
            value=np.load(old,allow_pickle=False);assert value.shape==(721,) and value.dtype==np.float16 and np.isfinite(value).all()
            shutil.copyfile(old,path);assert sha(path)==sha(old);self.reused[path.relative_to(ROOT).as_posix()]=dict(source=old.relative_to(ROOT).as_posix(),sha256=sha(old))
        else:
            value=self.render(self.gray[frame],crop)
            with path.open('xb') as f:np.save(f,value,allow_pickle=False)
        if len(self.cache)>=1000:self.cache.clear()
        self.cache[key]=value;return value


def recover_dis(video,obs):
    import cv2
    from tracking.classical_motion import OpenCVMotionEstimator
    folder=FULL/'ordinary_motion';folder.mkdir(parents=True,exist_ok=True);path=folder/(video+'.npz');marker=folder/(video+'.json')
    if marker.exists():assert sha(path)==read(marker)['sha256'];return path
    assert not path.exists();entry=read(FULL/'protocol.json')['videos'][video];source=ROOT/entry['video_path'];source_hash=sha(source)
    assert source_hash==entry['sources'][entry['video_path']]
    estimator=OpenCVMotionEstimator('dis',processing_scale=.25,sample_step=8,geometry_version='legacy_v1')
    cap=cv2.VideoCapture(str(source));assert cap.isOpened();vectors=[];points=None
    for frame in range(entry['frames']):
        ok,image=cap.read();assert ok;field=estimator.observe(image)
        if points is None:points=field.points_xy.copy()
        else:assert np.array_equal(points,field.points_xy)
        vectors.append(field.vectors_xy.copy())
        if frame%100==0:status('ordinary_motion_missing_control_recovery',video=video,frame=frame,total=entry['frames'],guard=obs.read())
    ok,_=cap.read();assert not ok;cap.release();assert sha(source)==source_hash
    values=np.asarray(vectors);assert not np.count_nonzero(values[0]) and np.isfinite(values).all()
    with path.open('xb') as f:np.savez_compressed(f,points=points,vectors=values)
    write(marker,dict(passed=True,video=video,source_sha256=source_hash,frames=entry['frames'],fps=entry['fps'],sha256=sha(path),
        processing_scale=.25,sample_step=8,geometry_version='legacy_v1',flow_gain_in_cache=1,
        units='Source pixels per adjacent frame; vector[f] is f-1->f; x right,y down; first zero.'),new=True)
    return path


class Representations:
    def __init__(self,video,kind,obs,backbone=None):
        self.video=video;self.kind=kind;self.obs=obs;self.backbone=backbone;self.renderer=None
        self.folder=FULL/'features'/video;self.folder.mkdir(parents=True,exist_ok=True);self.cache={};self.files={};self.reused={};self.new=0
        self.points=np.array(read(ROOT/'runs/z13_local_peak_20261021/geometry.json')['points'],np.float32)
        self.old_hashes={}
        for arm in ['G_R1','G_FLYVIS','G_PERM']:
            marker=OLD/'rollouts'/arm/(video+'.receipt.json')
            if marker.exists():self.old_hashes.update(read(marker)['feature_files'])
        marker=OLD/'cache_reuse'/(video+'.json')
        if marker.exists():self.old_hashes.update({n:v['sha256'] for n,v in read(marker)['features_reused'].items()})
        if kind in ['C','R1']:
            norm=read(ROOT/f'runs/z13_target_center_dev_20261018/{kind}/normalization.json');self.mean=np.array(norm['mean'])[:34];self.sd=np.array(norm['sd'])[:34]
        elif kind=='IMAGE':
            norm=read(ROOT/'runs/spatial_identity_maps_20261107/image_normalization.json');self.mean=np.array(norm['mean']);self.sd=np.where(np.array(norm['sd'])>=1e-12,norm['sd'],1)
        else:assert kind=='DIS'

    def get(self,sequence,now):
        assert all(0<=f<=now for f,_ in sequence);key=sequence_key(self.video,sequence)
        if key in self.cache:return self.cache[key]
        path=self.folder/(key+'.'+self.kind+'.npy');old=OLD/'features'/self.video/path.name;old_name=old.relative_to(ROOT).as_posix();computed=False
        if path.exists():value=np.load(path,allow_pickle=False)
        elif old_name in self.old_hashes:
            assert sha(old)==self.old_hashes[old_name];shutil.copyfile(old,path);value=np.load(path,allow_pickle=False)
            self.reused[path.relative_to(ROOT).as_posix()]=dict(source=old_name,sha256=sha(old),sequence=sequence)
        elif self.kind=='DIS':
            from research_identity_assignment_run_20261108.extract_features import interpolate_plan
            from tracking.spatial_identity_assignment import frame_points
            source=recover_dis(self.video,self.obs)
            with np.load(source,allow_pickle=False) as z:points=z['points'];vectors=z['vectors'][sequence[-1][0]]
            locations=frame_points(self.points,sequence[-1][1],391);vi,vw,valid=interpolate_plan(points,locations)
            value=np.sum(vectors[vi]*vw[:,:,None],axis=1).astype(np.float32);value[~valid]=0
            with path.open('xb') as f:np.save(f,value,allow_pickle=False)
            computed=True
        else:
            if self.renderer is None:self.renderer=Receptors(self.video,self.obs)
            if self.kind=='IMAGE':
                from research_spatial_identity_20261107.extract_reference import image_features
                from tracking.z13_readout import HexGeometry
                frame,crop=sequence[-1];geom=HexGeometry(self.points,391)
                raw=image_features(self.renderer.get(frame,crop,now).astype(float),self.points,geom.neighbors,crop,391)
                value=(raw.astype(float)-self.mean)/self.sd
            else:
                hx=np.stack([self.renderer.get(f,c,now) for f,c in sequence]).astype(np.float32)
                if self.kind=='R1':
                    from scripts.z13_target_model import receptor_features_rich
                    raw=receptor_features_rich(hx[None].astype(np.float64))[-1][0].T;value=((raw-self.mean)/self.sd).astype(np.float32)
                else:
                    import torch
                    assert self.backbone is not None
                    with torch.no_grad():
                        x=torch.from_numpy(hx[None]).to(self.backbone.dev);state=self.backbone.init(x[:,:1])
                        for j in range(1,len(hx)):state,feature=self.backbone.step(state,x[:,j-1],x[:,j])
                        value=(((feature[0].cpu().numpy().T.astype(float)-self.mean)/self.sd)[:,4:12]).astype(np.float32)
                    del x,state,feature
            with path.open('xb') as f:np.save(f,value,allow_pickle=False)
            computed=True
        assert value.shape[0]==721 and np.isfinite(value).all()
        self.new+=int(computed)
        self.files[path.relative_to(ROOT).as_posix()]=sha(path)
        if len(self.cache)>=500:self.cache.clear()
        self.cache[key]=value;return value
