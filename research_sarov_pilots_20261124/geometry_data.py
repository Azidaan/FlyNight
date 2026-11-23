"""Build causal association inputs from actual aligned YOLO26 detector outputs."""
from lab import ROOT, OUT, read, sha
from functools import lru_cache
from pathlib import Path
import hashlib
import zipfile
import numpy as np
from scipy.optimize import linear_sum_assignment

TRAIN=ROOT/"runs/train_cache"
VAL=ROOT/"runs/det_cache_ctl_960"
LABEL=ROOT/"data/bdd100k/labels/box_track_20"
FEATURES=14


@lru_cache(maxsize=1)
def detector_hash():
    return sha(ROOT/"weights/baseline_seed42/best.pt")


def split():
    from lab import GAPS
    with zipfile.ZipFile(ROOT/read(GAPS/"protocol.json")["labelzip"]) as archive:
        mots={Path(s).stem for s in archive.namelist() if "/rles/val/" in s and s.endswith(".json")}
    training=read(TRAIN/"selection.json")["videos"]["train"][:24]
    val=sorted((p.stem for p in (VAL/"val").glob("*.npz") if p.stem not in mots),
               key=lambda name:hashlib.sha256(name.encode()).hexdigest())[:20]
    assert len(val)==20 and not set(training)&set(val)
    return training,val


def sources(video, training):
    if training:
        d=TRAIN/video
        return [LABEL/"train"/(video+".json"),d/"detections/raw_video.npz",d/"alignment.json",d/"detections/raw_video.json"]
    return [LABEL/"val"/(video+".json"),VAL/"val"/(video+".npz")]


def ious(a,b):
    a=np.asarray(a,np.float32).reshape(-1,4)
    b=np.asarray(b,np.float32).reshape(-1,4)
    overlap=np.maximum(0,np.minimum(a[:,None,2:],b[None,:,2:])-np.maximum(a[:,None,:2],b[None,:,:2])).prod(-1)
    area1=np.maximum(a[:,2:]-a[:,:2],0).prod(-1)
    area2=np.maximum(b[:,2:]-b[:,:2],0).prod(-1)
    return overlap / np.maximum(area1[:,None]+area2[None,:]-overlap,1e-6)


def prepare_scenes(video, training):
    paths=sources(video,training)
    gt=read(paths[0])
    with np.load(paths[1],allow_pickle=False) as arr:
        if training:
            manifest=read(paths[2])["matches"]
            assert [x["reference_name"] for x in manifest]==[r["name"] for r in gt]
            assert read(paths[3])["identity"]["weights_sha256"]==detector_hash()
            raw_indices=[x["video_frame_index"] for x in manifest]
        else:
            assert list(arr["frame_names"])==[r["name"] for r in gt]
            raw_indices=range(len(gt))
        pick=(arr["conf"]>.1)&(arr["cls"]==2)
        frame=arr["frame_idx"][pick]
        boxes=arr["xyxy"][pick].astype(np.float32)
        conf=arr["conf"][pick].astype(np.float32)
    assert np.all(frame[:-1]<=frame[1:])
    history=[]
    scenes=[]
    for index,(target,raw) in enumerate(zip(gt,raw_indices)):
        lo,hi=np.searchsorted(frame,[raw,raw+1])
        b,c=boxes[lo:hi],conf[lo:hi]
        truth=[o for o in target["labels"] if o["category"]=="car" and not any((o.get("attributes") or {}).get(f,False)
                     for f in ("ignore","ignored","crowd","iscrowd"))]
        tb=[[o["box2d"][n] for n in ("x1","y1","x2","y2")] for o in truth]
        overlap=ious(tb,b)
        source={}
        if overlap.size:
            # Maximum cardinality above the existing IoU gate, then maximum overlap.
            valid=overlap>=.5
            benefit=np.where(valid,1+min(overlap.shape)+overlap,0)
            rr,cc=linear_sum_assignment(-benefit)
            source={truth[i]["id"]:int(j) for i,j in zip(rr,cc) if valid[i,j]}
        if index>=3 and index%3==0:
            last,prev,earlier=history[-1],history[-2],history[-3]
            owners=sorted(set(last)&set(prev))
            # The current identity is only used after the causal histories have been formed.
            past=np.array([[earlier.get(oid,prev[oid]),prev[oid],last[oid]] for oid in owners],np.float32).reshape(-1,3,4)
            labels=np.array([source.get(oid,-1) for oid in owners],np.int64)
            scenes.append(dict(video=video,frame=index,past=past,boxes=b,conf=c,y=labels))
        history.append({oid:b[j] for oid,j in source.items()})
        history=history[-3:]
    return scenes


def extrapolate(h):
    # Predict translation from the last two centers; preserve last observed box size.
    center=(h[:,:,:2]+h[:,:,2:])*.5
    velocity=center[:,-1]-center[:,-2]
    return h[:,-1]+np.concatenate([velocity,velocity],axis=1),center,velocity


def make_features(past,boxes,conf):
    predicted,center,velocity=extrapolate(past)
    last=past[:,-1]
    wh=np.maximum(last[:,2:]-last[:,:2],1)
    size=np.maximum(boxes[:,2:]-boxes[:,:2],1)
    bc=(boxes[:,:2]+boxes[:,2:])*.5
    delta=(bc[None,:,:]-(center[:,-1]+velocity)[:,None,:])/wh[:,None,:]
    ratio=np.log(size[None,:,:]/wh[:,None,:])
    cv=ious(predicted,boxes)
    shape=cv.shape
    motion=velocity/wh
    acc=(velocity-(center[:,-2]-center[:,-3]))/wh
    def expand(x):
        return np.broadcast_to(x,shape)
    features=np.stack([delta[:,:,0],delta[:,:,1],ratio[:,:,0],ratio[:,:,1],cv,ious(last,boxes),
        expand(conf[None,:]),expand(motion[:,0,None]),expand(motion[:,1,None]),expand(acc[:,0,None]),
        expand(acc[:,1,None]),expand(wh[:,0,None]/1280),expand(wh[:,1,None]/720),
        expand(center[:,-1,1,None]/720)],axis=-1)
    return np.ascontiguousarray(np.clip(features,-8,8),dtype=np.float32)


def assign(scores):
    rows,cols=scores.shape
    out=np.full(rows,-1,np.int64)
    if rows and cols:
        rr,cc=linear_sum_assignment(-np.c_[scores,np.zeros((rows,rows),np.float32)])
        ok=cc<cols
        rr,cc=rr[ok],cc[ok]
        positive=scores[rr,cc]>0
        out[rr[positive]]=cc[positive]
    return out
