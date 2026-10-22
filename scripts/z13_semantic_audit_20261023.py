"""Bind training-only candidate rows to original annotated object identities."""
from pathlib import Path
import numpy as np
from scripts.z13_candidate_reserve_20261021 import read,sha
from tracking.z13_semantic_sampling import semantic_negatives
ROOT=Path(__file__).resolve().parents[1]
VALID={'pedestrian','rider','car','bus','truck','train','motorcycle','bicycle'}


def metadata(videos,labels):
    from scripts.z13_data import raw_index
    byvideo={v:[] for v in videos}
    for i,l in enumerate(labels):byvideo[l['video']].append((i,l))
    answer=[None]*len(labels);offset=0;manifest={};checked=0
    for video in videos:
        paths=[ROOT/f'runs/z13_data/{video}.npz',ROOT/f'runs/train_cache/{video}/alignment.json',ROOT/f'data/bdd100k/labels/box_track_20/train/{video}.json']
        for path in paths:manifest[path.relative_to(ROOT).as_posix()]=sha(path)
        truth=read(paths[2]);alignment=read(paths[1]);vf=[m['video_frame_index'] for m in alignment['matches']]
        assert len(truth)==len(vf)
        objects=raw_index(truth,vf,VALID);categories={}
        for values in objects.values():
            for identity,category,box in values:
                if identity in categories:assert categories[identity]==category
                categories[identity]=category
        with np.load(paths[0]) as z:
            identity=z['ep_identity'];off=z['ep_offset'];frame=z['fr_frame'];pert=z['ep_pert']
            for index,l in byvideo[video]:
                e=l['episode']-offset;assert 0<=e<len(identity) and not pert[e]
                t=int(frame[int(off[e])+4+l['pos']]);items=objects.get(t,[]);target=str(identity[e])
                if l['kind']==1:
                    box=[b for ident,_,b in items if ident==target];assert len(box)==1,(video,e,target,t)
                    assert np.max(np.abs((box[0][:2]+box[0][2:])/2-l['gt']))<.001,(video,e,t)
                    checked+=1
                answer[index]=(target,categories.get(target),items,t)
            offset+=len(identity)
    assert all(x is not None for x in answer)
    return answer,manifest,checked


def audit(labels,centers,masks,meta):
    negative=np.zeros(masks.shape,bool);pervideo={}
    for i,(l,(identity,category,objects,t)) in enumerate(zip(labels,meta)):
        n=int(masks[i].sum());negative[i,:n]=semantic_negatives(centers[i,:n],l,identity,category,objects)
        count=pervideo.setdefault(l['video'],dict(rows=0,hard_rows=0,hard_candidates=0))
        count['rows']+=1;count['hard_rows']+=int(negative[i].any());count['hard_candidates']+=int(negative[i].sum())
    return negative,dict(hard_rows=int(negative.any(1).sum()),hard_candidates=int(negative.sum()),per_video=pervideo)
