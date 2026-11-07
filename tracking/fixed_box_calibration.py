"""Применение уже зафиксированной числовой калибровки к выходу любого трекера."""
import copy
import numpy as np
from tracking.pixel_box_refinement import metadata,coordinates,apply_offsets

def calibrate_rows(rows,frames,fit,alpha=.5):
    assert len(rows)==len(frames) and all(b>a for a,b in zip(frames,frames[1:]))
    out=copy.deepcopy(rows);past={};records=[];metas=[]
    for ref,(row,frame) in enumerate(zip(rows,frames)):
        ids=[str(l['id']) for l in row['labels']];assert len(ids)==len(set(ids))
        for j,label in enumerate(row['labels']):
            old=past.get(str(label['id']));dt=0 if old is None else frame-old
            metas.append(metadata(label,old is not None,dt))
            records.append(dict(reference=ref,index=j,id=label['id'],frame=frame,past_frame=old,box=coordinates(label).tolist()))
        # Прежняя калибровка использовала только наличие на предыдущем запросе.
        past={str(l['id']):frame for l in row['labels']}
    m=np.asarray(metas,np.float32).reshape(-1,12)
    design=np.c_[np.ones(len(m)),np.clip((m-fit['mean'])/fit['std'],-10,10)]
    offsets=np.clip(design@fit['beta'],-.25,.25).astype(np.float32)
    for rec,delta in zip(records,offsets):
        if alpha and np.any(delta):
            out[rec['reference']]['labels'][rec['index']]['box2d']={k:float(v) for k,v in zip(['x1','y1','x2','y2'],apply_offsets(rec['box'],delta,alpha))}
    return out,records,offsets
