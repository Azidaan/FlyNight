"""Одно правило поддержки текущим обнаружением для разных источников кандидатов."""
import numpy as np
from scipy.optimize import linear_sum_assignment


def box(label):return np.array([label['box2d'][k] for k in ('x1','y1','x2','y2')],dtype=float)


def iou(a,b):
    left=np.maximum(a[:2],b[:2]);right=np.minimum(a[2:],b[2:])
    area=np.maximum(right-left,0).prod();den=np.maximum(a[2:]-a[:2],0).prod()+np.maximum(b[2:]-b[:2],0).prod()-area
    return float(area/den) if den>0 else 0.


def supported_candidates(base,candidates,detections,threshold=.5):
    """Сохранить рамку кандидата при взаимно однозначной поддержке свободной области.

    Свободная область — обнаружение без IoU>=threshold с текущей выдачей базы.
    Это геометрическое условие, а не утверждение о внутренних назначениях трекера.
    Никакой разметки, времени будущего или изменения входных записей.
    """
    ids={str(x['id']) for x in base}
    assert len(ids)==len(base)
    c=sorted(candidates,key=lambda x:str(x['id']))
    assert len({str(x['id']) for x in c})==len(c) and not any(str(x['id']) in ids for x in c)
    free=[d for d in detections if all(iou(box(d),box(b))<threshold for b in base)]
    scores=np.array([[iou(box(a),box(b)) if a['category']==b['category'] else 0. for b in free] for a in c]).reshape(len(c),len(free))
    if not len(c) or not len(free):return [],dict(candidates=len(c),free_detections=len(free),accepted=0)
    assert np.isfinite(scores).all()
    rows,cols=linear_sum_assignment(-scores)
    accepted=[c[row] for row,col in zip(rows,cols) if scores[row,col]>=threshold]
    return accepted,dict(candidates=len(c),free_detections=len(free),accepted=len(accepted))
