"""Метки пар отдельно от причинных входов; неизвестное соответствие не negative."""
import numpy as np
from scipy.optimize import linear_sum_assignment

def label_detection_identities(boxes,classes,labels,class_names,threshold=.5):
    boxes=np.asarray(boxes,np.float32).reshape(-1,4)
    categories=set(class_names.values()) if isinstance(class_names,dict) else set(class_names)
    gt=[x for x in labels if x.get('category') in categories and x.get('box2d') and not x.get('attributes',{}).get('crowd',False)]
    result=[None]*len(boxes)
    if not len(boxes) or not gt:return result
    b=np.array([[x['box2d'][k] for k in ['x1','y1','x2','y2']] for x in gt],np.float32)
    lo=np.maximum(boxes[:,None,:2],b[None,:,:2]);hi=np.minimum(boxes[:,None,2:],b[None,:,2:])
    overlap=np.maximum(hi-lo,0).prod(-1)
    union=np.maximum(boxes[:,2:]-boxes[:,:2],0).prod(-1)[:,None]+np.maximum(b[:,2:]-b[:,:2],0).prod(-1)[None,:]-overlap
    score=overlap/np.maximum(union,1e-9)
    same=np.array([[class_names[int(c)]==x['category'] for x in gt] for c in classes],bool)
    score=np.where(same,score,0)
    rr,cc=linear_sum_assignment(-score)
    for i,j in zip(rr,cc):
        if score[i,j]>=threshold:result[int(i)]=str(gt[int(j)]['id'])
    return result

def identity_target(previous,current):
    known=previous is not None and current is not None
    return (int(previous==current) if known else -1),known

def pair_geometry(track,box,confidence,cost,current_frame,history_count):
    center=(np.asarray(box[:2])+box[2:])/2;size=np.maximum(np.asarray(box[2:])-box[:2],1)
    scale=np.maximum(track.mean[2:4],1);delta=(center-track.mean[:2])/scale
    ratio=np.log(size/scale);velocity=track.mean[4:6]/scale
    uncertainty=np.sqrt(np.maximum(np.diag(track.covariance)[:2],0))/scale
    return np.clip(np.asarray([cost,confidence,1.,*delta,*ratio,*velocity,(current_frame+1-track.frame_id)/30.,*uncertainty,history_count/6.,np.log(size.prod()/scale.prod())],np.float32),-10,10)
