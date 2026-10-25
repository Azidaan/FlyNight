"""Observable region inputs for a learned FlyVis readout; no annotation API."""
import numpy as np


def regions(geom, crop, prediction, boxes, confidence, classes, track_class):
    from tracking.z13_readout import to_eye
    boxes=np.asarray(boxes,float).reshape(-1,4);confidence=np.asarray(confidence,float)
    classes=np.asarray(classes,int);crop=np.asarray(crop,float)
    if confidence.shape!=(len(boxes),) or classes.shape!=confidence.shape:
        raise ValueError('Aligned real detection inputs required')
    points=geom.points;extent=np.maximum(crop[2:]-crop[:2],1)
    masks=[np.ones(len(points),bool)];geometry=[[0,0,0,0,0,0,0,1]];choices=[-1]
    for j,(box,conf,category) in enumerate(zip(boxes,confidence,classes)):
        a=np.asarray(to_eye(box[:2],crop,geom.eye_size));b=np.asarray(to_eye(box[2:],crop,geom.eye_size));c=(a+b)/2
        mask=((points>=a)&(points<=b)).all(1)
        supported=int(category)==int(track_class) and bool(np.all(c>=0) and np.all(c<=geom.eye_size)) and np.linalg.norm(points[geom.nearest(c)]-c)<=.6*geom.spacing and bool(mask.any())
        if not supported:continue
        relative=((box[:2]+box[2:])/2-np.asarray(prediction))/extent
        size=np.maximum(box[2:]-box[:2],1)/extent
        masks.append(mask);choices.append(j)
        geometry.append([*relative,*np.log(size),float(conf),float(np.linalg.norm(relative)),float(mask.mean()),0])
    return choices,np.asarray(masks,bool),np.asarray(geometry,np.float32)


def pool_activity(activity, masks):
    x=np.asarray(activity,float);m=np.asarray(masks,bool)
    if x.ndim!=2 or m.ndim!=2 or m.shape[1]!=len(x) or not m.any(1).all() or not np.isfinite(x).all():
        raise ValueError('Finite column features and nonempty region masks required')
    return np.stack([x[one].mean(0) for one in m]).astype(np.float32)


def apply_policy(logits, threshold):
    z=np.asarray(logits,float)
    if z.ndim!=1 or len(z)==0 or not np.isfinite(z).all():raise ValueError('Finite action logits required')
    k=int(z.argmax());offset=max(0.,float(z.max()))
    probability=float(np.exp(z[k]-offset)/(np.exp(-offset)+np.exp(z-offset).sum()))
    # Baseline unchanged has fixed logit0. No forced action on a negative maximum.
    return (k if z[k]>0 and probability>=threshold else -1),probability
