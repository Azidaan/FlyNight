"""Geometry proposals and centered inter-track costs; no ground-truth interface."""
from __future__ import annotations
import numpy as np
from scipy.spatial import cKDTree
from ultralytics.trackers.utils import matching
from tracking.xywh_bytetrack import XYWHByteTrack


def axis_grid(low,high,step,mean=None):
    if high<low:return np.empty(0)
    a=np.r_[np.arange(low,high+1e-10,step),high]
    if mean is not None and low<=mean<=high:a=np.r_[a,mean]
    return np.unique(a)


def frame_points(points,crop,eye_size):
    crop=np.asarray(crop,float)
    return crop[:2]+np.asarray(points)*(crop[2:]-crop[:2])/eye_size


def geometric_match(mean,covariance,history,current_crop,box,geometry):
    """One feature-independent minimum-residual transform on the frozen grid."""
    mu=np.asarray(mean,float);cov=np.asarray(covariance,float);box=np.asarray(box,float)
    old=history[-1];b0=np.asarray(old['box'],float);crop0=np.asarray(old['crop'],float)
    extent=b0[2:]-b0[:2];size=box[2:]-box[:2]
    p0=frame_points(geometry['points'],crop0,geometry['eye_size'])
    pt=frame_points(geometry['points'],current_crop,geometry['eye_size'])
    refmask=((p0>=b0[:2])&(p0<=b0[2:])).all(1)
    current_ids=np.flatnonzero(((pt>=box[:2])&(pt<=box[2:])).all(1))
    spacing0=geometry['spacing']*(crop0[2]-crop0[0])/geometry['eye_size']
    spacingt=geometry['spacing']*(current_crop[2]-current_crop[0])/geometry['eye_size']
    result=dict(valid=False,current_nodes=len(current_ids),reference_nodes=int(refmask.sum()))
    if min(extent)<=0 or len(current_ids)<3 or refmask.sum()<3:return result
    scales=[];offsets=[];chosen=[];counts=[]
    for d in range(2):
        sd=np.sqrt(max(0,cov[d+2,d+2]));w=mu[d+2]
        left=mu[d]-w/2;right=mu[d]+w/2
        sl=np.sqrt(max(0,cov[d,d]+cov[d+2,d+2]/4-cov[d,d+2]))
        sr=np.sqrt(max(0,cov[d,d]+cov[d+2,d+2]/4+cov[d,d+2]))
        best=None;valid_count=0
        for full in axis_grid(max(size[d],w-2*sd),w+2*sd,spacingt/2,w):
            gamma=full/extent[d]
            off=axis_grid(0,max(0,extent[d]-size[d]/gamma),spacing0/2)
            lo=box[d]-gamma*off
            valid=(np.abs(lo-left)<=2*sl+1e-9)&(np.abs(lo+full-right)<=2*sr+1e-9)
            for t,l in zip(off[valid],lo[valid]):
                valid_count+=1
                residual=((l-left)/max(sl,1e-6))**2+((l+full-right)/max(sr,1e-6))**2
                candidate=(float(residual),float(full),float(t))
                if best is None or candidate<best:best=candidate
        counts.append(valid_count)
        if best is None:result['position_axis_counts']=counts;return result
        scales.append(best[1]/extent[d]);offsets.append(best[2]);chosen.append(best)
    projected=b0[:2]+offsets+(pt[current_ids]-box[:2])/scales
    distance,q=cKDTree(p0).query(projected)
    keep=((projected>=b0[:2])&(projected<=b0[2:])).all(1)&refmask[q]&(distance<=.6*spacing0)
    result.update(valid=True,scale=scales,offset=offsets,full_size=[a[1] for a in chosen],
        residual=sum(a[0] for a in chosen),position_axis_counts=counts,
        p=current_ids[keep].tolist(),q=q[keep].tolist(),reference_ids=np.flatnonzero(refmask).tolist())
    return result


def corrected_cost(cost,votes,alpha,beta,arm,columns):
    """columns[j] has fixed rivals and their similarities, independently of labels."""
    base=np.asarray(cost);out=base.copy()
    if alpha==0 and beta==0:return out
    for i,j in votes:out[i,j]-=alpha
    if beta==0:return out
    for j,record in columns.items():
        j=int(j);rows=np.asarray(record['rows'],int)
        if len(rows)<2 or not record.get('geometry_valid',True):continue
        if arm=='G_SHIFT':out[rows,j]-=beta;continue
        values=np.array(record['scores'].get(arm,[None]*len(rows)),dtype=float)
        if len(values)!=len(rows) or not np.isfinite(values).all():continue
        values=np.clip(values,-1,1)
        out[rows,j]-=beta*(values-values.mean())
    return out


def centered_similarity(current,reference):
    x=np.asarray(current,float);y=np.asarray(reference,float)
    x=x-x.mean(0);y=y-y.mean(0)
    xx=float(np.sum(x*x));yy=float(np.sum(y*y))
    if xx<1e-12 or yy<1e-12:return None
    return float(np.clip(np.sum(x*y)/np.sqrt(xx*yy),-1,1))


def flow_similarity(current,reference,scale):
    x=np.asarray(current,float)/np.asarray(scale);y=np.asarray(reference,float)
    xx=float(np.sum(x*x));yy=float(np.sum(y*y))
    if xx<1e-12 or yy<1e-12:return None
    return float(np.clip(np.sum(x*y)/np.sqrt(xx*yy),-1,1))


class SpatialIdentityByteTrack(XYWHByteTrack):
    """Change only costs before one native assignment, preserving native ID handling."""
    def __init__(self,args,*,alpha=0.,beta=0.,arm='G',provider=None):
        super().__init__(args)
        self.identity_alpha=float(alpha);self.identity_beta=float(beta)
        self.identity_arm=arm;self.identity_provider=provider;self.identity_events=[]

    def _first_association(self,pool,detections,activated,refind):
        if self.identity_alpha==0 and self.identity_beta==0:
            return super()._first_association(pool,detections,activated,refind)
        original=self.get_dists(pool,detections)
        plan=self.identity_provider(self,pool,detections,original) if self.identity_provider else dict(votes=[],columns={})
        cost=corrected_cost(original,plan['votes'],self.identity_alpha,self.identity_beta,self.identity_arm,plan['columns'])
        matches,ut,ud=matching.linear_assignment(cost,thresh=self.args.match_thresh)
        self.identity_events.append(dict(frame=self.frame_id,track_ids=[int(t.track_id) for t in pool],
            votes=plan['votes'],matches=np.asarray(matches).reshape(-1,2).tolist(),
            changed_costs=int(np.count_nonzero(cost!=original))))
        self._apply_matches(matches,pool,detections,activated,refind)
        return ut,ud
