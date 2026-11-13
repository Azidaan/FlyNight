"""Independent CPU calculations; no project inference/helper imports."""
from itertools import product
import numpy as np

def pair_iou(first, second):
    first,second=[float(x) for x in first],[float(x) for x in second]
    width=max(0.0,min(first[2],second[2])-max(first[0],second[0]))
    height=max(0.0,min(first[3],second[3])-max(first[1],second[1]))
    intersection=width*height
    area=lambda b:max(0.0,b[2]-b[0])*max(0.0,b[3]-b[1])
    union=area(first)+area(second)-intersection
    return intersection/union if union>0 else 0.0

def assignment_optimum(rois, candidates):
    """Exhaustive check for small diagnostic requested-ROI sets, independent of Hungarian."""
    matrix=np.asarray([[pair_iou(a,b) for b in candidates] for a in rois],dtype=np.float64)
    assert len(rois)<=4 and (len(candidates)+1)**len(rois)<=2_000_000
    optimum=0.0
    optimal=[]
    for assignment in product(range(-1,len(candidates)),repeat=len(rois)):
        real=[index for index in assignment if index>=0]
        if len(real)!=len(set(real)):continue
        if any(index>=0 and matrix[row,index]<=0 for row,index in enumerate(assignment)):continue
        value=sum(matrix[row,index] for row,index in enumerate(assignment) if index>=0)
        if value>optimum+1e-12:
            optimum=value
            optimal=[assignment]
        elif abs(value-optimum)<=1e-12:
            optimal.append(assignment)
    return dict(iou=matrix,objective=float(optimum),optimal_assignments=optimal)

def coverage(mask, target, others=None):
    mask=np.asarray(mask)
    target=np.asarray(target)
    assert mask.dtype==np.bool_ and target.dtype==np.bool_ and mask.shape==target.shape
    total=int(mask.sum())
    target_total=int(target.sum())
    target_overlap=int(np.count_nonzero(mask&target))
    result=dict(mask_pixels=total,target_core_pixels=target_total,target_core_intersection=target_overlap,target_core_coverage=target_overlap/target_total if target_total else None,mask_fraction_in_target_core=target_overlap/total if total else None)
    for name,other in (others or {}).items():
        other=np.asarray(other)
        assert other.dtype==np.bool_ and other.shape==mask.shape
        overlap=int(np.count_nonzero(mask&other))
        result[name]=dict(intersection=overlap,mask_fraction=overlap/total if total else None,core_coverage=overlap/int(other.sum()) if other.any() else None)
    return result

def cosine(first,second):
    a,b=np.asarray(first,dtype=np.float64).ravel(),np.asarray(second,dtype=np.float64).ravel()
    assert a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all()
    an,bn=float(np.sqrt(np.dot(a,a))),float(np.sqrt(np.dot(b,b)))
    return float(np.dot(a,b)/(an*bn)) if an>0 and bn>0 else None

def timing(samples_ms):
    values=sorted(float(x) for x in samples_ms)
    assert values and all(np.isfinite(values)) and min(values)>=0
    def quantile(q):
        position=(len(values)-1)*q
        lower=int(position)
        upper=min(lower+1,len(values)-1)
        return values[lower]+(position-lower)*(values[upper]-values[lower])
    return dict(count=len(values),min_ms=min(values),median_ms=quantile(.5),p95_ms=quantile(.95),max_ms=max(values),mean_ms=sum(values)/len(values))
