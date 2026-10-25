"""Observable competition at the unchanged first-association cost matrix."""
import numpy as np


def assignment_context(costs,row,choices):
    matrix=np.asarray(costs,float)
    if matrix.ndim!=2 or not 0<=row<len(matrix) or not np.isfinite(matrix).all():raise ValueError('Finite original assignment matrix required')
    own=matrix[row];minimum=float(own.min()) if len(own) else 1.
    result=[]
    for detection in choices:
        if detection==-1:
            result.append([minimum,1.,minimum,0.]);continue
        if not 0<=detection<len(own):raise ValueError('Real detection index or explicit unmatched required')
        competitors=np.delete(matrix[:,detection],row)
        alternatives=np.delete(own,detection)
        competing=float(competitors.min()) if len(competitors) else 1.
        alternative=float(alternatives.min()) if len(alternatives) else 1.
        result.append([float(own[detection]),competing,alternative,float(own[detection]-competing)])
    return np.asarray(result,np.float32).reshape(-1,4)


def assignment_state_context(costs,row,choices,baseline_assignment):
    assigned=np.asarray(baseline_assignment)
    if assigned.shape!=(len(costs),) or assigned.dtype.kind not in 'iu':raise ValueError('Observable original assignment vector required')
    chosen=int(assigned[row]);width=np.asarray(costs).shape[1]
    if chosen < -1 or chosen>=width:raise ValueError('Invalid original assignment')
    cost_features=assignment_context(costs,row,choices)
    state=np.asarray([[float(chosen>=0),float(chosen==detection)] for detection in choices],np.float32).reshape(-1,2)
    return np.concatenate([cost_features,state],1)


def make_context_ranker(torch,activity=True):
    from tracking.z13_filter_descriptor import make_ranker
    model=make_ranker(torch,activity=activity)
    model.score[0]=torch.nn.Linear(94,32)
    return model
