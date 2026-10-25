"""Observable first-association admission for a complete-box proposal."""
import numpy as np


def admit(mode, assignment, row, candidate):
    assignment=np.asarray(assignment,dtype=int)
    if mode not in ('full','unmatched','free'):
        raise ValueError('Unknown preregistered admission rule')
    if candidate<0:
        return False
    if mode=='full':
        return True
    if assignment[row]>=0:
        return False
    return mode=='unmatched' or not any(j==candidate for i,j in enumerate(assignment) if i!=row)
