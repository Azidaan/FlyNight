"""Observed-only one-sided background residual, with unchanged Gaussian targets.

Foreground residual is s-y; background residual is max(0,s-y). This is a
convex squared-hinge variant inspired by DiMP, not its exact max(0,s)-y loss.
"""
import numpy as np
from tracking.z13_balanced_ridge import fit_observed,spatial_weights


def objective(theta,z,target,mask,q,lam):
    delta=z@theta-target;active=mask|(delta>0)
    residual=np.where(mask,delta,np.maximum(delta,0.))
    penalty=np.r_[np.full(z.shape[1]-1,lam),0.]
    value=.5*float(q@(residual*residual))+.5*float((penalty*theta)@theta)
    gradient=z.T@(q*residual)+penalty*theta
    return value,gradient,active,penalty


def fit_hinge(features,target,mask,lam):
    x=np.asarray(features,np.float64);y=np.asarray(target,np.float64);mask=np.asarray(mask,bool)
    initial=fit_observed(x,y,mask,lam,True);mu,w,b=initial
    z=np.column_stack([x-mu,np.ones(len(x))]);q=spatial_weights(mask);theta=np.r_[w,b]
    initial_value,initial_gradient,_,_=objective(theta,z,y,mask,q,lam)
    tolerance=1e-8*(1+float(np.max(np.abs(initial_gradient))))
    losses=[initial_value]
    for iteration in range(101):
        value,gradient,active,penalty=objective(theta,z,y,mask,q,lam)
        gradient_max=float(np.max(np.abs(gradient)))
        if gradient_max<=tolerance:
            return (mu,theta[:-1],float(theta[-1])),dict(iterations=iteration,gradient_max=gradient_max,tolerance=tolerance,initial_objective=initial_value,final_objective=value,monotone=all(a>=b-1e-14 for a,b in zip(losses,losses[1:])))
        if iteration==100:raise ArithmeticError('One-sided target solver did not converge; keep scientific tolerance')
        za=z[active];hessian=za.T@(q[active,None]*za)+np.diag(penalty)
        direction=np.linalg.solve(hessian,-gradient);slope=float(gradient@direction)
        if not np.isfinite(direction).all() or slope>=0:raise ArithmeticError('Invalid observed Newton direction')
        step=1.
        for backtrack in range(31):
            proposal=theta+step*direction;next_value=objective(proposal,z,y,mask,q,lam)[0]
            if next_value<=value+1e-4*step*slope:
                theta=proposal;losses.append(next_value);break
            step*=.5
        else:raise ArithmeticError('Observed-only line search failed')
