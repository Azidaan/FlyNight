"""Algebra and counterexamples before registering any real-data comparison."""
import json,hashlib,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
from tracking.channel_agreement_correspondence import agreement_channels
def pearson(a,b):
    x=a-a.mean();y=b-b.mean();return float(x@y/np.linalg.norm(x)/np.linalg.norm(y))
a=np.array([.1,.2,.4,.6,.8]);own=a+np.array([.01,-.01,0,.01,-.01]);foreign=2*a+.5
pown=pearson(a,own);pforeign=pearson(a,foreign)
co=float(agreement_channels(a[:,None],own[:,None])[0]);cf=float(agreement_channels(a[:,None],foreign[:,None])[0]);assert pforeign>pown and co>cf
rng=np.random.default_rng(20261109);error=bound=common_affine_error=0.
for _ in range(100):
    x=rng.normal(size=(39,3));y=rng.normal(size=(39,3));v=agreement_channels(x,y);mx=x.mean(0);my=y.mean(0);xc=x-mx;yc=y-my;va=(xc*xc).mean(0);vb=(yc*yc).mean(0);cov=(xc*yc).mean(0)
    direct=2*cov/(va+vb+(mx-my)**2);rho=cov/np.sqrt(va*vb);factor=2*np.sqrt(va*vb)/(va+vb+(mx-my)**2)
    error=max(error,float(abs(v-direct).max()),float(abs(v-rho*factor).max()));bound=max(bound,float(abs(v).max()));assert (factor<=1+1e-14).all()
    common_affine_error=max(common_affine_error,float(abs(agreement_channels(3*x+5,3*y+5)-v).max()))
assert error<1e-12 and common_affine_error<1e-12 and bound<=1
assert agreement_channels(a[:,None],np.ones((5,1))) is None
constant_disappearance=agreement_channels(np.c_[a,a],np.c_[np.ones(5),a]);assert constant_disappearance[0]==0 and abs(constant_disappearance[1]-1)<1e-12
assert agreement_channels(np.ones((5,1)),a[:,None]) is None
# Per-frame brightness changes in a TRUE match are penalized too: the strongest
# alternative is illumination sensitivity, not evidence of an identity benefit.
illuminated=float(agreement_channels(a[:,None],(a+.5)[:,None])[0]);assert illuminated<1 and pearson(a,a+.5)>1-1e-12
# Three affine early maps read with Pearson can be exact duplicates of input.
early_errors=[abs(pearson(scale*a+offset,scale*own+offset)-pown) for scale,offset in [(1.,0.),(-2.,3.),(.3,-4.)]];assert max(early_errors)<1e-12
result=dict(passed=True,pearson_own=pown,pearson_foreign=pforeign,agreement_own=co,agreement_foreign=cf,algebra_max_error=error,common_affine_invariance_max_error=common_affine_error,largest_abs_random_coefficient=bound,true_match_independent_brightness_score=illuminated,affine_early_channel_pearson_error=max(early_errors),constant_reference_abstains=True,disappearing_current_gives_zero=True,
    action='Candidate: replace perchannelPearson with agreement retaining its paired means/variances at the OLD fixed winner. Variance-only agreement is a separate control for the paired mean penalty. Ordinary image channels receive identical readout.',
    scope='Synthetic mathematical viability only. Not evidence that any actual foreign pair differs in mean/variance, or that preserved means are identity rather than illumination. No new core/maps/realpair estimates or fits. Affine early-map counterexample warns that new cell names alone may not add features under Pearson.',
    sources={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'tracking/channel_agreement_correspondence.py',ROOT/'tracking/channel_balanced_correspondence.py',ROOT/'tracking/channel_identity_head.py']},goal_achieved=False)
with (OUT/'math_checks.json').open('x',encoding='utf8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
print('AGREEMENT_MATHEMATICS_PASSED',pown,pforeign,co,cf,flush=True)
