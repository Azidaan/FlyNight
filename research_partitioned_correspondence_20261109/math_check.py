"""Pre-plan algebra and counterexamples only; no fit, core, video or pair selection."""
import os,sys,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=Path(__file__).resolve().parent
os.environ['CUDA_VISIBLE_DEVICES']='';sys.path.insert(0,str(ROOT))
import numpy as np
from tracking.partitioned_channel_correspondence import reference_parts,local_channel_vector,parts_at_match
from tracking.channel_identity_head import objective
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
points=np.c_[np.arange(6.),np.zeros(6)];cells=np.arange(6);box=[0,0,6,1];parts=reference_parts(points,cells,box)
a=np.tile(np.array([-1.,0,1,-1,0,1])[:,None],(1,34))
b=np.tile(np.array([-1.,0,1,1,0,-1])[:,None],(1,34));c=-b
global_b=local_channel_vector(a,b);global_c=local_channel_vector(a,c)
np.testing.assert_array_equal(global_b,global_c);assert np.max(abs(global_b))==0
pb=parts_at_match(a,b,parts['groups'],global_b);pc=parts_at_match(a,c,parts['groups'],global_c)
assert pb['valid'] and pc['valid'];np.testing.assert_allclose(pb['local_channels'],np.array([[1]*34,[-1]*34]),atol=1e-14,rtol=0)
assert np.dot(np.r_[np.zeros(34),np.ones(34),-np.ones(34)],pb['channels'])>0
assert np.dot(np.r_[np.zeros(34),np.ones(34),-np.ones(34)],pc['channels'])<0
permuted=np.array([5,1,3,0,2,4]);pp=reference_parts(points,permuted,box)
assert all(np.array_equal(cells[x],permuted[y]) for x,y in zip(parts['groups'],pp['groups']))
assert reference_parts(points,cells[:5],box) is None
constant=a.copy();constant[:3]=0;assert not parts_at_match(constant,constant,parts['groups'],local_channel_vector(constant,constant))['valid']
# Exact duplicate-feature control: same predictions, objective and stationarity,
# evaluated at already-saved weights. No ready model is fitted again.
train=read(ROOT/'research_domain_matched_reader_20261109/features.json')['records']
d=read(ROOT/'runs/domain_matched_native_20261109/deployment.json');stats=d['geometry_normalization'];t=np.array(d['models']['C34']['theta'])
g=(np.array([r['geometry'] for r in train])-stats['mean'])/stats['scale'];raw=np.array([r['variants']['C34']['channels'] for r in train])
y=np.array([r['label']=='own' for r in train],int);sw=len(y)/(2*np.bincount(y)[y]);x=np.c_[g,raw];xx=np.c_[g,np.tile(raw,(1,3))/np.sqrt(3)]
tt=np.r_[t[:7],np.tile(t[7:-1],3)/np.sqrt(3),t[-1]]
l,grad,_=objective(t,x,y,sw);ll,gg,_=objective(tt,xx,y,sw)
pred_error=float(np.max(abs(x@t[:-1]+t[-1]-xx@tt[:-1]-tt[-1])));loss_error=abs(l-ll)
gradient_error=max(float(np.max(abs(grad[:7]-gg[:7]))),float(abs(grad[-1]-gg[-1])),float(np.max(abs(np.tile(grad[7:-1],3)/np.sqrt(3)-gg[7:-1]))))
assert max(pred_error,loss_error,gradient_error)<1e-12
out=dict(passed=True,synthetic_global_identical=True,synthetic_local_distinguishable=True,partition_order_invariant=True,insufficient_or_constant_part_stays_undefined=True,
    duplicate_control=dict(prediction_error=pred_error,loss_error=loss_error,gradient_error=gradient_error,new_fits=0),
    strict_convexity='For a nonzero perturbation (v,b), Hessian quadratic form is mean(sw*p*(1-p)*(Xv+b)^2)+||v||^2. Positive for v!=0 by penalty and v=0,b!=0 by logistic curvature. Finite data and both classes, freeintercept; no generalization guarantee.',
    scope='Counterexample proves non-injectivity of global channel correlations and possible additional local information. Does NOT prove these maps carry identity, classify real cars, or improve ID/HOTA. Does not select any pair, threshold, neuron or split from labels.',
    sources={p.relative_to(ROOT).as_posix():sha(p) for p in [Path(__file__),ROOT/'tracking/partitioned_channel_correspondence.py',ROOT/'tracking/channel_balanced_correspondence.py',ROOT/'tracking/channel_identity_head.py',ROOT/'research_domain_matched_reader_20261109/features.json',ROOT/'runs/domain_matched_native_20261109/deployment.json']},goal_achieved=False)
with (OUT/'math_checks.json').open('x',encoding='utf8') as f:json.dump(out,f,ensure_ascii=False,indent=2)
print('PARTITION_MATH_PASSED',pred_error,loss_error,gradient_error,flush=True)
