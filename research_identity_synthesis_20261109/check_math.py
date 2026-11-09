"""Bounded mathematical checks of proposed readers, no real-data fitting."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
import sys,json,hashlib
from pathlib import Path
from datetime import datetime,timedelta,timezone
ROOT=Path(__file__).resolve().parents[1];OUT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'runs/flyvis_repairs_20261106/unification_20261107'))
from system_memory_guard import guard_entrypoint
remaining=(datetime(2026,11,9,23,54,tzinfo=timezone(timedelta(hours=3)))-datetime.now(timezone(timedelta(hours=3)))).total_seconds();assert remaining>0;guard_entrypoint(timeout=min(180,remaining))
import numpy as np
protocol=json.loads((OUT/'math_protocol.json').read_text(encoding='utf8'))
for p,h in protocol['sources'].items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
# A linear rule can reverse an incorrect uniform mean. This does not say
# these weights can be estimated reliably on a small actual dataset.
own=np.array([1.,-1.]);foreign=np.array([0.,.5]);w=np.array([1.,0.]);assert own.mean()<foreign.mean() and w@own>w@foreign
# Ridge-logistic Hessian: X' diag(p(1-p)) X + lambda diag(1,..,1,0).
# It is positive definite at finite logits: any nonzero weight direction
# has positive ridge energy; a pure intercept has positive likelihood energy.
rng=np.random.default_rng(42);x=np.c_[rng.normal(size=(45,34)),np.ones(45)];theta=rng.normal(size=35);z=x@theta;pr=np.exp(-np.logaddexp(0.,-z));h=x.T@((pr*(1-pr))[:,None]*x)+np.diag([1.]*34+[0.]);minimum=float(np.linalg.eigvalsh(h).min());assert minimum>0
original=np.array([0.,1.,2.]);renumbered=np.array([0.,9.,18.]);mad=lambda a:float(np.median(abs(a-np.median(a))));assert mad(original)!=mad(renumbered)
physical=np.array([[-1.,0.],[0.,0.],[1.,0.]]);same=physical.copy();assert np.array_equal(physical,same)
# A foreign object with an exactly copied visible pattern gives exactly the
# same per-channel score/peak. Zero disagreement therefore is not identity.
peaks=np.array([[0.,1.,0.],[0.,1.,0.],[0.,1.,0.]]);foreign_peaks=peaks.copy();assert np.array_equal(np.argmax(peaks,axis=1),np.argmax(foreign_peaks,axis=1))
out=dict(passed=True,uniform_means=dict(own=float(own.mean()),foreign=float(foreign.mean())),linear_example=dict(own=float(w@own),foreign=float(w@foreign)),ridge_logistic_hessian_min_eigenvalue=minimum,shift_index_mad_original=mad(original),shift_index_mad_renumbered=mad(renumbered),physical_points_unchanged=True,zero_peak_dispersion_not_identity=True,real_data_training=False,new_real_scores=False,conclusion='Mathematical viability is not statistical reliability. Do not infer actual channel dynamics or guarantee identity from these constructed examples.')
with (OUT/'math_checks.json').open('x',encoding='utf8') as f:json.dump(out,f,ensure_ascii=False,indent=2);f.write('\n')
print('Mathematical prerequisites checked, no real-data fitting',flush=True)
