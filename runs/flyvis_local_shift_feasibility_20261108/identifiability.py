"""Can the proposed score identify a shift? Synthetic mathematics, not tracker evaluation."""
from pathlib import Path
import json,numpy as np,hashlib
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay
P=Path(__file__).resolve().parent;ROOT=P.parents[1]
g=json.loads((ROOT/'runs/z13_local_peak_20261021/geometry.json').read_text());X=np.array(g['points'],float)
dirs=np.array([(u,v) for u in range(-2,3) for v in range(-2,3) if max(abs(u),abs(v),abs(u+v))<=2]);basis=np.array([[13.,6.5],[0.,13.]]);delta=dirs@basis;true=np.array([13.,6.5]);correct=int(np.flatnonzero((dirs==[1,0]).all(1))[0]);zero=int(np.flatnonzero((dirs==[0,0]).all(1))[0])
trial=X[:,None]+delta
mask=(Delaunay(X).find_simplex(trial.reshape(-1,2)).reshape(len(X),19)>=0).all(1);Q=X[mask]
def corr(a,b):
 a=a-a.mean();b=b-b.mean();return float(a@b/np.sqrt((a@a)*(b@b)))
def score(f):
 interp=LinearNDInterpolator(X,f(X-true));reference=f(Q)
 return np.array([corr(reference,interp(Q+d)) for d in delta])
affine=lambda q:.17*q[:,0]-.23*q[:,1]+.4
pattern=lambda q:np.sin(.031*q[:,0]+.013*q[:,1])+.7*np.cos(.009*q[:,0]-.037*q[:,1])+.4*np.sin(.017*q[:,0]-.021*q[:,1])
a=score(affine);b=score(pattern);assert np.ptp(a)<1e-12 and np.argmax(b)==correct and b[correct]>b[zero]
gradient=np.column_stack([.031*np.cos(.031*Q[:,0]+.013*Q[:,1])-.7*.009*np.sin(.009*Q[:,0]-.037*Q[:,1])+.4*.017*np.cos(.017*Q[:,0]-.021*Q[:,1]),.013*np.cos(.031*Q[:,0]+.013*Q[:,1])+.7*.037*np.sin(.009*Q[:,0]-.037*Q[:,1])-.4*.021*np.cos(.017*Q[:,0]-.021*Q[:,1])])
eigen=np.linalg.eigvalsh(gradient.T@gradient);assert eigen.min()>0
# Same original-frame spatial frequency, different physical averaging widths.
k=.18;amplitudes=[float(np.sinc(k*w/(2*np.pi))) for w in (8.,16.)];assert abs(amplitudes[0]-amplitudes[1])>.1
res=dict(passed=True,fixed_domain_points=len(Q),affine_score_range=[float(a.min()),float(a.max())],pattern_true_score=float(b[correct]),pattern_zero_score=float(b[zero]),pattern_margin=float(b[correct]-max(b[np.arange(19)!=correct])),pattern_best_shift=delta[int(np.argmax(b))].tolist(),pattern_structure_eigenvalues=eigen.tolist(),box_average_amplitudes=amplitudes,interpretation='Shift search is possible for a nondegenerate pattern. A flat/linear/repeated pattern does not identify motion even with many cells. Different physical smoothing cannot be undone by coordinate conversion.',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
with (P/'identifiability.json').open('x',encoding='utf-8') as f:json.dump(res,f,indent=2)
print(res)
