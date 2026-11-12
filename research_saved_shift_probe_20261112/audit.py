from common import *
import math
import numpy as np
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator
p=verify();raw=read(OUT/'raw.json');assert raw['patches_sha256']==sha(OUT/'patches.npz')
ag=read(ROOT/'research_agreement_readout_20261112/protocol.json');pred=read(ROOT/'research_agreement_readout_20261112/predictions.json');points=np.array(read(ROOT/'runs/z13_local_peak_20261021/geometry.json')['points']);mesh=Delaunay(points)
counts={};maxinterp=maxfeature=maxscore=0.
with np.load(OUT/'patches.npz',allow_pickle=False) as z:
    for r in raw['records']:
        a=z[r['reference_key']+'_a'];b=z[r['patch_key']+'_b'];q=z[r['patch_key']+'_query'];m=z[r['reference_key']+'_map'];independent=LinearNDInterpolator(mesh,m)(q)
        error=float(abs(independent-b).max());maxinterp=max(maxinterp,error);assert error<1e-9
        active=np.ptp(a,axis=0)!=0;other=np.ptp(b,axis=0)!=0;ma=a.mean(0);mb=b.mean(0);va=np.var(a,axis=0);vb=np.var(b,axis=0);va[~active]=0;vb[~other]=0
        den=va+vb+(ma-mb)**2;mse=np.mean((a-b)**2,axis=0);value=np.divide(den-mse,den,out=np.zeros(34),where=den>0);value[~active]=0;value*=34/active.sum()
        error=float(abs(value-r['channels']).max());maxfeature=max(maxfeature,error);assert error<1e-9
        original=ag[r['dataset']][r['index']];fit=next(f for f in pred['fits'] if f['model']==r['representation']+'__agreement' and f['video']==r['fit_video']);g=(np.array(original['geometry'])-fit['normalization']['mean'])/fit['normalization']['scale'];x=np.r_[g,(value-fit['signal_mean'])/fit['signal_scale']];theta=fit['fit']['theta'];score=math.fsum(float(v)*float(w) for v,w in zip(x,theta[:-1]))+theta[-1]
        error=abs(score-r['score']);maxscore=max(maxscore,error);assert error<1e-9 and (score>0)==r['accepted']
        k=r['dataset']+'|'+r['representation']+'|'+r['shift_selector'];counts.setdefault(k,dict(own_accept=0,foreign_accept=0,total=0));counts[k]['total']+=1;counts[k][r['label']+'_accept']+=int(score>0)
assert len(raw['records'])==279 and sum(r['identity_control'] for r in raw['records'])==93
assert len({(r['dataset'],r['index'],r['representation'],r['shift_selector']) for r in raw['records']})==279
case=[r for r in raw['records'] if r['dataset']=='screen' and r['old_id']==232];c=[r for r in case if r['representation']=='C34'];changed=any(r['accepted'] for r in c if not r['identity_control'])
verify();save('audit.json',dict(passed=True,created=now(),records=279,identity_controls=93,max_interpolation_error=maxinterp,max_feature_error=maxfeature,max_score_error=maxscore,counts=counts,case232=case,case232_shift_only_explanation_supported=changed,independence='Alternative SciPy interpolation, MSE identity for agreement and math.fsum score. Shared maps, labels, triangles and saved heads disclosed; no independent physical correspondences.'))
print('AUDIT_PASSED','232_SHIFT_SUFFICIENT',changed,'errors',maxinterp,maxfeature,maxscore)
