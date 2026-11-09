"""Independent train-only statistics, objective, gradient and held-out predictions."""
import json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf8'))
for n,h in read(OUT/'protocol.json')['sources'].items():assert hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h,n
rows=read(OUT/'features.json')['records'];pred=read(OUT/'predictions.json');old=read(ROOT/'research_domain_matched_reader_20261109/features.json')['records'];groups=np.array([r['video'] for r in rows]);y=np.array([r['label']=='own' for r in rows],int);G=np.array([r['geometry'] for r in rows]);maxgrad=maxloss=maxprediction=0.
for a,b in zip(rows,old):assert all(a[k]==b[k] for k in ['video','frame','old_id','raw_id','geometry','label','full_cells'])
for f in pred['folds']:
    tr=np.flatnonzero(groups!=f['video']);te=np.flatnonzero(groups==f['video']);arm,kind=f['model'].split('__');raw=np.array([r['variants'][arm][kind] for r in rows]);mu=raw[tr].mean(0);sd=raw[tr].std(0);constant=np.ptp(raw[tr],axis=0)==0;scale=np.where(constant,1.,np.sqrt(np.mean(sd[~constant]**2)))
    np.testing.assert_array_equal(mu,f['signal_mean']);np.testing.assert_array_equal(scale,f['signal_scale']);assert sorted(set(groups[tr]))==f['training_videos']
    g=(G-f['normalization']['mean'])/f['normalization']['scale'];X=np.c_[g,(raw-mu)/scale];t=np.array(f['fit']['theta']);z=X[tr]@t[:-1]+t[-1];w=len(tr)/(2*np.bincount(y[tr])[y[tr]]);res=w*(np.exp(-np.logaddexp(0,-z))-y[tr]);grad=np.r_[X[tr].T@res/len(tr)+t[:-1],res.mean()];loss=float(np.mean(w*np.logaddexp(0,np.where(y[tr]==1,-z,z)))+.5*(t[:-1]@t[:-1]));maxgrad=max(maxgrad,float(abs(grad).max()));maxloss=max(maxloss,abs(loss-f['fit']['loss']));ztest=X[te]@t[:-1]+t[-1];maxprediction=max(maxprediction,float(abs(ztest-np.array([pred['records'][i]['scores'][f['model']] for i in te])).max()))
assert len(pred['folds'])==120 and maxgrad<1e-7 and max(maxloss,maxprediction)<1e-12
with (OUT/'audit.json').open('x',encoding='utf8') as f:json.dump(dict(passed=True,audited_fits=120,gradient_max=maxgrad,loss_error=maxloss,prediction_error=maxprediction,train_only_scaling_verified=True,old_labels_geometry_cells_winners_preserved=True,no_refits=True,actual_IDs_evaluated=False),f,indent=2)
print('AGREEMENT_AUDIT_PASSED',maxgrad,flush=True)
