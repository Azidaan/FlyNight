"""Экспорт PCA и регрессия вычисления базы без меток для включения Б1 в трекер."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'): os.environ[key] = '2'
from pathlib import Path
import sys,json,hashlib,time
from datetime import datetime, timezone, timedelta
import numpy as np
import joblib

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
BASE = ROOT / 'runs/b1_real_inputs_20261026'
OUT = ROOT / 'runs/b123_20261027/b1_base_export'


def status(stage, **kw):
    (OUT/'status.json').write_bytes((json.dumps(dict(stage=stage, updated=datetime.now(timezone(timedelta(hours=3))).isoformat(), **kw),ensure_ascii=False,indent=2)+'\n').encode())


def main():
    from tracking.b1_appearance import appearance_features, observe_tag, gate_features
    assert not OUT.exists(); OUT.mkdir()
    with np.load(BASE / 'base_inputs.npz') as z: data = {k:z[k] for k in z.files}
    index = {int(u): i for i,u in enumerate(data['uid'])}; count,width = data['mask'].shape
    region = np.zeros((count,width,256),np.float32); coarse = np.zeros((count,256),np.float32); snapped = coarse.copy()
    for video in np.unique(data['name']):
        status('reading observed embeddings',video=video)
        with np.load(ROOT / f'runs/z13_masa_regions_20261026/recovery1/videos/{video}.npz') as z:
            for j,u in enumerate(z['uid']):
                i=index[int(u)]; k=min(z['region'].shape[1],width-1)
                region[i,1:1+k]=z['region'][j,:k]; coarse[i]=z['exemplar'][j]
        with np.load(ROOT / f'runs/z13_masa_exemplar_snap_20261026/videos/{video}.npz') as z:
            for j,u in enumerate(z['uid']): snapped[index[int(u)]]=z['exemplar'][j]
    real=data['mask'].copy();real[:,0]=False
    sample=region[real];mean=sample.mean(0)
    _,_,vt=np.linalg.svd(sample[::max(1,len(sample)//20000)]-mean,full_matrices=False);projection=vt[:32].T
    previous=json.loads((ROOT/'runs/z13_box_supervision_followup_20261025/temporal_readiness.json').read_text(encoding='utf-8'))['previous_query']
    states={};built=np.empty_like(data['F1'])
    for i in np.argsort(data['uid']):
        prior=previous.get(str(int(data['uid'][i])));j=index.get(int(prior['previous_uid'])) if prior else None
        built[i],bank=appearance_features(data['geo'][i],data['mask'][i],region[i],coarse[i],snapped[i],mean,projection,
                                        states.get(j),prior['gap_frames'] if j is not None else 0)
        states[i]=observe_tag(data['geo'][i],data['mask'][i],region[i],bank)
        if len(states)%1000==0: status('rebuilding observable features',done=len(states),total=count)
    finite=np.isfinite(data['F1'])
    assert np.array_equal(np.isnan(built),np.isnan(data['F1']))
    difference=float(np.abs(built[finite]-data['F1'][finite]).max())
    # Reconstructed SVD may differ at roundoff; tree predictions must be exact.
    assert difference<1e-5, difference
    p=json.loads((BASE/'protocol.json').read_text(encoding='utf-8'));inner={v:i%4 for i,v in enumerate(sorted(p['fit84']))}
    probability=np.full((count,width),np.nan)
    for fold in range(5):
        videos=p['fit20'] if fold==4 else [v for v in p['fit84'] if inner[v]==fold]
        inside=np.isin(data['name'],videos)[:,None]&real
        probability[inside]=joblib.load(BASE/f'identity_fold{fold}.joblib').predict_proba(built[inside])[:,1]
    with np.load(BASE/'base_predictions.npz') as z:
        assert np.array_equal(probability,z['identity_probability'],equal_nan=True)
        original_gate=z['calibration_gate']
    model=joblib.load(BASE/'gate.joblib')
    from scripts.z13_identity_gate_readout_20261026 import gate_features as original_features
    expected=original_features(data,probability)
    rebuilt=np.stack([gate_features(data['geo'][i],data['mask'][i],probability[i]) for i in range(count)])
    assert np.array_equal(rebuilt,expected,equal_nan=True)
    inside=np.isin(data['name'],p['fit20'])[:,None]&data['mask']&(data['geo'][...,-1]<.5)
    valid=np.isfinite(original_gate)&inside
    assert np.array_equal(model.predict_proba(rebuilt[valid])[:,1],original_gate[valid])
    np.savez_compressed(OUT/'pca.npz',mean=mean,projection=projection)
    report=dict(passed=True,queries=count,feature_max_difference=difference,identity_predictions_exact=True,gate_features_exact=True,
                gate_predictions_exact=True,width=width,ground_truth_used=False,hota_measured=False,
                pca_sha256=hashlib.sha256((OUT/'pca.npz').read_bytes()).hexdigest(),
                limitation='Regression on reused saved inputs. Live MASA extraction and full tracker zero-residual regression remain to be checked.')
    (OUT/'report.json').write_bytes((json.dumps(report,ensure_ascii=False,indent=2)+'\n').encode())
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':main()
