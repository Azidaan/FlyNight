"""Verify the third diagnostic from saved coefficients, without new fits."""
from original_geometry import *
from scipy.special import expit

p=read(OUT/'original_geometry_protocol.json');verify(p)
rows=read(OUT/'features.json')['records'];pred=read(OUT/'original_geometry_predictions.json');s=read(OUT/'original_geometry_results.json');previous=read(BASE/'predictions.json')
y=np.array([r['label']=='own' for r in rows],int);groups=np.array([r['video'] for r in rows]);G=np.array([r['geometry'][:7] for r in rows]);error=0.;gradient_error=0.
assert len(rows)==91 and y.sum()==15 and len(set(groups))==32
assert [(r['video'],r['uid'],r['raw_id']) for r in rows]==[(r['video'],r['uid'],r['raw_id']) for r in pred['records']]
new_names=[k for k in s['metrics'] if '_full7_' in k];values={k:np.array([r['scores'][k] for r in pred['records']]) for k in s['metrics']}
for key in new_names:
    reconstructed=values['G_ready'].copy();name,mode,kind=key.rsplit('_',2)
    for video in sorted(set(groups)):
        train=np.flatnonzero(groups!=video);test=np.flatnonzero(groups==video);fold=next(a for a in pred['folds'] if a['held_out_video']==video and a['model']==key)
        valid=np.array([r['variants'][name]['full']['valid'] for r in rows]);tr=valid[train];te=valid[test]
        if 'fallback' in fold:assert set(y[train][tr])!={0,1};continue
        old=next(a for a in previous['folds'] if a['held_out_video']==video and a['model']=='G')
        assert fold['geometry_normalization']==old['geometry_normalization']
        assert video not in fold['training_videos'] and set(fold['training_videos'])==set(groups[train][tr])
        mean=np.array(fold['geometry_normalization']['mean']);scale=np.array(fold['geometry_normalization']['scale']);gt=(G[train]-mean)/scale;ge=(G[test]-mean)/scale
        def extra(indices):return np.array([[rows[j]['variants'][name]['full']['scalar']] if kind=='mean' else rows[j]['variants'][name]['full']['channels'] for j in indices])
        xt=np.c_[gt[tr],extra(train[tr])];theta=np.array(fold['fit']['theta']);yy=y[train][tr];counts=np.bincount(yy,minlength=2);sw=len(yy)/(2*counts[yy]);design=np.c_[xt,np.ones(len(xt))];penalty=theta.copy();penalty[-1]=0
        gradient=design.T@(sw*(expit(design@theta)-yy))/len(yy)+penalty;gradient_error=max(gradient_error,float(abs(gradient).max()))
        assert fold['fit']['threshold']==0 and fold['fit']['regularization']==1
        if te.any():reconstructed[test[te]]=np.c_[ge[te],extra(test[te]),np.ones(int(te.sum()))]@theta
    error=max(error,float(abs(reconstructed-values[key]).max()));np.testing.assert_array_equal(reconstructed>0,values[key]>0)
    for field,value in measures(y,reconstructed).items():assert abs(value-s['metrics'][key][field])<=1e-12
assert error<=1e-9 and gradient_error<=1e-7
for name in NAMES:
    for kind in ['mean','channels']:
        key=f'{name}_full7_{kind}';a=s['metrics'][key];bases=[s['metrics'][b] for b in ['G_ready','G_support',f'{name}_old7_{kind}']]
        assert s['pilot_admission'][key]==(a['own_accept']>=max(b['own_accept'] for b in bases) and a['foreign_accept']<min(b['foreign_accept'] for b in bases) and a['balanced_accuracy']>max(b['balanced_accuracy'] for b in bases))
assert [k for k,v in s['pilot_admission'].items() if v]==['C34_full7_channels']
assert s['selected_development_arm']=='C34_full7_channels'
changes={};target=values['C34_full7_channels']>0
for baseline in ['G_ready','G_support','C34_old7_channels','R134_spatial_full7_channels','R134_temporal_full7_channels']:
    before=values[baseline]>0;cases=[]
    for i in np.flatnonzero(before!=target):
        e=read(OUT/'protocol.json')['edges'][i]
        cases.append(dict(video=e['video'],uid=e['uid'],frame=e['frame'],old=e['old'],raw_id=e['raw_id'],label=e['label'],before=float(values[baseline][i]),after=float(values['C34_full7_channels'][i]),change=('own_returned' if target[i] else 'own_lost') if y[i] else ('foreign_new' if target[i] else 'foreign_rejected')))
    changes[baseline]=dict(counts={k:sum(v['change']==k for v in cases) for k in ['own_returned','own_lost','foreign_new','foreign_rejected']},cases=cases)
write('original_geometry_changes.json',dict(passed=True,target='C34_full7_channels',comparisons=changes,no_new_fitting_or_signal_scoring=True))
write('original_geometry_audit.json',dict(passed=True,edges=91,own=15,foreign=76,videos=32,new_models=6,folds=len(pred['folds']),
    stored_prediction_max_error=error,independent_gradient_max=gradient_error,original_normalization_reused=True,old_models_and_scores_reused=True,no_video_leakage=True,
    only_admitted_arm='C34_full7_channels',all_admission_conditions_recomputed=True,no_new_neural_calls=True,no_ID_assignments=True,
    sources={path.relative_to(ROOT).as_posix():sha(path) for path in [Path(__file__),OUT/'original_geometry_protocol.json',OUT/'original_geometry_predictions.json',OUT/'original_geometry_results.json']},
    limit='Third post-result development diagnostic. Full tracker parity, ID fate, independent HOTA and unique FlyVis benefit remain unmeasured.'))
print('Original-geometry stored-model audit passed',error,gradient_error,flush=True)
for key,value in changes.items():print(key,value['counts'],flush=True)
