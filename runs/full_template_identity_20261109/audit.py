"""Independent persisted-model verification, no maps or fitting repeated."""
from compare import *
from scipy.special import expit

def main():
    p=read(OUT/'protocol.json');verify(p);verify(read(OUT/'comparison_protocol.json'))
    rows=read(OUT/'features.json')['records'];pred=read(OUT/'predictions.json');s=read(OUT/'summary.json');ready=read(BASE/'predictions.json')['records']
    y=np.array([r['label']=='own' for r in rows],int);groups=np.array([r['video'] for r in rows]);G=np.array([r['geometry'] for r in rows])
    assert len(rows)==91 and y.sum()==15 and len(set(groups))==32
    assert [(r['video'],r['uid'],r['raw_id']) for r in rows]==[(r['video'],r['uid'],r['raw_id']) for r in pred['records']]
    assert all(e['cohort']=='strict_last' for e in p['edges'])
    names=list(s['metrics']);reconstructed={n:np.full(len(rows),np.nan) for n in names};reconstructed['G_ready']=np.array([r['scores']['G'] for r in ready]);grad_error=0.;prediction_error=0.
    for video in sorted(set(groups)):
        train=np.flatnonzero(groups!=video);test=np.flatnonzero(groups==video)
        geo=next(a for a in pred['folds'] if a['held_out_video']==video and a['model']=='G_support')
        mean=np.array(geo['geometry_normalization']['mean']);scale=np.array(geo['geometry_normalization']['scale']);expected=G[train].std(0);expected[np.ptp(G[train],axis=0)==0]=1
        np.testing.assert_array_equal(mean,G[train].mean(0));np.testing.assert_array_equal(scale,expected)
        gt=(G[train]-mean)/scale;ge=(G[test]-mean)/scale
        for name in names[1:]:
            fold=next(a for a in pred['folds'] if a['held_out_video']==video and a['model']==name)
            if name=='G_support':tr=np.ones(len(train),bool);te=np.ones(len(test),bool);xt=gt;xe=ge
            else:
                representation,mode,kind=name.rsplit('_',2);valid=np.array([r['variants'][representation][mode]['valid'] for r in rows]);tr=valid[train];te=valid[test]
                reconstructed[name][test]=reconstructed['G_support'][test]
                if 'fallback' in fold:assert set(y[train][tr])!={0,1};continue
                def extra(indices):return np.array([[rows[j]['variants'][representation][mode]['scalar']] if kind=='mean' else rows[j]['variants'][representation][mode]['channels'] for j in indices])
                xt=np.c_[gt[tr],extra(train[tr])];xe=np.c_[ge[te],extra(test[te])] if te.any() else None
            assert video not in fold['training_videos'] and set(fold['training_videos'])==set(groups[train][tr])
            theta=np.array(fold['fit']['theta']);yy=y[train][tr];counts=np.bincount(yy,minlength=2);assert (counts>0).all();np.testing.assert_array_equal(counts,fold['fit']['class_counts'])
            sw=len(yy)/(2*counts[yy]);design=np.c_[xt,np.ones(len(xt))];penalty=theta.copy();penalty[-1]=0
            gradient=design.T@(sw*(expit(design@theta)-yy))/len(yy)+penalty;grad_error=max(grad_error,float(abs(gradient).max()))
            assert fold['fit']['threshold']==0 and fold['fit']['regularization']==1
            if xe is not None:reconstructed[name][test[te]]=np.c_[xe,np.ones(len(xe))]@theta
    assert grad_error<=1e-7
    for name,z in reconstructed.items():
        actual=np.array([r['scores'][name] for r in pred['records']]);assert np.isfinite(z).all();prediction_error=max(prediction_error,float(abs(z-actual).max()))
        np.testing.assert_array_equal(z>0,actual>0)
        for key,value in measures(y,z).items():assert abs(value-s['metrics'][name][key])<=1e-12
    assert prediction_error<=1e-9
    gates={}
    for representation in NAMES:
        for kind in ['mean','channels']:
            name=f'{representation}_full_{kind}';a=s['metrics'][name];bases=[s['metrics'][b] for b in ['G_ready','G_support',f'{representation}_old_{kind}']]
            gates[name]=(a['own_accept']>=max(b['own_accept'] for b in bases) and a['foreign_accept']<min(b['foreign_accept'] for b in bases) and a['balanced_accuracy']>max(b['balanced_accuracy'] for b in bases))
    assert gates==s['pilot_admission']
    assert [name for name,passed in gates.items() if passed]==['R134_temporal_full_channels']
    assert s['selected_development_arm']=='R134_temporal_full_channels'
    changes={}
    for name in ['C34_full_channels','R134_spatial_full_channels','R134_temporal_full_channels']:
        baselines=['G_ready',name.replace('_full_','_old_')]
        for baseline in baselines:
            before=reconstructed[baseline]>0;after=reconstructed[name]>0;cases=[]
            for i in np.flatnonzero(before!=after):
                e=p['edges'][i]
                cases.append(dict(video=e['video'],uid=e['uid'],frame=e['frame'],old=e['old'],raw_id=e['raw_id'],label=e['label'],before=float(reconstructed[baseline][i]),after=float(reconstructed[name][i]),change=('own_returned' if after[i] else 'own_lost') if y[i] else ('foreign_new' if after[i] else 'foreign_rejected')))
            changes[name+'__'+baseline]=dict(counts={k:sum(v['change']==k for v in cases) for k in ['own_returned','own_lost','foreign_new','foreign_rejected']},cases=cases)
    write('decision_changes.json',dict(passed=True,comparisons=changes,no_new_fitting_or_signal_scoring=True))
    write('audit.json',dict(passed=True,edges=91,own=15,foreign=76,videos=32,models=len(names),fitted_models=len(pred['folds']),new_neural_calls=0,new_ID_assignments=0,
        no_video_leakage=True,old_G_scores_reused=True,old_correlations_reused=True,full_support_fixed_over_shifts=True,
        independent_stored_prediction_max_error=prediction_error,independent_gradient_max=grad_error,pilot_conditions_independently_recomputed=True,admitted_arm='R134_temporal_full_channels',FlyVis_not_admitted=True,
        sources={str(path.relative_to(ROOT)):sha(path) for path in [Path(__file__),OUT/'protocol.json',OUT/'comparison_protocol.json',OUT/'features.json',OUT/'predictions.json',OUT/'summary.json']},
        limit='Mathematics, OOF decisions and admission are verified. Actual ID continuity, native matching parity and independent HOTA remain unmeasured.'))
    print('Full-template audit passed',prediction_error,grad_error,flush=True)
    for key,value in changes.items():print(key,value['counts'],flush=True)

if __name__=='__main__':main()
