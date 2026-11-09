"""Independent replay of persisted OOF decisions, without fitting or maps."""
from study import *
from scipy.special import expit

def main():
    p=read(OUT/'protocol.json');verify(p)
    f=read(OUT/'features.json');pred=read(OUT/'predictions.json');s=read(OUT/'summary.json')
    rows=f['records'];y=np.array([r['label']=='own' for r in rows],int);groups=np.array([r['video'] for r in rows]);G=np.array([r['geometry'] for r in rows]);names=p['models']
    assert len(rows)==91 and int(y.sum())==15 and len(set(groups))==32
    assert all(e['cohort']=='strict_last' and e['reference_relative']==3 for e in p['edges'])
    excluded=set(read(DATA/'protocol.json')['excluded_development_videos'])
    assert not set(groups)&excluded
    assert [(r['video'],r['uid'],r['raw_id'],r['label']) for r in rows]==[(r['video'],r['uid'],r['raw_id'],r['label']) for r in pred['records']]
    reconstructed={name:np.full(len(rows),np.nan) for name in names};error=0.;gradient_max=0.
    for video in sorted(set(groups)):
        train=np.flatnonzero(groups!=video);test=np.flatnonzero(groups==video)
        base=next(a for a in pred['folds'] if a['held_out_video']==video and a['model']=='G')
        stats=base['geometry_normalization'];mean=np.array(stats['mean']);scale=np.array(stats['scale'])
        expected=G[train].std(0);expected[np.ptp(G[train],axis=0)==0]=1
        np.testing.assert_array_equal(mean,G[train].mean(0));np.testing.assert_array_equal(scale,expected)
        gt=(G[train]-mean)/scale;ge=(G[test]-mean)/scale
        for name in names:
            fold=next(a for a in pred['folds'] if a['held_out_video']==video and a['model']==name)
            if name=='G':tr=np.ones(len(train),bool);te=np.ones(len(test),bool);xt=gt;xe=ge
            else:
                representation,kind=name[2:].rsplit('_',1)
                valid=np.array([r['variants'][representation]['valid'] for r in rows]);tr=valid[train];te=valid[test]
                reconstructed[name][test]=reconstructed['G'][test]
                if 'fallback' in fold:assert set(y[train][tr])!={0,1};continue
                def extra(indices):
                    return np.array([[rows[j]['variants'][representation]['scalar']] if kind=='mean' else rows[j]['variants'][representation]['channels'] for j in indices])
                xt=np.c_[gt[tr],extra(train[tr])];xe=np.c_[ge[te],extra(test[te])] if te.any() else None
            assert video not in fold['training_videos']
            assert set(fold['training_videos'])==set(groups[train][tr])
            model=fold['fit'];theta=np.array(model['theta']);yy=y[train][tr];counts=np.bincount(yy,minlength=2);assert (counts>0).all();np.testing.assert_array_equal(counts,model['class_counts'])
            sw=len(yy)/(2*counts[yy]);design=np.c_[xt,np.ones(len(xt))];z=design@theta
            penalty=theta.copy();penalty[-1]=0
            gradient=design.T@(sw*(expit(z)-yy))/len(yy)+penalty
            gradient_max=max(gradient_max,float(abs(gradient).max()))
            assert model['threshold']==0 and model['regularization']==1
            if xe is not None:reconstructed[name][test[te]]=np.c_[xe,np.ones(len(xe))]@theta
    assert gradient_max<=1e-7
    for name,z in reconstructed.items():
        actual=np.array([r['scores'][name] for r in pred['records']]);assert np.isfinite(z).all()
        error=max(error,float(abs(z-actual).max()))
        np.testing.assert_array_equal(z>0,actual>0)
        metrics=measures(y,z)
        for key,value in s['metrics'][name].items():assert abs(metrics[key]-value)<=1e-12
    assert error<=1e-9 and not any(s['point_pilot_gates'].values()) and s['selected_development_arm'] is None
    changes={};base=reconstructed['G']>0
    for name,z in reconstructed.items():
        now=z>0;details=[]
        for i in np.flatnonzero(now!=base):
            e=p['edges'][i]
            details.append(dict(video=e['video'],uid=e['uid'],frame=e['frame'],old=e['old'],raw_id=e['raw_id'],label=e['label'],before=float(reconstructed['G'][i]),after=float(z[i]),change=('own_returned' if now[i] else 'own_lost') if y[i] else ('foreign_new' if now[i] else 'foreign_rejected')))
        changes[name]=dict(counts={kind:sum(d['change']==kind for d in details) for kind in ['own_returned','own_lost','foreign_new','foreign_rejected']},cases=details)
    write('decision_changes.json',dict(passed=True,baseline='G',models=changes,no_fitting_or_new_scoring=True))
    write('audit.json',dict(passed=True,edges=91,own=15,foreign=76,videos=32,models=len(names),folds=len(pred['folds']),no_video_leakage=True,last_observation_only=True,excluded_prior15_videos=True,
        stored_model_prediction_max_error=error,independent_gradient_max=gradient_max,counts_and_acceptance_recomputed=True,all_pilot_gates_failed=True,
        sources={path.relative_to(ROOT).as_posix():sha(path) for path in [OUT/'protocol.json',OUT/'features.json',OUT/'predictions.json',OUT/'summary.json',Path(__file__)]},
        no_new_training_maps_or_threshold_search=True))
    print('Independent OOF audit passed;',error,gradient_max,flush=True)
    print(json.dumps(changes['G_C34_channels']['counts']),flush=True)

if __name__=='__main__':main()
