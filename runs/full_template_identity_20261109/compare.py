"""Frozen signal comparison after the preregistered support geometry gate."""
from geometry import *
from tracking.channel_identity_head import spatial_channel_features,geometry_transform,fit_ridge_logistic,decision_function

NAMES=['C34','R134_spatial','R134_temporal']

def register():
    p=read(OUT/'protocol.json');verify(p);g=read(OUT/'geometry_results.json');assert g['passed'] and g['gate']
    old=read(BASE/'protocol.json');verify(old)
    files=[Path(__file__),OUT/'protocol.json',OUT/'geometry_results.json',OUT/'МАТЕМАТИКА.md',
        ROOT/'scripts/z13_target_model.py',ROOT/'tracking/z13_readout.py',ROOT/'tracking/channel_identity_head.py']
    sources={name:value for name,value in read(BASE/'maps_complete.json')['sources'].items()}
    for path in files:sources[path.relative_to(ROOT).as_posix()]=sha(path)
    verify(dict(sources=sources))
    write('comparison_protocol.json',dict(registered=datetime.now(MSK).isoformat(),sources=sources,
        conditions_from='protocol.json',geometry_gate_only_seen=True,real_signal_outcomes_not_seen=True,
        unchanged='Use all91 old rows; no sample or threshold change after the geometry gate. Every model, admission condition and comparator was fixed in protocol.json before the gate.',
        old_predictions_reused='G_ready and all old correlations come from the completed prior experiment. Old-head models at augmented G_support are new matched fits, not repeats of the old seven-geometry-feature fit.'))
    journal('Проверка геометрии целого шаблона пройдена, сигналы ещё не считались',f'*Получено (`{OUT.name}`).* Все6 математических тестов пройдены. Прежние ячейки и координаты воспроизведены точно, максимальная ошибка0. В90из91 пар полная опора больше: медиана23→49 ячеек при медиане77 ячеек прошлой рамки. Медианная сохранённая доля прошлого шаблона0,302632→0,704762; это медианы отношений, а не отношение медиан. Источник `runs/{OUT.name}/geometry_results.json`.\n\n*Вывод.* Усечение поддержкой всех сдвигов действительно удаляет пространственные измерения, но их полезность не измерена.\n\n*Решение.* По исходному критерию разрешён следующий этап без сети: готовые114 снимков, старые признаки и оценки переиспользуются. Условия и новые исходники перед вычислением сигналов зафиксированы в `runs/{OUT.name}/comparison_protocol.json`. Сравниваются одинаковые старые/полные модели при той же расширенной геометрии; порог0 и штраф1. Предыдущее отрицательное обучение остаётся отрицательным.')
    print('Signal implementation registered before new correlations or fits',flush=True)

def arrays(key,old):
    values={}
    for name,item in old['requests'][key]['ready'].items():
        with np.load(ROOT/item['path'],allow_pickle=False) as z:values[name]=z[item['array']].copy()
    path=BASE/'maps'/(key+'.npz')
    if path.exists():
        assert sha(path)==read(path.with_suffix('.json'))['sha256']
        with np.load(path,allow_pickle=False) as z:
            for name in z.files:assert name not in values;values[name]=z[name].copy()
    if 'C34' not in values:
        c=np.empty((721,34),np.float32);c[:,4:12]=values['C8'];c[:,[0,1,2,3]+list(range(12,34))]=values['C_OTHER'];values['C34']=c
    return values

def extract():
    p=read(OUT/'protocol.json');verify(p);verify(read(OUT/'comparison_protocol.json'))
    if (OUT/'features.json').exists():raise SystemExit('Ready full-template features exist')
    old=read(BASE/'protocol.json');ready=read(BASE/'features.json')['records'];g=read(OUT/'geometry_results.json');geom=read(ROOT/'runs/z13_local_peak_20261021/geometry.json')
    reader=ObservedSpatialReader(geom['points'],geom['eye_size'],[1280,720])
    from tracking.z13_readout import HexGeometry
    from scripts.z13_target_model import neighbor_aug
    nb=HexGeometry(np.array(geom['points']),geom['eye_size']).neighbors;cache={};rows=[];error=0.
    def features(key):
        if key not in cache:
            a=arrays(key,old);r=a['R1'].astype(float)
            cache[key]=dict(C34=a['C34'].astype(float),R134_temporal=r,R134_spatial=neighbor_aug(r[:,:17].T[None],nb)[0].T)
        return cache[key]
    for i,(e,record,entry) in enumerate(zip(p['edges'],ready,g['records'])):
        assert (e['video'],e['uid'],e['raw_id'])==(record['video'],record['uid'],record['raw_id'])==(entry['video'],entry['uid'],entry['raw_id'])
        ref,cur=[features(k) for k in e['keys']];variants={};plan=entry['plan']
        for name in NAMES:
            value=spatial_channel_features(reader.mesh,ref[name],cur[name],plan['cells'],plan['query'])
            witness=reader.compare(ref[name],cur[name],plan)
            assert value['winner']==witness['winner'] and value['scores']==witness['scores']
            if value['valid']:error=max(error,abs(np.mean(value['channels'])-witness['score']))
            variants[name]=dict(old=record['variants'][name],full=value)
        geometry=np.r_[record['geometry'],np.log1p(entry['full_count']),entry['full_count']/entry['reference_count']]
        rows.append(dict(video=e['video'],uid=e['uid'],old=e['old'],raw_id=e['raw_id'],label=e['label'],geometry=geometry,variants=variants))
        print('FULL FEATURES',i+1,len(ready),flush=True)
    assert error<=1e-12;verify(p);verify(read(OUT/'comparison_protocol.json'))
    write('features.json',dict(passed=True,records=rows,old_features_reused=True,full_mean_scalar_parity_max_error=error,zero_neural_calls=True,fixed_cells_per_shift=True))

def measures(y,z):
    accept=np.asarray(z)>0;own=np.asarray(y)==1
    tp=int((accept&own).sum());fn=int((~accept&own).sum());fp=int((accept&~own).sum());tn=int((~accept&~own).sum())
    return dict(own_accept=tp,own_reject=fn,foreign_accept=fp,foreign_reject=tn,balanced_accuracy=.5*(tp/(tp+fn)+tn/(fp+tn)))

def fit():
    p=read(OUT/'protocol.json');verify(p);verify(read(OUT/'comparison_protocol.json'))
    if (OUT/'summary.json').exists() or (OUT/'predictions.json').exists():raise SystemExit('Ready comparison exists')
    rows=read(OUT/'features.json')['records'];previous=read(BASE/'predictions.json')['records']
    assert [(r['video'],r['uid'],r['raw_id'],r['label']) for r in rows]==[(r['video'],r['uid'],r['raw_id'],r['label']) for r in previous]
    groups=np.array([r['video'] for r in rows]);videos=sorted(set(groups));y=np.array([r['label']=='own' for r in rows],int);G=np.array([r['geometry'] for r in rows])
    names=['G_ready','G_support']+[f'{name}_{mode}_{kind}' for name in NAMES for mode in ['old','full'] for kind in ['mean','channels']]
    scores={name:np.full(len(rows),np.nan) for name in names};scores['G_ready']=np.array([r['scores']['G'] for r in previous]);fits=[]
    for fold,video in enumerate(videos):
        train=np.flatnonzero(groups!=video);test=np.flatnonzero(groups==video);assert set(y[train])=={0,1}
        gt,ge,stats=geometry_transform(G[train],G[test]);base=fit_ridge_logistic(gt,y[train]);gb=decision_function(base,ge);scores['G_support'][test]=gb
        fits.append(dict(held_out_video=video,model='G_support',training_videos=sorted(set(groups[train])),fit=base,geometry_normalization=stats))
        for name in NAMES:
            for mode in ['old','full']:
                valid=np.array([r['variants'][name][mode]['valid'] for r in rows]);tr=valid[train];te=valid[test]
                for kind in ['mean','channels']:
                    model_name=f'{name}_{mode}_{kind}';scores[model_name][test]=gb
                    if not tr.any() or set(y[train][tr])!={0,1}:
                        fits.append(dict(held_out_video=video,model=model_name,fallback='training_feature_rows_lack_both_classes'));continue
                    def extra(indices):return np.array([[rows[j]['variants'][name][mode]['scalar']] if kind=='mean' else rows[j]['variants'][name][mode]['channels'] for j in indices])
                    xt=np.c_[gt[tr],extra(train[tr])];model=fit_ridge_logistic(xt,y[train][tr])
                    if te.any():scores[model_name][test[te]]=decision_function(model,np.c_[ge[te],extra(test[te])])
                    fits.append(dict(held_out_video=video,model=model_name,training_videos=sorted(set(groups[train][tr])),fit=model,test_fallbacks=int((~te).sum())))
        print('FOLD',fold+1,len(videos),flush=True)
    assert all(np.isfinite(v).all() for v in scores.values());metrics={k:measures(y,v) for k,v in scores.items()};gates={}
    for name in NAMES:
        for kind in ['mean','channels']:
            key=f'{name}_full_{kind}';a=metrics[key];bases=[metrics[b] for b in ['G_ready','G_support',f'{name}_old_{kind}']]
            gates[key]=(a['own_accept']>=max(b['own_accept'] for b in bases) and a['foreign_accept']<min(b['foreign_accept'] for b in bases) and a['balanced_accuracy']>max(b['balanced_accuracy'] for b in bases))
    choices=[k for k,v in gates.items() if v];selected=sorted(choices,key=lambda k:(-metrics[k]['balanced_accuracy'],metrics[k]['foreign_accept'],k.endswith('channels'),NAMES.index(k.rsplit('_',2)[0])))[0] if choices else None
    rng=np.random.default_rng(20261109);byvideo=[np.flatnonzero(groups==v) for v in videos];draws=[]
    for _ in range(4000):
        idx=np.concatenate([byvideo[j] for j in rng.integers(len(videos),size=len(videos))])
        if len(set(y[idx]))==2:draws.append(idx)
    comparisons=[]
    for name in NAMES:
        for kind in ['mean','channels']:
            comparisons += [(f'{name}_full_{kind}',b) for b in ['G_ready','G_support',f'{name}_old_{kind}']]
    primary_bases=['C34_old_channels','G_support','R134_spatial_full_channels','R134_temporal_full_channels']
    comparisons += [('C34_full_channels',b) for b in primary_bases]
    intervals={}
    for current,baseline in comparisons:
        key=current+'__'+baseline
        if key in intervals:continue
        delta=[measures(y[i],scores[current][i])['balanced_accuracy']-measures(y[i],scores[baseline][i])['balanced_accuracy'] for i in draws]
        value=dict(point=metrics[current]['balanced_accuracy']-metrics[baseline]['balanced_accuracy'],ci95=np.percentile(delta,[2.5,97.5]),valid_draws=len(delta))
        if current=='C34_full_channels' and baseline in primary_bases:value['ci98_75_primary_family']=np.percentile(delta,[.625,99.375])
        intervals[key]=value
    verify(p);verify(read(OUT/'comparison_protocol.json'))
    write('predictions.json',dict(passed=True,records=[dict(video=r['video'],uid=r['uid'],old=r['old'],raw_id=r['raw_id'],label=r['label'],scores={name:z[i] for name,z in scores.items()}) for i,r in enumerate(rows)],folds=fits,ready_G_reused=True,zero_threshold_tuning=True,zero_ID_assignment=True))
    write('summary.json',dict(passed=True,edges=len(rows),own=int(y.sum()),foreign=int((1-y).sum()),videos=len(videos),metrics=metrics,pilot_admission=gates,selected_development_arm=selected,paired_video_intervals=intervals,geometry_only_coverage_control=True,old_features_and_G_reused=True,new_neural_calls=0,new_ID_assignments=0))
    print(json.dumps(cast(dict(metrics=metrics,gates=gates,selected=selected)),ensure_ascii=False),flush=True)

if __name__=='__main__':{'register':register,'extract':extract,'fit':fit}[sys.argv[1]]()
