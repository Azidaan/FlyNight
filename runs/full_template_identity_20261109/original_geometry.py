"""Third/final signal diagnostic: change support without changing geometry."""
from compare import *
from record import journal_atomic,atomic_json

ID='full_template_original_geometry_20261109'

def register():
    p=read(OUT/'protocol.json');verify(p);verify(read(OUT/'comparison_protocol.json'));assert read(OUT/'audit.json')['passed']
    assert read(OUT/'final_checks.json')['passed']
    s=read(OUT/'summary.json');assert s['metrics']['G_ready']['own_accept']==13 and s['metrics']['G_support']['own_accept']==10
    files=[Path(__file__),OUT/'protocol.json',OUT/'comparison_protocol.json',OUT/'features.json',OUT/'predictions.json',OUT/'summary.json',OUT/'audit.json',OUT/'record.py',
        BASE/'protocol.json',BASE/'predictions.json',BASE/'summary.json',ROOT/'tracking/channel_identity_head.py']
    protocol=dict(registered=datetime.now(MSK).isoformat(),head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        action='Same full-template support change, direct original-seven-geometry ablation. No new maps/correlations/thresholds. Six new fits compare directly to ready old-support/seven-geometry models.',
        diagnostic_number=3,maximum_signal_diagnostics=3,reason='The added two coverage geometry features alone degraded own accepts13→10. The direct old7+old support vs old7+full support contrast was missing. This is a post-result explanatory ablation, not independent confirmation.',
        data='Same91 strict-last pairs,15 own/76 foreign,32 videos. Reuse full and old feature vectors and all baseline decisions. Exactly first7 geometry features and the STORED old training-fold geometry normalization.',
        model='Same fixedlambda1,free intercept,class-balanced logistic loss,zero init,damped Newton,zero threshold. Full-mean and full-channels for C34,R134spatial,R134temporal;6 new models, no hyperparameter or sample search. Undefined feature falls back to ready G7; no G model refit.',
        validation='Same32 leave-one-video-out folds, no video/ID/label/owner as predictive features. OOF development selection, not a new official split.',
        admission='For each new full7 arm, own accepts >= G_ready,G_support and its same-representation/same-kind ready OLD7 model; foreign accepts STRICTLY LOWER than all three, balanced accuracy higher than all three. These are the same preservation/rejection requirements as before, with the correct original7 comparator.',
        selection='Primary C34channels enters a candidate pilot if it passes. Otherwise select a passing arm by balanced accuracy, then fewer foreign accepts, then mean before34weights, then C34,R134spatial,R134temporal order. If none pass, the already admitted temporal-full9 arm remains a separate possible candidate; do not reinterpret any failed gate.',
        uncertainty='Same4000 paired video draws,seed20261109. Descriptive95%;99% intervals for five fixed C34channel contrasts vs G_ready,G_support,readyold7C34channels,full7R134spatialchannels,full7R134temporalchannels. No tracker or core success from this development ablation.',
        next='After this final signal diagnostic, implement actual causal assignment and complete zero-intervention parity, then evaluate actual IDs/HOTA or stop the version. No more channel/geometry/threshold variants on these pairs.',
        save_stop=STOP.isoformat(),authorization='research_identity_synthesis_20261109/authorization.json',
        sources={f.relative_to(ROOT).as_posix():sha(f) for f in files})
    write('original_geometry_protocol.json',protocol)
    path=ROOT/'experiments/project_history.json';cat=read(path);assert not any(c['id']==ID for c in cat['experiments'])
    cat['experiments'].append(dict(id=ID,title='Целый шаблон при прежней геометрии — прямой контроль',area='flyvis',state='running',started='2026-11-09',
        question='Не вызвана ли часть потерь своими двумя добавленными геометрическими признаками, и что даёт только полная опора при исходной геометрии?',
        hypothesis='Добавление геометрических признаков изменило функцию решения и могло скрыть пользу новой поддержки. Прямое сравнение с прежними7 признаками устраняет это различие; альтернативы — обычный вход и случайная переносимость малой группы.',
        expected='Третья и последняя диагностика сигнала перед реальными назначениями: тот же считыватель и ровно одна замена пространственной опоры.',
        acceptance=[dict(criterion='Прежние оценки и нормировка переиспользованы,6 новых моделей меняют только поддержку; нет утечки видео и порогов.',met=None),dict(criterion=protocol['admission'],met=None)],
        obtained='',interpretation='',decision='',cannot_claim='',evidence=[dict(kind='file',ref='runs/full_template_identity_20261109/original_geometry_protocol.json',claim='Условия до новых шести обучений; объяснительный характер после результата указан.')],tags=['fly-problem-P1']))
    atomic_json(path,cat)
    journal_atomic('Зафиксирован прямой контроль целого шаблона с прежней геометрией',f'*Постановка (`{ID}`).* Расширенная геометрия сама изменила принятие своих13→10; поэтому дополнительно проверяется прямое сравнение старой и полной опор при прежних7 признаках. Ранее готовые модели, оценки и нормировка переиспользуются; только6 новых полных вариантов на34-канальных представлениях, те же32 исключения видео, штраф1 и порог0. Это объяснительная проверка после результата, не независимое подтверждение. Третья и последняя сигнальная диагностика одной правки; далее реальные назначения и HOTA либо остановка. Источник `runs/full_template_identity_20261109/original_geometry_protocol.json`. Прежние состояния/критерии не переписаны.')
    print('Direct original-geometry ablation registered before new fits',flush=True)

def fit():
    p=read(OUT/'original_geometry_protocol.json');verify(p)
    if (OUT/'original_geometry_results.json').exists():raise SystemExit('Ready direct comparison exists')
    rows=read(OUT/'features.json')['records'];previous=read(BASE/'predictions.json');support=read(OUT/'predictions.json')['records']
    assert [(r['video'],r['uid'],r['raw_id'],r['label']) for r in rows]==[(r['video'],r['uid'],r['raw_id'],r['label']) for r in previous['records']]
    y=np.array([r['label']=='own' for r in rows],int);groups=np.array([r['video'] for r in rows]);G=np.array([r['geometry'][:7] for r in rows]);scores={'G_ready':np.array([r['scores']['G'] for r in previous['records']]),'G_support':np.array([r['scores']['G_support'] for r in support])};fits=[]
    for name in NAMES:
        for kind in ['mean','channels']:
            scores[f'{name}_old7_{kind}']=np.array([r['scores'][f'G_{name}_{kind}'] for r in previous['records']])
            scores[f'{name}_full7_{kind}']=np.full(len(rows),np.nan)
    for fold,video in enumerate(sorted(set(groups))):
        train=np.flatnonzero(groups!=video);test=np.flatnonzero(groups==video)
        base=next(f for f in previous['folds'] if f['held_out_video']==video and f['model']=='G');stats=base['geometry_normalization'];mean=np.array(stats['mean']);scale=np.array(stats['scale'])
        np.testing.assert_array_equal(mean,G[train].mean(0));std=G[train].std(0);std[np.ptp(G[train],axis=0)==0]=1;np.testing.assert_array_equal(scale,std)
        gt=(G[train]-mean)/scale;ge=(G[test]-mean)/scale
        for name in NAMES:
            valid=np.array([r['variants'][name]['full']['valid'] for r in rows]);tr=valid[train];te=valid[test]
            for kind in ['mean','channels']:
                key=f'{name}_full7_{kind}';scores[key][test]=scores['G_ready'][test]
                if not tr.any() or set(y[train][tr])!={0,1}:
                    fits.append(dict(held_out_video=video,model=key,fallback='training_feature_rows_lack_both_classes'));continue
                def extra(indices):return np.array([[rows[j]['variants'][name]['full']['scalar']] if kind=='mean' else rows[j]['variants'][name]['full']['channels'] for j in indices])
                model=fit_ridge_logistic(np.c_[gt[tr],extra(train[tr])],y[train][tr])
                if te.any():scores[key][test[te]]=decision_function(model,np.c_[ge[te],extra(test[te])])
                fits.append(dict(held_out_video=video,model=key,training_videos=sorted(set(groups[train][tr])),geometry_normalization=stats,fit=model,test_fallbacks=int((~te).sum())))
        print('ORIGINAL G FOLD',fold+1,32,flush=True)
    assert all(np.isfinite(v).all() for v in scores.values());metrics={k:measures(y,v) for k,v in scores.items()};gates={}
    for name in NAMES:
        for kind in ['mean','channels']:
            key=f'{name}_full7_{kind}';a=metrics[key];bases=[metrics[b] for b in ['G_ready','G_support',f'{name}_old7_{kind}']]
            gates[key]=(a['own_accept']>=max(b['own_accept'] for b in bases) and a['foreign_accept']<min(b['foreign_accept'] for b in bases) and a['balanced_accuracy']>max(b['balanced_accuracy'] for b in bases))
    choices=[k for k,v in gates.items() if v]
    selected='C34_full7_channels' if gates['C34_full7_channels'] else (sorted(choices,key=lambda k:(-metrics[k]['balanced_accuracy'],metrics[k]['foreign_accept'],k.endswith('channels'),NAMES.index(k.rsplit('_',2)[0])))[0] if choices else None)
    rng=np.random.default_rng(20261109);byvideo=[np.flatnonzero(groups==v) for v in sorted(set(groups))];draws=[]
    for _ in range(4000):
        idx=np.concatenate([byvideo[j] for j in rng.integers(len(byvideo),size=len(byvideo))])
        if len(set(y[idx]))==2:draws.append(idx)
    comparisons=[]
    for name in NAMES:
        for kind in ['mean','channels']:comparisons += [(f'{name}_full7_{kind}',b) for b in ['G_ready','G_support',f'{name}_old7_{kind}']]
    primary_bases=['G_ready','G_support','C34_old7_channels','R134_spatial_full7_channels','R134_temporal_full7_channels'];comparisons += [('C34_full7_channels',b) for b in primary_bases];intervals={}
    for current,baseline in comparisons:
        key=current+'__'+baseline
        if key in intervals:continue
        values=[measures(y[i],scores[current][i])['balanced_accuracy']-measures(y[i],scores[baseline][i])['balanced_accuracy'] for i in draws]
        v=dict(point=metrics[current]['balanced_accuracy']-metrics[baseline]['balanced_accuracy'],ci95=np.percentile(values,[2.5,97.5]),valid_draws=len(values))
        if current=='C34_full7_channels' and baseline in primary_bases:v['ci99_primary_family']=np.percentile(values,[.5,99.5])
        intervals[key]=v
    verify(p)
    write('original_geometry_predictions.json',dict(passed=True,records=[dict(video=r['video'],uid=r['uid'],old=r['old'],raw_id=r['raw_id'],label=r['label'],scores={k:z[i] for k,z in scores.items()}) for i,r in enumerate(rows)],folds=fits,old_models_scores_and_geometry_normalization_reused=True,no_new_maps_or_correlations=True))
    write('original_geometry_results.json',dict(passed=True,edges=91,own=15,foreign=76,videos=32,metrics=metrics,pilot_admission=gates,selected_development_arm=selected,paired_video_intervals=intervals,diagnostic_number=3,post_result_explanatory_ablation=True,zero_neural_calls=True,zero_ID_assignments=True))
    print(json.dumps(cast(dict(metrics=metrics,gates=gates,selected=selected))),flush=True)

if __name__=='__main__':{'register':register,'fit':fit}[sys.argv[1]]()
