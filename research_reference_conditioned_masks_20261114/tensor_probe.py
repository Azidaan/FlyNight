from data import *
if sys.argv[1]=='prepare':
    save('tensor_protocol.json',dict(registered=now(),reason='After equal-width76channel comparisons, adding C34toRGB worsened queryIoU on all3cases. Probe frozen joint reader dependence on current C34, without refitting or changing RGB/coordinates.',inputs={**read(OUT/'protocol.json')['inputs'],rel(OUT/'color_readers.npz'):sha(OUT/'color_readers.npz'),rel(OUT/'color_scores.npz'):sha(OUT/'color_scores.npz'),rel(OUT/'color_result.json'):sha(OUT/'color_result.json'),rel(Path(__file__)):sha(__file__)},variants=['baseline','zero_c34: set standardized first68columns to0 (reference mean)','query_recenter: subtract query column means from standardized C34','query_rescale: recenter and divide by query column std, floor1e-8','spatial_shuffle: same fixedseed42row permutation for all68C34columns','reference_c34: replace standardized C34with past reference at same gridcell, a hybrid diagnostic not an asserted physical model'],invariants='Frozen all3seedFlyRGB76weights; same query RGB+Gcolumns bitwise, reference training/labels unchanged, no new core computation. Report every variant, including worsening; no promotion or threshold selection.',deadline='2026-11-14T08:55:00+03:00'))
    journal('Зарегистрировано вмешательство в фиксированный совместный считыватель: после измеренного ухудшения при добавленииC34кRGBпроверяются нейтрализация, перенормировка, перемешивание и подстановка прошлой активности при неизменныхRGB, координатах и весах. Источник: research_reference_conditioned_masks_20261114/tensor_protocol.json. Это диагностические гибридные входы, не новая рабочая версия.')
    raise SystemExit
p=read(OUT/'protocol.json');tp=read(OUT/'tensor_protocol.json');w=np.load(OUT/'color_readers.npz');saved=np.load(OUT/'color_scores.npz');results=[];all_scores={};moments=[];maxerr=0.
for n,h in tp['inputs'].items():assert sha(ROOT/n)==h,n
for case in [c for c in p['cases'] if c['primary']]:
    ref=load_record(next(r for r in p['records'] if r['id']==case['reference']));query=load_record(next(r for r in p['records'] if r['id']==case['query']));prefix=case['id']+'__FlyRGB76';mu=w[prefix+'__mu'];sd=w[prefix+'__sd'];raw=np.c_[query['features']['FlyVis'],query['features']['RGB+G']];base=(raw-mu)/sd;rraw=np.c_[ref['features']['FlyVis'],ref['features']['RGB+G']];reference=(rraw-mu)/sd
    moments.append(dict(case=case['id'],max_abs_query_C34_mean=float(abs(base[:,:68].mean(0)).max()),C34_abs_p99=float(np.quantile(abs(base[:,:68]),.99))))
    for mode in ['baseline','zero_c34','query_recenter','query_rescale','spatial_shuffle','reference_c34']:
        x=base.copy()
        if mode=='zero_c34':x[:,:68]=0
        elif mode in ['query_recenter','query_rescale']:
            x[:,:68]-=x[:,:68].mean(0)
            if mode=='query_rescale':x[:,:68]/=np.maximum(x[:,:68].std(0),1e-8)
        elif mode=='spatial_shuffle':x[:,:68]=x[np.random.default_rng(42).permutation(721),:68]
        elif mode=='reference_c34':x[:,:68]=reference[:,:68]
        assert np.array_equal(x[:,68:],base[:,68:]);x=x.astype('float32');ss=[]
        for seed in [42,43,44]:
            pref=prefix+'__'+str(seed)+'__';h=np.maximum(0,x@w[pref+'0.weight'].T+w[pref+'0.bias']);s=h@w[pref+'2.weight'].T+w[pref+'2.bias'];s-=s.max(1,keepdims=True);e=np.exp(s);ss.append(e/e.sum(1,keepdims=True))
        score=np.mean(ss,0);all_scores[case['id']+'__'+mode]=score
        if mode=='baseline':
            err=float(np.max(np.abs(score-saved[case['id']+'__FlyRGB76__query'])));maxerr=max(maxerr,err);assert err<1e-5
        metrics,_=measure(score.argmax(1),query)
        if not metrics['inside_negative_pixels']:metrics['inside_fpr']=None
        results.append(dict(case=case['id'],variant=mode,**metrics))
with (OUT/'tensor_scores.npz').open('xb') as f:np.savez_compressed(f,**all_scores)
for n,h in tp['inputs'].items():assert sha(ROOT/n)==h,n
save('tensor_result.json',dict(records=results,moments=moments,source_unchanged=True,RGB_and_coordinates_bitwise_unchanged=True,baseline_max_score_error=maxerr,scope='Frozen joint-readout interventions; no inference that normalization or C34is a universal cause.'))
print(json.dumps(results),flush=True)
