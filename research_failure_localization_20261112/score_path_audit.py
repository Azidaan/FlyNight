"""Exact arithmetic audit of saved heads; no fitting, feature intervention or tracker."""
from common import *
import math, subprocess
import numpy as np
DEST=OUT/'score_path'
FILES=['research_agreement_readout_20261112/protocol.json','research_agreement_readout_20261112/features.json','research_agreement_readout_20261112/predictions.json','runs/domain_matched_native_20261109/deployment.json','research_signal_scaling_20261109/deployment.json','research_signal_scaling_20261109/deployment_screen.json','research_identity_training_stability_20261109/comparison.json','research_domain_matched_reader_20261109/features.json','tracking/channel_identity_head.py','research_failure_localization_20261112/score_path_audit.py']
def put(name,value):
    with (DEST/name).open('x',encoding='utf8') as f:json.dump(cast(value),f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def catalog(fn):
    p=ROOT/'experiments/project_history.json';raw=p.read_bytes();d=json.loads(raw);fn(next(c for c in d['experiments'] if c['id']==ID));assert raw==p.read_bytes();p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
if sys.argv[1]=='register':
    DEST.mkdir(exist_ok=False)
    for n in ['experiments/project_history.json','experiments/system_problems.json','dissertation/ЕДИНЫЙ_ПЛАН.md']:(DEST/Path(n).name).write_bytes((ROOT/n).read_bytes())
    put('protocol.json',dict(created=now(),head=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),inputs={n:sha(ROOT/n) for n in FILES},question='Does the wrong score order arise in geometry or in the weighted visual term of the saved reader?',method='Exact linear sum, centered at the training signal mean. Report every channel and geometry contribution, all five cases for native/common/agreement heads, all26 held-out agreement evaluations. No causal neuron claim; centering is a reference convention, not a deployed alternative.',acceptance=['Reconstruct all stored scores within1e-12 using independent math.fsum and raw coefficient expansion.','Report all cases and all channels without selecting by outcome; unchanged thresholds and weights.'],action='Decide whether geometry-only changes could explain the current visual score failures; no new shift search or physical correspondence surrogate.',new_fits=0,new_maps=0,tracker_calls=0))
    catalog(lambda c:c['acceptance'].append(dict(criterion='Дополнительная арифметическая сверка, записана до расчёта: восстановить оценки сохранённых слоёв из отдельных вкладов геометрии и активности без обучения и изменения порога.',met=None)))
    journal('Постановка дополнительной арифметической сверки: разделить вклады геометрии и зрительных признаков в сохранённых оценках трёх считывателей пяти случаев и 26 проверочных оценках согласия. Новых обучений, карт и вмешательств нет. Источник research_failure_localization_20261112/score_path/protocol.json.')
    print('REGISTERED');sys.exit()
p=read(DEST/'protocol.json')
for n,h in p['inputs'].items():assert sha(ROOT/n)==h,n
ag=read(ROOT/FILES[0]);feat=read(ROOT/FILES[1]);pred=read(ROOT/FILES[2]);native=read(ROOT/FILES[3]);common=read(ROOT/FILES[4]);oldscreen=read(ROOT/FILES[5]);nativepred=read(ROOT/FILES[6]);oldtrain=read(ROOT/FILES[7])['records']
g_names=['original_cost','gap','log_width_ratio','log_height_ratio','dx_over_reference_width','dy_over_reference_height','log1p_admitted_cells']
rows=[]
def record(row,model,theta,stats,signal,mu,scale,expected,dataset):
    theta=np.array(theta);signal=np.array(signal);mu=np.array(mu);scale=np.array(scale);g=(np.array(row['geometry'])-stats['mean'])/stats['scale'];s=(signal-mu)/scale
    gc=g*theta[:7];sc=s*theta[7:-1];bias=float(theta[-1]);z=math.fsum([*gc,*sc,bias]);direct=float(np.r_[g,s]@theta[:-1]+bias)
    raw_bias=bias-math.fsum(mu*theta[7:-1]/scale)-math.fsum(np.array(stats['mean'])*theta[:7]/stats['scale'])
    expanded=math.fsum([raw_bias,*((np.array(row['geometry'])*theta[:7])/stats['scale']),*(signal*theta[7:-1]/scale)])
    assert max(abs(z-expected),abs(z-direct),abs(z-expanded))<1e-12
    rows.append(dict(dataset=dataset,model=model,video=row['video'],frame=row['frame'],old_id=row['old_id'],label=row['label'],score=z,stored_score=expected,error=abs(z-expected),raw_expansion_error=abs(z-expanded),geometry_contributions=dict(zip(g_names,gc)),geometry_sum=math.fsum(gc),intercept=bias,geometry_plus_intercept=math.fsum([*gc,bias]),centered_visual_sum=math.fsum(sc),channel_contributions=sc,channel_values=signal,channel_training_mean=mu,raw_channel_weights=theta[7:-1]/scale,signal_mean=float(np.mean(signal)),all_weights_nonnegative=bool((theta[7:-1]>=0).all()),positive_weight_channels=int((theta[7:-1]>0).sum())))
for dataset in ['train','screen']:
    for i,row in enumerate(feat[dataset]):
        fit=next(f for f in pred['fits'] if f['model']=='C34__agreement' and f['video']==('FINAL' if dataset=='screen' else row['video']))
        record(row,'agreement',fit['fit']['theta'],fit['normalization'],row['variants']['C34']['agreement'],fit['signal_mean'],fit['signal_scale'],pred[dataset][i]['scores']['C34__agreement'],dataset)
for i,row in enumerate(ag['screen']):
    signal=row['old_variants']['C34']['channels'];mu=np.array([r['variants']['C34']['channels'] for r in oldtrain]).mean(0)
    t=np.array(native['models']['C34']['theta']);t[-1]+=mu@t[7:-1]
    expected=next(r for r in nativepred['cases'] if r['old_id']==row['old_id'])['baseline']['C34']
    record(row,'native',t,native['geometry_normalization'],signal,mu,np.ones(34),expected,'screen')
    stats=common['signal_normalization']['C34'];expected=next(r for r in oldscreen['records'] if r['old_id']==row['old_id'])['scores']['C34']
    record(row,'common',common['training_original']['C34']['theta'],common['geometry_normalization'],signal,stats['mean'],stats['scale'],expected,'screen')
comparisons=[]
for model in ['native','common','agreement']:
    foreign=next(r for r in rows if r['dataset']=='screen' and r['model']==model and r['old_id']==574)
    for own_id in [232,832]:
        own=next(r for r in rows if r['dataset']=='screen' and r['model']==model and r['old_id']==own_id)
        diff=np.array(foreign['channel_contributions'])-own['channel_contributions']
        comparisons.append(dict(model=model,own_id=own_id,foreign_id=574,foreign_minus_own_score=foreign['score']-own['score'],geometry_difference=foreign['geometry_sum']-own['geometry_sum'],visual_difference=float(diff.sum()),channel_contribution_differences=diff,foreign_higher_raw_channels=int((np.array(foreign['channel_values'])>own['channel_values']).sum()),visual_channels_favoring_foreign=int((diff>0).sum())))
for n,h in p['inputs'].items():assert sha(ROOT/n)==h,n
put('result.json',dict(passed=True,created=now(),records=rows,comparisons=comparisons,max_score_error=max(r['error'] for r in rows),max_raw_expansion_error=max(r['raw_expansion_error'] for r in rows),scope='Arithmetic localization in saved heads only. No counterfactual physical samples, neurons, new model or ID-quality result.'))
for r in rows:
    if r['dataset']=='screen':print(r['model'],r['old_id'],'G+b',round(r['geometry_plus_intercept'],6),'V',round(r['centered_visual_sum'],6),'z',round(r['score'],6))
print('PAIR_DIFFS',json.dumps(cast(comparisons)))
