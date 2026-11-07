"""Independent NumPy prediction and video bootstrap, without training code or Torch."""
import json,hashlib,csv,gzip,time
from pathlib import Path
import numpy as np
OUT=Path('runs/flyvis_readout_comparison_20261107')
KINDS=('linear','mlp','products','sigma_pi')
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,value):
    with Path(p).open('x',encoding='utf8') as stream:json.dump(value,stream,indent=2,ensure_ascii=False,allow_nan=False)

def predict(kind,g,parameters):
    z=(g-parameters['mu'])/parameters['sd']
    if kind=='linear':residual=np.einsum('nki,i->nk',z,parameters['w'])+parameters['b']
    elif kind=='products':
        terms=np.concatenate((z,np.stack([z[...,i]*z[...,j] for i in range(8) for j in range(i+1,8)],-1)),-1)
        residual=np.einsum('nki,i->nk',terms,parameters['w'])+parameters['b']
    elif kind=='mlp':
        hidden=np.tanh(np.einsum('nki,hi->nkh',z,parameters['w1'])+parameters['b1'])
        residual=np.einsum('nkh,h->nk',hidden,parameters['w2'])+parameters['b2']
    elif kind=='sigma_pi':
        gates=1/(1+np.exp(-parameters['alpha']));mix=1/(1+np.exp(-parameters['gamma']))
        # Explicit scalar factor loop independently implements paper equation8.
        product=np.ones(g.shape[:2],dtype='float64')
        for i in range(8):product*=1-gates[i]+gates[i]*z[...,i]
        linear=np.einsum('nki,i->nk',z,parameters['w'])+parameters['b']
        residual=(1-mix)*linear+mix*parameters['m']*product
    else:raise ValueError(kind)
    return np.sum(g,axis=-1)+residual

def credits(scores,y):
    margin=scores[:,0]-scores[:,1]
    return np.where(abs(margin)<=1e-10,.5,((margin>0)==y).astype(float))

def main():
    started=time.perf_counter();p=read(OUT/'protocol.json');pack=read(OUT/'data_prepared.json');done=read(OUT/'training_completed.json');selected=read(OUT/'selection.json')
    assert done['passed'] and done['selection_sha256']==sha(OUT/'selection.json') and selected['selected_before_evaluation']
    for f,h in p['source_sha256'].items():assert sha(f)==h,f
    with gzip.open('runs/flyvis_exclusive_t4t5_20261107/choices.csv.gz','rt') as stream:
        original={int(r['uid']):r for r in csv.DictReader(stream) if r['exactly2']=='1' and r['label_reason']=='ok'}
    datasets={};provenance={}
    for split in ['fit84','fit20','evaluate26']:
        file=OUT/('data_'+split+'.npz');assert sha(file)==pack['data_sha256'][split]
        with np.load(file) as z:datasets[split]={k:z[k] for k in z.files}
        data=datasets[split];expected={u:r for u,r in original.items() if r['split']==split}
        assert set(map(int,data['uid']))==set(expected)
        np.testing.assert_array_equal(data['g'],data['activation']*data['target'][:,None,:])
        for i,u in enumerate(data['uid']):
            row=expected[int(u)];assert bool(data['y'][i])==(int(row['target_side'])==1)
            assert p['videos'][split][int(data['video'][i])]==row['video']
            provenance[int(u)]=(split,i)
    # Recheck the observable candidate contributions directly against the source shards.
    source_rows=0
    for file,h in pack['source_sha256'].items():
        assert sha(file)==h,file
        if not file.endswith('.npz'):continue
        with np.load(file) as z:
            ids=z['uid'];slots=[(i,int(u)) for i,u in enumerate(ids) if int(u) in provenance]
            if not slots:continue
            x=z['x'];seed=z['seed']
            for index,u in slots:
                split,i=provenance[u];data=datasets[split]
                expected=x[index,1:3,4:12].astype('float64')*seed[index,4:12].astype('float64')
                np.testing.assert_array_equal(expected,data['g'][i]);source_rows+=1
    assert source_rows==len(original)
    fit=datasets['fit84']['g'];mu=fit.reshape(-1,8).mean(0);sd=np.maximum(fit.reshape(-1,8).std(0),1e-8)
    dev=datasets['fit20'];evaluation=datasets['evaluate26'];configs=[];maximum=0.;devpred={}
    for key,receipt in selected['manifest'].items():
        file=OUT/'models'/(key+'.npz');assert sha(file)==receipt['sha256']
        with np.load(file) as z:parameters={k:z[k] for k in z.files}
        np.testing.assert_array_equal(parameters['mu'],mu);np.testing.assert_array_equal(parameters['sd'],sd)
        prediction=predict(receipt['kind'],dev['g'],parameters)
        with np.load(OUT/'predictions'/(key+'_dev.npz')) as z:
            assert np.array_equal(dev['uid'],z['uid']);error=float(abs(prediction-z['scores']).max());maximum=max(maximum,error)
            assert error<=1e-9 and np.array_equal(credits(prediction,dev['y']),credits(z['scores'],dev['y']))
            devpred[key]=z['scores'].copy()
    for kind in KINDS:
        for reg in p['training']['regularization']:
            keys=[f'{kind}_l{reg:g}_s{s}' for s in p['training']['seeds']];mean=np.mean([devpred[k] for k in keys],0)
            margin=mean[:,0]-mean[:,1];loss=float(np.mean(np.logaddexp(0,margin)-dev['y']*margin))
            receipt=selected['manifest'][keys[0]]
            configs.append(dict(kind=kind,regularization=reg,dev_accuracy=float(credits(mean,dev['y']).mean()),dev_bce=loss,parameters=receipt['parameters'],keys=keys))
    best={k:sorted([c for c in configs if c['kind']==k],key=lambda c:(-c['dev_accuracy'],c['dev_bce'],-c['regularization']))[0] for k in KINDS}
    winner=sorted(best.values(),key=lambda c:(-c['dev_accuracy'],c['dev_bce'],c['parameters'],KINDS.index(c['kind'])))[0]
    assert winner==selected['selected'] and best==selected['best_per_family']
    scores={'baseline':evaluation['g'].sum(-1)};seed_effects={}
    baseline_credit=credits(scores['baseline'],evaluation['y'])
    regression=evaluation['regression172'];assert regression.sum()==172 and baseline_credit[regression].sum()==104
    for kind,config in best.items():
        parts=[];seed_effects[kind]=[]
        for key in config['keys']:
            file=OUT/'models'/(key+'.npz');receipt=selected['manifest'][key]
            with np.load(file) as z:parameters={k:z[k] for k in z.files}
            pred=predict(kind,evaluation['g'],parameters)
            file=OUT/'predictions'/(key+'_eval.npz');assert sha(file)==done['evaluation_predictions_sha256'][key]
            with np.load(file) as z:
                assert np.array_equal(z['uid'],evaluation['uid']);error=float(abs(z['scores']-pred).max());maximum=max(maximum,error)
                assert error<=1e-9 and np.array_equal(credits(pred,evaluation['y']),credits(z['scores'],evaluation['y']))
                parts.append(z['scores'].copy())
                seed_effects[kind].append(float(100*(credits(z['scores'],evaluation['y'])-baseline_credit).mean()))
        scores[kind]=np.mean(parts,0)
    votes={k:credits(v,evaluation['y']) for k,v in scores.items()};video=evaluation['video'];nvideo=len(p['videos']['evaluate26'])
    counts=np.bincount(video,minlength=nvideo);draws=np.random.default_rng(42).multinomial(nvideo,np.full(nvideo,1/nvideo),size=20000)
    denominator=draws@counts;valid=denominator>0;draws=draws[valid];denominator=denominator[valid]
    def comparison(a,b,family):
        delta=votes[a]-votes[b];blocks=np.bincount(video,weights=delta,minlength=nvideo)
        boot=100*(draws@blocks)/denominator;alpha=.05/(2*family)
        return dict(name=a+'-'+b,delta_pp=float(delta.mean()*100),ci95_pp=np.quantile(boot,[.025,.975]).tolist(),corrected_confidence=1-.05/family,
            corrected_ci_pp=np.quantile(boot,[alpha,1-alpha]).tolist(),gains=int((delta>0).sum()),losses=int((delta<0).sum()))
    primary=[comparison(k,'baseline',4) for k in KINDS]
    secondary=[comparison('sigma_pi',k,3) for k in KINDS if k!='sigma_pi']
    chosen=winner['kind'];chosen_test=next(x for x in primary if x['name']==chosen+'-baseline')
    main_pass=chosen_test['corrected_ci_pp'][0]>0 and all(d>=0 for d in seed_effects[chosen])
    sigma_pass=all(x['corrected_ci_pp'][0]>0 for x in secondary)
    conditions=[dict(name=k,correct=float(v.sum()),accuracy=float(v.mean()),ties=int((v==.5).sum())) for k,v in votes.items()]
    result=dict(passed=True,pairs=len(video),contributing_videos=int(np.count_nonzero(counts)),cluster_videos=nvideo,fit_counts=pack['counts'],
        conditions=conditions,primary=primary,sigma_secondary=secondary,selected=winner,seed_effects_pp=seed_effects,main=bool(main_pass),sigma_specific_benefit=bool(sigma_pass),
        maximum_numpy_torch_difference=maximum,all_discrete_choices_exact=True,source_rows_exact=source_rows,legacy172_correct=104,seconds=time.perf_counter()-started,
        selection_sha256=sha(OUT/'selection.json'),protocol_sha256=sha(OUT/'protocol.json'),dropped_empty_bootstrap=int((~valid).sum()),
        cases=[dict(uid=int(u),video=p['videos']['evaluate26'][int(v)],correct={k:float(value[i]) for k,value in votes.items()}) for i,(u,v) in enumerate(zip(evaluation['uid'],video))],
        limitation=p['limitation'],next='Prepare matched full tracker protocol and execute authorized conditional check' if main_pass else 'Do not run full tracker for this failed admission; no test-based replacement or tuning.')
    write(OUT/'independent_audit.json',result)
    print(json.dumps({k:result[k] for k in ['conditions','primary','selected','main','sigma_specific_benefit']}),flush=True)

if __name__=='__main__':main()
