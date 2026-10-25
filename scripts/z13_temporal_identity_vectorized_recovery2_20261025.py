"""Same independent heads/loss/batches, vectorized execution inside the original phase budget."""
from pathlib import Path
from datetime import datetime,timedelta
import copy,os,sys,time,gc
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts import z13_temporal_identity_trial_20261025 as S
from scripts.z13_temporal_identity_trial_recovery1_20261025 import retry_write,old_contrast
OUT=S.RUN/'recovery2'


def policy(logits):
    n=len(logits);arg=np.zeros(n,np.int64);prob=np.zeros(n,float);positive=np.zeros(n,bool)
    lengths=np.array([len(z) for z in logits])
    for k in np.unique(lengths):
        ii=np.flatnonzero(lengths==k);z=np.stack([np.asarray(logits[i],float) for i in ii]);j=z.argmax(1)
        maximum=z[np.arange(len(z)),j];offset=np.maximum(maximum,0.)
        pr=np.exp(maximum-offset)/(np.exp(-offset)+np.exp(z-offset[:,None]).sum(1))
        arg[ii]=j;prob[ii]=pr;positive[ii]=maximum>0
    return arg,prob,positive


def select(cpu,indices,logits,tau,videos,rows_needed=True):
    best,prob,positive=policy(logits);chosen=np.where(positive&(prob>=tau),best,-1)
    vi=cpu['video'][indices];kind=cpu['kind'][indices];known=kind>=0;which=np.maximum(chosen,0)
    good=(chosen>=0)&known&cpu['good'][indices,which];harm=(chosen>=0)&known&cpu['harm'][indices,which]
    net=good.astype(int)-harm.astype(int)
    den=np.bincount(vi[known],minlength=len(videos));num=np.bincount(vi[known],weights=net[known],minlength=len(videos))
    per={v:float(num[i]/den[i]) for i,v in enumerate(videos) if den[i]}
    counts=dict(rows=len(indices),selected=int((chosen>=0).sum()),beneficial=int(good.sum()),harmful=int(harm.sum()),unknown_selected=int(((chosen>=0)&~known).sum()))
    rows=[dict(uid=int(cpu['uid'][index]),video=videos[int(vi[i])],kind=int(kind[i]),action=int(chosen[i]),probability=float(prob[i]),beneficial=bool(good[i]),harmful=bool(harm[i])) for i,index in enumerate(indices)] if rows_needed else []
    return rows,per,counts


def loss_function(torch,template,arm):
    def loss(params,buffers,batch):
        score,association,valid=torch.func.functional_call(template,(params,buffers),tuple(batch[k] for k in ('x','seed','geo','mask','px','pm','pg')),strict=True)
        target=batch['good'].to(score.dtype)-batch['harm'].to(score.dtype)
        action=(((score-target).square().masked_fill(~batch['mask'],0)).sum(1)/batch['mask'].sum(1)).mean()
        use=batch['id_known']&valid;count=use.sum((1,2))
        error=torch.nn.functional.binary_cross_entropy_with_logits(association,batch['id_target'],reduction='none')
        per=error.masked_fill(~use,0).sum((1,2))/count.clamp(min=1)
        identity=per.sum()/(count>0).sum().clamp(min=1)
        return action+(0 if arm=='D_no_identity' else 1)*identity
    return loss


def prepare():
    assert not OUT.exists()
    old=S.read(S.RUN/'controller_result.json');assert old['exit_code']==1 and old['all_owned_ended'] and not old['remaining']
    first=S.RUN/'internal/D/seed0/fold0';assert S.read(first/'done.json')['predictions_reproduced_exact']
    original=S.read(S.RUN/'execution_receipt.json');p=S.read(S.RUN/'protocol.json')
    deadline=datetime.fromisoformat(original['at'])+timedelta(seconds=p['internal_budget_seconds'])
    assert S.now()<deadline
    OUT.mkdir()
    retry_write(OUT/'protocol.json',dict(device='cuda',budget_seconds=(deadline-S.now()).total_seconds(),save_at=deadline.isoformat(),
        hard_stop=(deadline+timedelta(seconds=20)).isoformat(),stop_request_path=(OUT/'stop.request').relative_to(ROOT).as_posix(),
        original_internal_deadline_retained=True,original_training_protocol_sha256=S.sha(S.RUN/'protocol.json'),
        source_sha256=S.sha(Path(__file__)),reason='Same losses,each model1150updates,batch128,Adam.001,all original seeds/UID batches. torch.func.vmap batches independent heads; pure NumPy policy groups equal candidate counts. Existing first model retained,not retrained. Original failed price gate preserved. New price gate after verified numerical parity inside original deadline.'))
    retry_write(OUT/'status.json',dict(state='ready',updated=S.now().isoformat()))


def verify(torch,cpu,data,arm):
    from tracking.z13_temporal_identity_readout import make_model,identity_loss
    from tracking.z13_region_value_loss import signed_action_loss
    folder=S.RUN/'internal/D/seed0/fold0';model=make_model(torch).to('cuda:0');model.load_state_dict(torch.load(folder/'model.pt',map_location='cuda:0',weights_only=True))
    idx=torch.arange(128,device='cuda:0');batch={k:data[k][idx] for k in ('x','seed','geo','mask','px','pm','pg','id_target','id_known','good','harm')}
    a,association,valid=model(*[batch[k] for k in ('x','seed','geo','mask','px','pm','pg')])
    single=signed_action_loss(torch,a,batch['good'],batch['harm'],batch['mask'])+identity_loss(torch,association,batch['id_target'],batch['id_known'],valid)
    ordinary=torch.autograd.grad(single,tuple(model.parameters()),allow_unused=True)
    models=[copy.deepcopy(model),copy.deepcopy(model)];params,buffers=torch.func.stack_module_state(models);template=copy.deepcopy(model).to('meta')
    stacked={k:torch.stack([v,v]) for k,v in batch.items()};fn=loss_function(torch,template,'D')
    grads,values=torch.func.vmap(torch.func.grad_and_value(fn),in_dims=(0,0,0))(params,buffers,stacked)
    maximum=0.
    for (name,param),g in zip(model.named_parameters(),ordinary):
        expected=torch.zeros_like(param) if g is None else g
        maximum=max(maximum,float((expected-grads[name][0]).abs().max()))
    loss_difference=float((values-single.detach()).abs().max());assert maximum<1e-5 and loss_difference<1e-6,(maximum,loss_difference)
    # Compare one ordinary Adam update with the stacked elementwise update, without training any completed head again.
    opt=torch.optim.Adam(model.parameters(),lr=.001)
    for par,g in zip(model.parameters(),ordinary):par.grad=g
    opt.step();batched_opt=torch.optim.Adam(list(params.values()),lr=.001)
    for name,value in params.items():value.grad=grads[name]
    batched_opt.step()
    parameter_difference=max(float((param-params[name][0]).abs().max()) for name,param in model.named_parameters())
    assert parameter_difference<1e-5,parameter_difference
    # Policy parity on the SAME real logits, including historical probabilities and every frozen threshold.
    indices=np.arange(min(512,len(cpu['uid'])))
    reference_model=make_model(torch).to('cuda:0');reference_model.load_state_dict(torch.load(folder/'model.pt',map_location='cuda:0',weights_only=True))
    logits,_=S.predict(torch,reference_model,data,indices)
    raw={k:v.numpy() for k,v in cpu.items()}
    for tau in S.THRESHOLDS:
        left=S.selection(cpu,indices,logits,tau,S.read(S.RUN/'protocol.json')['fit84']+S.read(S.RUN/'protocol.json')['fit20'])
        right=select(raw,indices,logits,tau,S.read(S.RUN/'protocol.json')['fit84']+S.read(S.RUN/'protocol.json')['fit20'])
        assert left==right,'Vectorized probabilities/actions must exactly match original'
    retry_write(OUT/'numerical_parity.json',dict(at=S.now().isoformat(),gradient_max_abs=maximum,loss_max_abs=loss_difference,
        one_adam_update_parameter_max_abs=parameter_difference,all_frozen_thresholds_and_probabilities_exact=True,
        reused_completed_model_for_readonly_gradient_test=True,completed_model_retrained=False))
    del model,models,reference_model,template,params,buffers,opt,batched_opt;gc.collect();torch.cuda.empty_cache()


def train_group(torch,p,cpu,data,videos,descriptors,arm,deadline,group_number,phase):
    from tracking.z13_temporal_identity_readout import make_model
    started=time.monotonic();models=[];batches=[]
    raw={k:v.numpy() for k,v in cpu.items()}
    for d in descriptors:
        torch.manual_seed(d['seed']);models.append(make_model(torch,activity=arm!='geometry',history=arm!='D_blank').to('cuda:0'))
        train=np.flatnonzero(np.isin(raw['video'],[videos.index(v) for v in d['split']['train']])&(raw['kind']>=0))
        rng=np.random.default_rng(d['seed']);batches.append(np.stack([rng.choice(train,p['batch'],replace=False) for _ in range(p['steps'])]))
    params,buffers=torch.func.stack_module_state(models);template=copy.deepcopy(models[0]).to('meta');del models
    idx=torch.from_numpy(np.stack(batches)).to('cuda:0');optimizer=torch.optim.Adam(list(params.values()),lr=p['lr'])
    objective=torch.func.vmap(torch.func.grad_and_value(loss_function(torch,template,arm)),in_dims=(0,0,0))
    keys=('x','seed','geo','mask','px','pm','pg','id_target','id_known','good','harm')
    checkpoint=OUT/'checkpoints'/phase/arm/f'group{group_number}.pt';checkpoint.parent.mkdir(parents=True,exist_ok=True)
    for step in range(p['steps']):
        if step%50==0:
            if S.now()>=deadline or (OUT/'stop.request').exists():raise TimeoutError('Retained original phase deadline')
            retry_write(S.RUN/'status.json',dict(state='running',variant=arm,stage='Независимые модели одним GPU-пакетом',heads=len(descriptors),step=step,steps=p['steps'],models_done=len(list((S.RUN/phase).glob('*/seed*/fold*/done.json'))) if phase=='internal' else 90+len(list((S.RUN/'final').glob('*/seed*/done.json'))),models_total=90 if phase=='internal' else 108,updated=S.now().isoformat()))
        optimizer.zero_grad(set_to_none=True);batch={k:data[k][idx[:,step]] for k in keys};gradients,loss=objective(params,buffers,batch)
        for name,value in params.items():value.grad=gradients[name]
        optimizer.step()
        if (step+1)%50==0:
            assert bool(torch.isfinite(loss).all())
            torch.save(dict(parameters=params,buffers=buffers,optimizer=optimizer.state_dict(),step=step+1,descriptors=descriptors),checkpoint)
    records=[]
    for j,d in enumerate(descriptors):
        folder=Path(d['folder']);folder.mkdir(parents=True,exist_ok=True)
        state={name:value[j].detach().clone() for name,value in params.items()};state.update({name:value[j].detach().clone() for name,value in buffers.items()})
        model=make_model(torch,activity=arm!='geometry',history=arm!='D_blank').to('cuda:0');model.load_state_dict(state)
        calibration=np.flatnonzero(np.isin(raw['video'],[videos.index(v) for v in d['split']['calibration']]))
        validation=np.flatnonzero(np.isin(raw['video'],[videos.index(v) for v in d['split']['validation']]))
        zcal,_=S.predict(torch,model,data,calibration);zval,identity=S.predict(torch,model,data,validation)
        torch.save(model.state_dict(),folder/'model.pt')
        restored=make_model(torch,activity=arm!='geometry',history=arm!='D_blank').to('cuda:0');restored.load_state_dict(torch.load(folder/'model.pt',map_location='cuda:0',weights_only=True))
        replay,replayed_identity=S.predict(torch,restored,data,validation)
        assert all(np.array_equal(a,b) for a,b in zip(zval,replay)) and identity==replayed_identity
        grid=[]
        for tau in S.THRESHOLDS:
            _,per,count=select(raw,calibration,zcal,tau,videos,rows_needed=False)
            grid.append(dict(threshold=tau,mean=float(np.mean(list(per.values()))),counts=count))
        best=max(range(len(grid)),key=lambda i:(grid[i]['mean'],-grid[i]['counts']['selected'],i));tau=S.THRESHOLDS[best]
        rows,per,count=select(raw,validation,zval,tau,videos);retry_write(folder/'rows.json',rows)
        rec=dict(tau=tau,per_video=per,counts=count,calibration=grid,identity_diagnostic=identity,
            seconds=(time.monotonic()-started)/len(descriptors),rows_sha256=S.sha(folder/'rows.json'),vectorized_group=str(checkpoint.relative_to(ROOT)))
        retry_write(folder/'report.json',rec);retry_write(folder/'done.json',dict(model_sha256=S.sha(folder/'model.pt'),report_sha256=S.sha(folder/'report.json'),predictions_reproduced_exact=True,steps=p['steps']))
        records.append(rec);del model,restored
    retry_write(checkpoint.with_suffix('.json'),dict(complete=True,heads=len(descriptors),seconds=time.monotonic()-started,all_steps=1150,all128minibatches_unchanged=True))
    del params,buffers,optimizer,template,idx;gc.collect();torch.cuda.empty_cache()
    return records,time.monotonic()-started


def worker():
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    import torch
    from tracking.gpu_lock import GpuLock
    torch.set_num_threads(2);torch.set_default_device('cpu');p=S.read(S.RUN/'protocol.json');service=S.read(OUT/'protocol.json')
    assert S.sha(Path(__file__))==service['source_sha256']
    deadline=datetime.fromisoformat(service['save_at']);results={};report_hashes={};new_models=0;began=time.monotonic()
    with GpuLock('flyvis_fovea_gpu'):
        for arm in S.ARMS:
            cpu,videos=S.load_data(torch,arm);data={k:v.to('cuda:0') for k,v in cpu.items()}
            if arm=='D':verify(torch,cpu,data,arm)
            pending=[]
            for seed in p['seeds']:
                for fold,split in enumerate(p['folds']):
                    folder=S.RUN/'internal'/arm/f'seed{seed}'/f'fold{fold}'
                    if (folder/'done.json').exists():
                        done=S.read(folder/'done.json');assert S.sha(folder/'model.pt')==done['model_sha256'];assert S.sha(folder/'report.json')==done['report_sha256']
                        results[arm,seed,fold]=S.read(folder/'report.json');report_hashes[folder.relative_to(ROOT).as_posix()]=S.sha(folder/'report.json')
                    else:pending.append(dict(seed=seed,fold=fold,split=split,folder=str(folder)))
            # All pending independent heads of this arm share kernel dispatch, not examples or parameters.
            if pending:
                records,elapsed=train_group(torch,p,cpu,data,videos,pending,arm,deadline,0,'internal');new_models+=len(records)
                for d,r in zip(pending,records):
                    results[arm,d['seed'],d['fold']]=r;folder=Path(d['folder']);report_hashes[folder.relative_to(ROOT).as_posix()]=S.sha(folder/'report.json')
                if arm=='D':
                    estimate=2*elapsed/len(pending)*(90-len(report_hashes))+180;left=(deadline-S.now()).total_seconds()
                    retry_write(OUT/'price.json',dict(group_heads=len(pending),group_seconds=elapsed,conservative_remaining_seconds=estimate,remaining=left,proceed=estimate<left,original_failed_price_preserved=True));assert estimate<left,'Vectorized model price does not fit retained original budget'
            del cpu,data;gc.collect();torch.cuda.empty_cache()
    assert len(report_hashes)==90
    means={a:{} for a in S.ARMS}
    for a in S.ARMS:
        for v in videos:
            values=[]
            for seed in p['seeds']:
                values.append(next(results[a,seed,f]['per_video'][v] for f in range(5) if v in results[a,seed,f]['per_video']))
            means[a][v]=float(np.mean(values))
    comparisons=S.contrast(means);passed=S.passes(comparisons)
    retry_write(S.RUN/'internal_report.json',dict(complete=True,models=90,means=means,paired=comparisons,admit_final=passed,conditional_intervals=True,new26evaluations=0))
    retry_write(S.RUN/'report.json',dict(complete=True,models=90,internal_pass=passed,final_pass=None,final_evaluated=False,
        models_report_sha256=report_hashes,hota_measured=False,core_trained=False,seconds=time.monotonic()-began,
        vectorized_execution=True,original_internal_deadline_retained=service['save_at'],final_pending=passed))
    retry_write(S.RUN/'status.json',dict(state='internal_pass_final_pending' if passed else 'completed',models_done=90,updated=S.now().isoformat()))


if __name__=='__main__':
    if '--prepare' in sys.argv:prepare()
    elif '--worker' in sys.argv:
        try:worker()
        except Exception as e:retry_write(OUT/'failure.json',dict(at=S.now().isoformat(),error=repr(e)));raise
