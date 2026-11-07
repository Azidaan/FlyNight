"""Одинаковое обучение десяти вариантов истории наfit84/fit20, безval."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
from pathlib import Path
from datetime import datetime
import sys,time,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.continuation_service_20261028 import RUN,service
read,write,sha,now=service.read,service.write,service.sha,service.now


def torch_save(torch,path,value):
    temp=path.with_suffix('.tmp');torch.save(value,temp);temp.replace(path)


def main():
    import numpy as np,torch
    from tracking.gpu_lock import GpuLock
    from tracking.history_pair_features import from_cache
    from tracking.history_pair_model import HistoryPairHead,weighted_pair_loss
    folder=RUN/'history_factorial_training';p=read(folder/'protocol.json')
    for predecessor in ['history_visual_audit','history_model_checks']:
        f=RUN/predecessor;c=read(f/'controller_result.json')
        assert c['exit_code']==0 and c['all_owned_ended'] and not any(service.matching(k,v) for k,v in c['owned'].items())
        assert read(f/'report.json')['passed']
    for name,digest in p['sha256'].items():assert sha(ROOT/name)==digest,name
    end=datetime.fromisoformat(read(folder/'execution_deadline.json')['deadline']).timestamp()
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    visual=read(RUN/'history_visual_inputs/report.json');assert visual['complete']
    entries=visual['completed'];counts=np.zeros(2,np.int64); sources={};loaded={}
    for e in entries:
        v=e['video'];pair=RUN/'history_pair_dataset/videos'/v;cache=RUN/'history_visual_inputs/videos'/v
        with np.load(pair/'inputs.npz') as z:x={k:z[k] for k in z.files}
        with np.load(pair/'targets.npz') as z:y=z['identity'].copy();known=z['known'].copy()
        with np.load(cache/'index.npz') as z:index={k:z[k] for k in z.files}
        assert np.array_equal(known,y>=0) and sha(pair/'inputs.npz')==e['input_sha256']
        assert all(sha(cache/k)==h for k,h in e['sha256'].items())
        if e['split']=='fit84':counts+=np.bincount(y[known],minlength=2)
        loaded[v]=(x,y,index,np.load(cache/'embeddings.npy',mmap_mode='r'),np.load(cache/'images.npy',mmap_mode='r'))
        sources[v]=dict(split=e['split'],inputs_sha256=sha(pair/'inputs.npz'),targets_sha256=sha(pair/'targets.npz'),visual=e['sha256'])
    assert all(counts>0) and {e['split'] for e in entries}=={'fit84','fit20'}
    class_weights=counts.sum()/(2*counts)
    write(folder/'data_manifest.json',dict(videos=sources,train_counts=counts.tolist(),class_weights=class_weights.tolist(),val_used=False))
    train=[e for e in entries if e['split']=='fit84'];dev=[e for e in entries if e['split']=='fit20']
    assert len(train)==84 and len(dev)==20
    completed=[]
    with GpuLock('Local\\DissertationSingleGpu'):
        weights=torch.tensor(class_weights,dtype=torch.float32,device='cuda')
        for seed in p['seeds']:
            for variant in p['variants']:
                out=folder/'models'/f'{variant}_seed{seed}';out.mkdir(parents=True,exist_ok=True)
                if (out/'report.json').is_file():
                    r=read(out/'report.json');assert sha(out/'best.pt')==r['best_sha256'];completed.append(r);continue
                torch.manual_seed(seed);model=HistoryPairHead().cuda()
                initial=hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest()
                optimizer=torch.optim.AdamW(model.parameters(),lr=p['learning_rate'],weight_decay=p['weight_decay'])
                epochs=[];best=float('inf');start_epoch=0
                if (out/'resume.pt').is_file():
                    saved=torch.load(out/'resume.pt',map_location='cuda',weights_only=False)
                    model.load_state_dict(saved['state']);optimizer.load_state_dict(saved['optimizer']);epochs=saved['epochs'];best=saved['best'];start_epoch=len(epochs)
                for epoch in range(start_epoch,p['epochs']):
                    epoch_start=time.monotonic();rng=np.random.default_rng(seed+1000*epoch)
                    ordered=[train[i] for i in rng.permutation(len(train))];model.train();train_sum=0.;train_n=0
                    for e in ordered:
                        v=e['video'];x,y,ind,emb,img=loaded[v];rows=rng.permutation(np.flatnonzero(y>=0))
                        for start in range(0,len(rows),p['batch_size']):
                            if time.time()>end-30:raise TimeoutError('History training original budget')
                            selected=rows[start:start+p['batch_size']]
                            f=from_cache(x,ind,emb,img,selected,variant)
                            batch={k:torch.from_numpy(vv).to('cuda') for k,vv in f.items()}
                            target=torch.from_numpy(y[selected]).to('cuda')
                            optimizer.zero_grad(set_to_none=True);output=model(batch)
                            loss=weighted_pair_loss(output,target,weights)
                            assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
                            train_sum+=float(loss.detach())*len(selected);train_n+=len(selected)
                        write(folder/'status.json',dict(state='training',seed=seed,variant=variant,epoch=epoch+1,epochs=p['epochs'],video=v,models_done=len(completed),models_total=len(p['seeds'])*len(p['variants']),updated=now()),required=False)
                    model.eval();dev_sum=0.;dev_n=0;predictions=[]
                    with torch.no_grad():
                        for e in dev:
                            v=e['video'];x,y,ind,emb,img=loaded[v];rows=np.flatnonzero(y>=0);scores=[]
                            for start in range(0,len(rows),p['batch_size']):
                                selected=rows[start:start+p['batch_size']]
                                f=from_cache(x,ind,emb,img,selected,variant)
                                batch={k:torch.from_numpy(vv).to('cuda') for k,vv in f.items()}
                                output=model(batch);target=torch.from_numpy(y[selected]).to('cuda')
                                loss=weighted_pair_loss(output,target,weights)
                                dev_sum+=float(loss)*len(selected);dev_n+=len(selected);scores.append(output.cpu().numpy())
                            predictions.append((v,rows,np.concatenate(scores) if scores else np.empty(0,np.float32)))
                    assert train_n==int(counts.sum()) and dev_n>0
                    item=dict(epoch=epoch+1,train_loss=train_sum/train_n,dev_loss=dev_sum/dev_n,seconds=time.monotonic()-epoch_start)
                    epochs.append(item)
                    if item['dev_loss']<best:
                        best=item['dev_loss'];torch_save(torch,out/'best.pt',dict(state={k:v.cpu() for k,v in model.state_dict().items()},seed=seed,variant=variant,epoch=epoch+1,initial_sha256=initial,protocol_sha256=sha(folder/'protocol.json')))
                        pred=out/'dev_predictions';pred.mkdir(exist_ok=True)
                        for v,rows,scores in predictions:np.savez_compressed(pred/(v+'.npz'),rows=rows,logits=scores)
                    torch_save(torch,out/'resume.pt',dict(state=model.state_dict(),optimizer=optimizer.state_dict(),epochs=epochs,best=best))
                    write(out/'epochs.json',epochs)
                    if not (folder/'price.json').exists():
                        remaining=p['epochs']*len(p['variants'])*len(p['seeds'])-1
                        available=end-time.time();need=2*item['seconds']*remaining+300
                        price=dict(first_full_epoch_seconds=item['seconds'],remaining_epochs=remaining,conservative_seconds=need,available_seconds=available,proceed=need<available)
                        write(folder/'price.json',price)
                        if not price['proceed']:
                            write(folder/'report.json',dict(complete=False,stopped_on_cost=True,completed=completed,partial_model=f'{variant}_seed{seed}',saved_epoch=1,hota_measured=False,scientific_verdict=None,updated=now()));return
                r=dict(seed=seed,variant=variant,epochs=epochs,best_dev_loss=best,best_sha256=sha(out/'best.pt'),initial_sha256=initial,
                    parameters=sum(v.numel() for v in model.parameters()),dev_prediction_sha256={f.stem:sha(f) for f in (out/'dev_predictions').glob('*.npz')})
                write(out/'report.json',r);completed.append(r);write(folder/'completed_models.json',completed)
    write(folder/'report.json',dict(complete=True,completed=completed,models=len(completed),data_manifest_sha256=sha(folder/'data_manifest.json'),val_used=False,hota_measured=False,scientific_verdict=None,updated=now()))


if __name__=='__main__':main()
