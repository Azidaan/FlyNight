"""Полное фиксированное обучение двух одинаковых голов на отдельных train/dev; не итог HOTA."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='0';os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2'
from pathlib import Path
from datetime import datetime,timezone
import json,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.recovery_research_service_20261027 import RUN as BASE,read,write,sha
RUN=BASE/'admission_learning'

def run():
    p=read(RUN/'protocol.json');feature_report=read(BASE/'appearance_features/report.json');assert feature_report['complete']
    controller=read(BASE/'appearance_features/controller_result.json');assert controller['exit_code']==0 and controller['all_owned_ended']
    for n,h in p['sha256'].items():assert sha(ROOT/n)==h,n
    import numpy as np,torch
    from tracking.learned_recovery_admission import head,raw_input
    from tracking.gpu_lock import GpuLock
    torch.set_num_threads(2);device='cuda';started=time.monotonic();end=datetime.fromisoformat(read(RUN/'execution_deadline.json')['deadline']).timestamp()
    def status(**kw):
        if time.time()>end-5 or (RUN/'stop.request').exists():raise TimeoutError('Admission learning budget')
        write(RUN/'status.json',dict(state='running',updated=datetime.now(timezone.utc).isoformat(),**kw),required=False)
    data={};video_rows={};source_hashes={}
    for split in ['train','dev']:
        xs=[];es=[];ys=[];cursor=0;video_rows[split]=[]
        for item in feature_report['videos']:
            if item['split']!=split:continue
            path=BASE/'appearance_features'/split/(item['video']+'.npz');assert sha(path)==item['output_sha256'];source_hashes[path.relative_to(ROOT).as_posix()]=item['output_sha256']
            z=np.load(path);xs.append(z['X']);es.append(z['E']);ys.append(z['y']);video_rows[split].append(dict(video=item['video'],start=cursor,count=len(z['y'])));cursor+=len(z['y'])
        data[split]=dict(X=np.concatenate(xs),E=np.concatenate(es),y=np.concatenate(ys))
        assert set(np.unique(data[split]['y'])).issubset({-1,0,1})
    known=data['train']['y']>=0;assert set(data['train']['y'][known])=={0,1}
    numeric_mean=data['train']['X'][known].mean(0);numeric_std=np.maximum(data['train']['X'][known].std(0),.001)
    feature_dim=feature_report['feature_dim'];projection=np.random.default_rng(p['projection_seed']).normal(0,1/np.sqrt(len(numeric_mean)),(len(numeric_mean),4*feature_dim)).astype(np.float32)
    results=[];expected_parameters=None;all_cost_started=None
    with GpuLock('Local\\DissertationSingleGpu'):
        for arm in p['arms']:
            raw={s:raw_input(d['X'],d['E'],arm,numeric_mean,numeric_std,projection) for s,d in data.items()}
            mu=raw['train'][known].mean(0);std=np.maximum(raw['train'][known].std(0),.001)
            np.savez_compressed(RUN/(arm+'_normalization.npz'),numeric_mean=numeric_mean,numeric_std=numeric_std,projection=projection,input_mean=mu,input_std=std)
            normalized={s:np.clip((v-mu)/std,-10,10).astype(np.float32) for s,v in raw.items()}
            tensors={s:torch.from_numpy(normalized[s][data[s]['y']>=0]).to(device) for s in data}
            targets={s:torch.from_numpy(data[s]['y'][data[s]['y']>=0].astype(np.float32)).to(device) for s in data}
            assert len(targets['train']) and len(targets['dev'])
            for seed in p['seeds']:
                torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);generator=torch.Generator(device=device).manual_seed(seed)
                model=head(tensors['train'].shape[1]).to(device);count=sum(x.numel() for x in model.parameters())
                if expected_parameters is None:expected_parameters=count
                assert count==expected_parameters
                optimizer=torch.optim.AdamW(model.parameters(),lr=p['lr'],weight_decay=p['weight_decay'])
                curves=[];best=float('inf');best_state=None;best_epoch=None;begin=time.monotonic()
                for epoch in range(p['epochs']):
                    status(stage='Обучение допуска',arm=arm,seed=seed,epoch=epoch+1,epochs=p['epochs'],known_train=len(targets['train']))
                    model.train();order=torch.randperm(len(targets['train']),device=device,generator=generator);loss_sum=0.
                    for offset in range(0,len(order),p['batch_size']):
                        ids=order[offset:offset+p['batch_size']];optimizer.zero_grad()
                        loss=torch.nn.functional.binary_cross_entropy_with_logits(model(tensors['train'][ids])[:,0],targets['train'][ids]);assert torch.isfinite(loss)
                        loss.backward();optimizer.step();loss_sum+=float(loss.detach())*len(ids)
                    model.eval()
                    with torch.inference_mode():dev_loss=float(torch.nn.functional.binary_cross_entropy_with_logits(model(tensors['dev'])[:,0],targets['dev']))
                    assert np.isfinite(dev_loss)
                    if dev_loss<best:
                        best=dev_loss;best_epoch=epoch+1;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                    curves.append(dict(epoch=epoch+1,train_bce=loss_sum/len(order),dev_bce=dev_loss))
                    if arm==p['arms'][0] and seed==p['seeds'][0] and epoch==2:
                        torch.cuda.synchronize();first=time.monotonic()-begin;remaining_epochs=p['epochs']*len(p['arms'])*len(p['seeds'])-3
                        cost=2*first/3*remaining_epochs+120;available=end-time.time()
                        write(RUN/'price.json',dict(first3_epochs_seconds=first,conservative_remaining_seconds=cost,available_seconds=available,proceed=cost<available));assert cost<available
                name=arm+'_seed'+str(seed)+'.pt';torch.save(dict(state_dict=best_state,arm=arm,seed=seed,input_dim=tensors['train'].shape[1],best_epoch=best_epoch),RUN/name)
                write(RUN/(arm+'_seed'+str(seed)+'_curve.json'),curves)
                model.load_state_dict(best_state)
                predictions={}
                for split in data:
                    full=torch.from_numpy(normalized[split]).to(device)
                    with torch.inference_mode():prob=model(full).sigmoid()[:,0].cpu().numpy()
                    assert np.isfinite(prob).all() and ((prob>=0)&(prob<=1)).all()
                    np.save(RUN/(arm+'_seed'+str(seed)+'_'+split+'_probabilities.npy'),prob)
                    predictions[split]=len(prob)
                results.append(dict(arm=arm,seed=seed,parameters=count,input_dim=tensors['train'].shape[1],best_epoch=best_epoch,best_dev_bce=best,
                    checkpoint=name,checkpoint_sha256=sha(RUN/name),normalization=arm+'_normalization.npz',normalization_sha256=sha(RUN/(arm+'_normalization.npz')),predictions=predictions))
                write(RUN/'completed_models.json',results)
        write(RUN/'report.json',dict(complete=True,models=results,source_features_sha256=source_hashes,video_rows=video_rows,
            unknown_excluded_from_loss=True,unknown_predicted_at_inference=True,train_dev_disjoint=True,visual_numeric_head_parameters_equal=True,
            thresholds_selected=False,hota_measured=False,val_used_for_fitting=False,flyvis_core_used=False,seconds=time.monotonic()-started,limitation=p['limitation']))
    write(RUN/'status.json',dict(state='completed',updated=datetime.now(timezone.utc).isoformat()))

if __name__=='__main__':
    try:run()
    except Exception as e:write(RUN/'status.json',dict(state='failed',error=repr(e),updated=datetime.now(timezone.utc).isoformat()));raise
