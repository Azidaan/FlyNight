"""Обучить шесть одинаковых моделей геометрии; выбор по dev, val не читается."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='0';os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2'
from pathlib import Path
from datetime import datetime
import sys,time,copy,math
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.recovery_research_service_20261027 import RUN as BASE,read,write,sha,now
RUN=BASE/'pixel_box_learning'

def run():
    import numpy as np,torch
    from torch.nn import functional as F
    from tracking.pixel_box_refinement import make_model
    from tracking.gpu_lock import GpuLock
    data=BASE/'pixel_box_dataset';assert read(data/'audit.json')['passed']
    done=read(BASE/'pixel_box_audit/controller_result.json');assert done['exit_code']==0 and done['all_owned_ended']
    p=read(RUN/'protocol.json');end=datetime.fromisoformat(read(RUN/'execution_deadline.json')['deadline']).timestamp();started=time.monotonic()
    for name,h in p['sha256'].items():assert sha(ROOT/name)==h,name
    torch.set_num_threads(2)
    def status(**kw):
        if time.time()>end-10 or (RUN/'stop.request').exists():raise TimeoutError('Pixel learning original budget')
        write(RUN/'status.json',dict(state='running',updated=now(),**kw),required=False)
    arrays={}
    for split in ['train','dev']:
        ims=[];ms=[];ys=[];all_count=0
        for item in read(data/'report.json')['videos']:
            if item['split']!=split:continue
            folder=data/split/item['video'];z=np.load(folder/'targets.npz');keep=z['known'];all_count+=len(keep)
            im=np.load(folder/'images.npy',mmap_mode='r');ims.append(np.asarray(im[keep]));ms.append(z['M'][keep]);ys.append(z['Y'][keep])
        arrays[split]=(np.concatenate(ims),np.concatenate(ms),np.concatenate(ys));status(stage='Данные в памяти',split=split,known=len(arrays[split][0]),all=all_count)
    train=arrays['train'];dev=arrays['dev'];batch=p['batch_size'];steps=math.ceil(len(train[0])/batch)
    models=[];curves={};price_steps=min(100,steps);measured_start=None;global_steps=0
    def tensors(values,ids):
        images,meta,target=values
        x=torch.from_numpy(images[ids]).to('cuda',dtype=torch.float32).permute(0,1,4,2,3)/255
        x=(x-.5)/.25
        return x,torch.from_numpy(meta[ids]).to('cuda'),torch.from_numpy(target[ids]).to('cuda')
    with GpuLock('Local\\DissertationSingleGpu'):
        for arm in p['arms']:
            for seed in p['seeds']:
                torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);rng=np.random.default_rng(seed)
                model=make_model().to('cuda');optim=torch.optim.AdamW(model.parameters(),lr=p['lr'],weight_decay=p['weight_decay'])
                name=arm+'_seed'+str(seed);curve=[];best=float('inf');best_epoch=None;best_state=None
                for epoch in range(p['epochs']):
                    order=rng.permutation(len(train[0]));model.train();total=0
                    for offset in range(0,len(order),batch):
                        status(stage='Обучение геометрии',arm=arm,seed=seed,epoch=epoch+1,epochs=p['epochs'],samples_done=offset,total=len(order))
                        x,m,y=tensors(train,order[offset:offset+batch]);optim.zero_grad(set_to_none=True)
                        if measured_start is None:torch.cuda.synchronize();measured_start=time.monotonic()
                        out=model(x,m,arm=='temporal');loss=F.smooth_l1_loss(out,y,beta=.05);assert torch.isfinite(loss)
                        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10);optim.step();total+=float(loss.detach())*len(x);global_steps+=1
                        if global_steps==price_steps:
                            torch.cuda.synchronize();seconds=time.monotonic()-measured_start
                            remaining=steps*p['epochs']*len(p['arms'])*len(p['seeds'])-global_steps
                            # Dev each epoch + saving overhead included conservatively.
                            cost=2*seconds/global_steps*remaining*(1+len(dev[0])/len(train[0]))+180
                            available=end-time.time();write(RUN/'price.json',dict(measured_steps=global_steps,seconds=seconds,remaining_steps=remaining,conservative_remaining_seconds=cost,available_seconds=available,proceed=cost<available));assert cost<available,'Training cost gate'
                    model.eval();validation=0
                    with torch.inference_mode():
                        for offset in range(0,len(dev[0]),batch):
                            ids=np.arange(offset,min(offset+batch,len(dev[0])));x,m,y=tensors(dev,ids);out=model(x,m,arm=='temporal')
                            validation+=float(F.smooth_l1_loss(out,y,beta=.05))*len(x)
                    validation/=len(dev[0]);curve.append(dict(epoch=epoch+1,train=total/len(train[0]),dev=validation))
                    if validation<best:best=validation;best_epoch=epoch+1;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                    curves[name]=curve;write(RUN/'curves.json',curves)
                checkpoint=RUN/(name+'.pt');torch.save(dict(state_dict=best_state,arm=arm,seed=seed,epoch=best_epoch,dev_loss=best),checkpoint)
                models.append(dict(name=name,arm=arm,seed=seed,checkpoint=checkpoint.relative_to(BASE).as_posix(),sha256=sha(checkpoint),best_epoch=best_epoch,dev_loss=best,parameters=sum(v.numel() for v in model.parameters())))
                write(RUN/'completed_models.json',models)
        assert len({x['parameters'] for x in models})==1
    write(RUN/'report.json',dict(complete=True,models=models,curves=curves,known_train=len(train[0]),known_dev=len(dev[0]),seconds=time.monotonic()-started,val_used=False,flyvis_core_used=False))
    write(RUN/'status.json',dict(state='completed',updated=now()))

if __name__=='__main__':
    try:run()
    except Exception as e:write(RUN/'status.json',dict(state='failed',error=repr(e),updated=now()));raise
