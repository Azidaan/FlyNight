"""Короткая GPUцена замороженного извлекателя и новой головы; не проверка качества."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='0';os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2'
from pathlib import Path
from datetime import datetime,timezone
import json,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.recovery_research_service_20261027 import RUN as BASE,read,write,sha
RUN=BASE/'appearance_gpu_profile'

def run():
    p=read(RUN/'protocol.json');data=read(BASE/'admission_dataset/report.json');assert data['complete']
    for n,h in p['sha256'].items():assert sha(ROOT/n)==h,n
    import cv2,numpy as np,torch
    from tracking.busca_runtime import load_model
    from tracking.gpu_lock import GpuLock
    torch.set_num_threads(2);torch.manual_seed(27);torch.cuda.manual_seed_all(27)
    started=time.monotonic();end=datetime.fromisoformat(read(RUN/'execution_deadline.json')['deadline']).timestamp()
    def status(stage):
        if time.time()>end-3 or (RUN/'stop.request').exists():raise TimeoutError('GPU profile deadline')
        write(RUN/'status.json',dict(state='running',stage=stage,updated=datetime.now(timezone.utc).isoformat()),required=False)
    video=p['video'];folder=BASE/'admission_dataset/train'/video;records=read(folder/'records.json')[:64];assert records
    input_began=time.monotonic();images=[]
    for row in records:
        for key in ['past','current']:
            image=cv2.imread(str(folder/row[key]));assert image is not None and image.shape==(384,128,3);images.append(image)
    array=np.stack(images).astype(np.float32)/255
    array=(array-np.array([.406,.456,.485],np.float32))/np.array([.225,.224,.299],np.float32)
    # Авторская BGRнормировка, затем RGBCHW, как в BUSCA. BNявно заморожена для причинного кодирования.
    tensor=torch.from_numpy(array[...,::-1].copy()).permute(0,3,1,2)
    input_seconds=time.monotonic()-input_began
    with GpuLock('Local\\DissertationSingleGpu'):
        status('Строгая загрузка и явная заморозка нормировки');model,coverage=load_model('cuda');encoder=model.reid_encoder
        encoder.use_domain_adaptation=False;encoder.model.eval()
        before={n:t.clone() for n,t in encoder.model.named_buffers()}
        features=[];torch.cuda.reset_peak_memory_stats();begin=time.monotonic()
        with torch.inference_mode():
            for offset in range(0,len(tensor),16):
                status('Стоимость изображения реальной области');_,z=encoder(tensor[offset:offset+16].to('cuda'));assert torch.isfinite(z).all();features.append(z.cpu())
        torch.cuda.synchronize();feature_seconds=time.monotonic()-begin;features=torch.cat(features)
        assert all(torch.equal(before[n],t) for n,t in encoder.model.named_buffers())
        with torch.inference_mode():
            _,together=encoder(tensor[:4].to('cuda'))
            separate=torch.cat([encoder(tensor[i:i+1].to('cuda'))[1] for i in range(4)])
        similarity=torch.nn.functional.cosine_similarity(together,separate,dim=1)
        assert torch.all(similarity>.99999),'Признак области зависит от соседних/будущих образцов'
        x=features.reshape(len(records),2,-1);x=torch.cat([x[:,0],x[:,1],abs(x[:,0]-x[:,1]),x[:,0]*x[:,1]],dim=1).to('cuda')
        head=torch.nn.Sequential(torch.nn.LayerNorm(x.shape[1]),torch.nn.Linear(x.shape[1],32),torch.nn.ReLU(),torch.nn.Linear(32,1)).to('cuda')
        optimizer=torch.optim.AdamW(head.parameters(),lr=.001);target=torch.tensor([float(r['target']==1) for r in records],device='cuda')
        known=torch.tensor([r['target']>=0 for r in records],device='cuda');assert known.any()
        losses=[];begin=time.monotonic()
        for step in range(20):
            optimizer.zero_grad();loss=torch.nn.functional.binary_cross_entropy_with_logits(head(x)[known,0],target[known]);assert torch.isfinite(loss);loss.backward();optimizer.step();losses.append(float(loss))
        torch.cuda.synchronize();head_seconds=time.monotonic()-begin
        total_candidates=sum(v['candidates'] for v in data['videos']);estimate=2*(feature_seconds+input_seconds)/len(tensor)*(2*total_candidates)+300
        write(RUN/'report.json',dict(complete=True,technical_profile_only=True,real_model_improvement=False,weights_saved=False,
            strict_weight_coverage=coverage,domain_adaptation_batchnorm=False,batchnorm_buffers_unchanged=True,
            batch_vs_single_cosine=similarity.cpu().tolist(),input_images=len(tensor),input_seconds=input_seconds,feature_seconds=feature_seconds,head20_seconds=head_seconds,
            head_losses=losses,features_per_image=features.shape[1],head_parameters=sum(t.numel() for t in head.parameters()),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),estimated_full_feature_seconds_conservative=estimate,
            cost_gate_for_next_feature_stage=estimate<min(3600,datetime.fromisoformat(p['author_stop']).timestamp()-time.time()),
            seconds=time.monotonic()-started,limitation=p['limitation']))
    write(RUN/'status.json',dict(state='completed',updated=datetime.now(timezone.utc).isoformat()))

if __name__=='__main__':
    try:run()
    except Exception as e:write(RUN/'status.json',dict(state='failed',error=repr(e),updated=datetime.now(timezone.utc).isoformat()));raise
