"""Полные признаки областей на фиксированном извлекателе; без обучения и новых меток."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='0';os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2'
from pathlib import Path
from datetime import datetime,timezone
import json,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.recovery_research_service_20261027 import RUN as BASE,read,write,sha
RUN=BASE/'appearance_features'

def run():
    p=read(RUN/'protocol.json');data=read(BASE/'admission_dataset/report.json');audit=read(BASE/'admission_dataset/audit.json')
    assert data['complete'] and audit['passed'];assert read(BASE/'appearance_gpu_profile/report.json')['cost_gate_for_next_feature_stage']
    for n,h in p['sha256'].items():assert sha(ROOT/n)==h,n
    import cv2,numpy as np,torch
    from tracking.busca_runtime import load_model
    from tracking.gpu_lock import GpuLock
    torch.set_num_threads(2);torch.manual_seed(27);torch.cuda.manual_seed_all(27)
    end=datetime.fromisoformat(read(RUN/'execution_deadline.json')['deadline']).timestamp();started=time.monotonic();items=[];remaining=audit['candidates']
    def status(**kw):
        if time.time()>end-5 or (RUN/'stop.request').exists():raise TimeoutError('Feature extraction deadline')
        write(RUN/'status.json',dict(state='running',updated=datetime.now(timezone.utc).isoformat(),**kw),required=False)
    with GpuLock('Local\\DissertationSingleGpu'):
        status(stage='Строгая загрузка фиксированных весов');model,coverage=load_model('cuda');encoder=model.reid_encoder
        encoder.use_domain_adaptation=False;encoder.model.eval();before={n:t.clone() for n,t in encoder.model.named_buffers()}
        for index,item in enumerate(data['videos']):
            began=time.monotonic();video=item['video'];split=item['split'];folder=BASE/'admission_dataset'/split/video
            source=np.load(folder/'features.npz');records=read(folder/'records.json');assert len(source['X'])==len(records)==item['candidates']
            assert sha(folder/'features.npz')==item['features_sha256'] and sha(folder/'records.json')==item['records_sha256']
            output=[]
            for offset in range(0,len(records),8):
                status(stage='Признаки прошлой и текущей области',video=video,index=index,total=165,pairs_done=offset,pairs_total=len(records))
                images=[]
                for row in records[offset:offset+8]:
                    for key in ['past','current']:
                        image=cv2.imread(str(folder/row[key]));assert image is not None and image.shape==(384,128,3);images.append(image)
                array=np.stack(images).astype(np.float32)/255
                array=(array-np.array([.406,.456,.485],np.float32))/np.array([.225,.224,.299],np.float32)
                tensor=torch.from_numpy(array[...,::-1].copy()).permute(0,3,1,2).to('cuda')
                with torch.inference_mode():_,features=encoder(tensor)
                assert torch.isfinite(features).all();output.append(features.cpu().numpy().reshape(-1,2,features.shape[-1]))
            embeddings=np.concatenate(output) if output else np.zeros((0,2,p['feature_dim']),np.float32)
            assert embeddings.shape==(len(records),2,p['feature_dim'])
            out=RUN/split;out.mkdir(parents=True,exist_ok=True);np.savez_compressed(out/(video+'.npz'),E=embeddings,X=source['X'],y=source['y'])
            assert all(torch.equal(before[n],t) for n,t in encoder.model.named_buffers())
            elapsed=time.monotonic()-began;remaining-=len(records)
            items.append(dict(video=video,split=split,candidates=len(records),seconds=elapsed,output_sha256=sha(out/(video+'.npz')),source_feature_sha256=item['features_sha256'],source_records_sha256=item['records_sha256']))
            write(RUN/'completed_videos.json',items)
            if index==0:
                projected=2*elapsed/max(len(records),1)*remaining+120;available=end-time.time()
                write(RUN/'price.json',dict(first_full_video_seconds=elapsed,first_pairs=len(records),remaining_pairs=remaining,conservative_remaining_seconds=projected,available_seconds=available,proceed=projected<available));assert projected<available
        write(RUN/'report.json',dict(complete=True,videos=items,feature_dim=p['feature_dim'],normalization_matches_profile=True,
            batchnorm_buffers_unchanged=True,domain_adaptation_batchnorm=False,strict_weight_coverage=coverage,
            unknown_labels_unchanged=True,training_run=False,seconds=time.monotonic()-started))
    write(RUN/'status.json',dict(state='completed',updated=datetime.now(timezone.utc).isoformat()))

if __name__=='__main__':
    try:run()
    except Exception as e:write(RUN/'status.json',dict(state='failed',error=repr(e),updated=datetime.now(timezone.utc).isoformat()));raise
