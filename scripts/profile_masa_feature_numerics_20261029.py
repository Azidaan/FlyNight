"""One fixed frame: distinguish ROI batching and TF32 from wrong saved vectors."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.continuation_service_20261029 import RUN,service

def main():
    import torch,numpy as np,cv2
    from tracking.masa_r50 import build as r50_build,preprocess as input1024
    from tracking.masa_detic_prepared import build as detic_build
    from tracking.masa_detic_bdd_input import preprocess_bdd
    from tracking.gpu_lock import GpuLock
    torch.set_num_threads(2);cv2.setNumThreads(1);torch.set_default_device('cpu')
    folder=RUN/'masa_feature_numerics';p=service.read(RUN/'masa_query_pilot/protocol.json')
    source=p['sources'][0];video=source['video'];out=RUN/'masa_query_pilot/videos'/video
    ledger=np.load(out/'vectors.npz');frame=service.read(out/'tracking_completed.json')['query_frames'][0]
    capture=cv2.VideoCapture(str(ROOT/source['video_path']))
    try:
        for _ in range(frame+1):ok,image=capture.read();assert ok
    finally:capture.release()
    data=[]
    with GpuLock('Local\\DissertationSingleGpu'):
        models={'r50':r50_build('cpu'),'detic':detic_build('cpu')}
        flags=dict(cudnn_tf32=torch.backends.cudnn.allow_tf32,matmul_tf32=torch.backends.cuda.matmul.allow_tf32)
        for name in ['r50_1024','r50_1333','detic_1333']:
            model=models['detic' if name.startswith('detic') else 'r50']
            model.cpu();model.requires_grad_(False)
            pre=input1024 if name.endswith('1024') else preprocess_bdd
            boxes=ledger[f'{frame}_{name}_boxes'];saved=ledger[f'{frame}_{name}_vectors']
            if not len(boxes):continue
            service.write(folder/'status.json',dict(stage='One-frame numerical discriminator',variant=name,rois=len(boxes),updated=service.now()))
            with torch.inference_mode():
                tensor,scale=pre(image,'cpu');features=model.features(tensor)
                all_at_once=torch.nn.functional.normalize(model.embed_boxes(features,torch.as_tensor(boxes),scale),dim=1).numpy()
                batched=np.concatenate([torch.nn.functional.normalize(model.embed_boxes(features,torch.as_tensor(boxes[i:i+16]),scale),dim=1).numpy() for i in range(0,len(boxes),16)])
                del tensor,features
                model.cuda();tensor,scale=pre(image,'cuda')
                values={}
                for mode in ['original_flags','tf32_off']:
                    if mode=='tf32_off':
                        torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
                    features=model.features(tensor)
                    z=np.concatenate([torch.nn.functional.normalize(model.embed_boxes(features,torch.as_tensor(boxes[i:i+16],device='cuda'),scale),dim=1).cpu().numpy() for i in range(0,len(boxes),16)])
                    values[mode]=dict(vs_saved=float(np.max(abs(z-saved))),vs_cpu_batched=float(np.max(abs(z-batched))))
                    del features
                model.cpu();del tensor
                torch.backends.cudnn.allow_tf32=flags['cudnn_tf32'];torch.backends.cuda.matmul.allow_tf32=flags['matmul_tf32']
            data.append(dict(variant=name,rois=len(boxes),cpu_all_vs_saved=float(np.max(abs(all_at_once-saved))),
                cpu_batched_vs_saved=float(np.max(abs(batched-saved))),cpu_batch_effect=float(np.max(abs(all_at_once-batched))),gpu=values))
            service.write(folder/'partial.json',dict(flags=flags,comparisons=data,scientific_verdict=None))
    service.write(folder/'report.json',dict(complete=True,diagnostic_only=True,scientific_verdict=None,original_tolerance=p['feature_cpu_gpu_normalized_tolerance'],flags=flags,comparisons=data,updated=service.now()))

if __name__=='__main__':main()
