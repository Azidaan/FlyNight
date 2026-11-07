"""Compare actual normalized association vectors, before any validation video.

Old raw-feature FP32 test remains failed. This additional prospective contract
tests the downstream API, not a retroactive change to the old tolerance.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.continuation_service_20261029 import RUN,service

def main():
    import cv2,torch,numpy as np
    from torchvision.models import swin_b
    from tracking.masa_detic_prepared import build
    from tracking.masa_detic_bdd_input import preprocess_bdd
    from scripts.run_bdd_video_rate_experiment import load_detection_cache,detections_for_frame
    folder=RUN/'masa_output_interface';p=service.read(folder/'protocol.json')
    for path,digest in p['sha256'].items():assert service.sha(ROOT/path)==digest,path
    torch.set_default_device('cpu');torch.set_num_threads(1);cv2.setNumThreads(1)
    # Reuse the fully strict reviewed mapping; do not rerun its failed old gate.
    source=ROOT/'scripts/check_masa_detic_backbone_reference_recovery2_20261029.py'
    code=source.read_text(encoding='utf-8')
    start=code.index("    model=build('cpu')")
    end=code.index('    checked=[]',start)
    import textwrap
    ns=dict(torch=torch,swin_b=swin_b,build=build)
    exec(compile(textwrap.dedent(code[start:end]),str(source),'exec'),ns)
    model,reference=ns['model'],ns['reference']
    before={k:v.clone() for k,v in model.named_buffers()}
    video='00e11419-e4eeba84';cache=load_detection_cache(ROOT/f'runs/train_cache/{video}/detections/raw_video.npz')
    cap=cv2.VideoCapture(str(ROOT/f'data/bdd100k/videos/train/{video}.mov'))
    checked=[]
    try:
        for frame in range(max(p['frames'])+1):
            ok,image=cap.read();assert ok
            if frame not in p['frames']:continue
            service.write(folder/'status.json',dict(stage='Comparing full normalized output interface',frame=frame,updated=service.now()))
            boxes,scores,_=detections_for_frame(cache,frame);boxes=boxes[scores>=.1]
            assert len(boxes)>0
            tensor,scale=preprocess_bdd(image)
            assert tensor.shape[-2]%32==tensor.shape[-1]%32==0
            assert min(tensor.shape[-2:])//32>7
            began=time.monotonic()
            with torch.inference_mode():
                own=model.features(tensor)
                x=tensor;ref=[]
                for index,layer in enumerate(reference):
                    x=layer(x)
                    if index in [3,5,7]:
                        norm=getattr(model.backbone,'norm'+str((index-1)//2))
                        ref.append(norm(x).permute(0,3,1,2).contiguous())
                ref=model.fpn(ref)
                for layer in model.fusion:ref=layer(ref)
                b=torch.as_tensor(boxes,dtype=torch.float32)
                left=torch.nn.functional.normalize(model.embed_boxes(own,b,scale),dim=1)
                right=torch.nn.functional.normalize(model.embed_boxes(ref,b,scale),dim=1)
                vector_difference=float((left-right).abs().max())
                cosine_difference=float((left@left.T-right@right.T).abs().max())
            item=dict(frame=frame,rois=len(boxes),shape=list(tensor.shape),seconds=time.monotonic()-began,
                      maximum_normalized_vector_difference=vector_difference,maximum_cosine_difference=cosine_difference)
            checked.append(item);service.write(folder/'partial.json',dict(checked=checked,scientific_verdict=None))
            assert vector_difference<=p['normalized_vector_tolerance'],item
            assert cosine_difference<=p['cosine_tolerance'],item
    finally:cap.release()
    assert all(torch.equal(before[k],v) for k,v in model.named_buffers())
    assert not torch.cuda.is_initialized()
    service.write(folder/'report.json',dict(passed=True,complete=True,checked=checked,gt_read=False,
        old_raw_feature_gate_passed=False,normalized_output_interface_passed=True,
        scope='Independent Swin feature implementation with shared previously reviewed FPN/fusion/ROI/head; functional interface only, not HOTA',
        scientific_verdict=None,updated=service.now()))

if __name__=='__main__':main()
