"""One existing TRAIN image, CPU-only interface check; never reads GT."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
from datetime import datetime
import sys,time,json,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def main():
    import cv2,torch
    from tracking.masa_detic_prepared import build,preprocess,EXPECTED
    from scripts.run_bdd_video_rate_experiment import load_detection_cache,detections_for_frame
    torch.set_default_device('cpu');torch.set_num_threads(1);cv2.setNumThreads(1)
    assert time.time()<datetime.fromisoformat('2026-10-29T07:30:00+03:00').timestamp()
    out=ROOT/'runs/continuation_20261028/reid_candidates_preparation'
    source='00e11419-e4eeba84'
    alignment=json.loads((ROOT/f'runs/train_cache/{source}/alignment.json').read_text(encoding='utf-8'))
    frame=int(alignment['matches'][0]['video_frame_index'])
    capture=cv2.VideoCapture(str(ROOT/f'data/bdd100k/videos/train/{source}.mov'))
    try:
        for _ in range(frame+1):ok,image=capture.read();assert ok
    finally:capture.release()
    cache=load_detection_cache(ROOT/f'runs/train_cache/{source}/detections/raw_video.npz')
    boxes,scores,classes=detections_for_frame(cache,frame)
    selected=boxes[scores>=.1]
    assert len(selected)>0
    model=build('cpu');before={k:v.clone() for k,v in model.named_buffers()}
    tensor,scale=preprocess(image,'cpu');started=time.monotonic()
    with torch.inference_mode():
        features=model.features(tensor)
        embeddings=model.embed_boxes(features,torch.as_tensor(selected,dtype=torch.float32),scale)
        assert embeddings.shape==(len(selected),256) and torch.isfinite(embeddings).all()
        assert (embeddings.norm(dim=1)>0).all()
    elapsed=time.monotonic()-started
    assert all(torch.equal(before[k],v) for k,v in model.named_buffers())
    assert not torch.cuda.is_initialized()
    report=dict(passed=True,video=source,split='existing train; not quality assessment',frame=frame,
        rois=len(selected),embedding_shape=list(embeddings.shape),input_shape=list(tensor.shape),
        frame_sha256=hashlib.sha256(image.tobytes()).hexdigest(),checkpoint_sha256=EXPECTED,
        cpu_seconds=elapsed,gt_read=False,cuda_initialized=False,buffers_unchanged=True,
        official_full_parity=False,hota_measured=False,scientific_verdict=None)
    (out/'real_input_check.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False))

if __name__=='__main__':main()
