"""CPU check of corrected official BDD input; preserve previous 1024 check."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import sys, json, hashlib, time
from pathlib import Path
from datetime import datetime
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def main():
    import cv2, torch
    from tracking.masa_detic_prepared import build, EXPECTED
    from tracking.masa_detic_bdd_input import preprocess_bdd
    from scripts.run_bdd_video_rate_experiment import load_detection_cache, detections_for_frame
    torch.set_default_device('cpu'); torch.set_num_threads(1); cv2.setNumThreads(1)
    assert time.time() < datetime.fromisoformat('2026-10-29T07:30:00+03:00').timestamp()
    out = ROOT / 'runs/continuation_20261028/reid_candidates_preparation'
    source = '00e11419-e4eeba84'
    alignment = json.loads((ROOT/f'runs/train_cache/{source}/alignment.json').read_text(encoding='utf-8'))
    frame = int(alignment['matches'][0]['video_frame_index'])
    cap = cv2.VideoCapture(str(ROOT/f'data/bdd100k/videos/train/{source}.mov'))
    try:
        for _ in range(frame+1):
            ok, image = cap.read(); assert ok
    finally:
        cap.release()
    cache = load_detection_cache(ROOT/f'runs/train_cache/{source}/detections/raw_video.npz')
    boxes, scores, _ = detections_for_frame(cache, frame)
    boxes = boxes[scores >= .1]
    model = build('cpu')
    before = {k: v.clone() for k,v in model.named_buffers()}
    tensor, scale = preprocess_bdd(image)
    started = time.monotonic()
    with torch.inference_mode():
        embedding = model.embed_boxes(model.features(tensor), torch.as_tensor(boxes, dtype=torch.float32), scale)
    assert embedding.shape == (len(boxes), 256) and torch.isfinite(embedding).all()
    assert all(torch.equal(before[k],v) for k,v in model.named_buffers())
    assert not torch.cuda.is_initialized()
    report = dict(passed=True, video=source, frame=frame, rois=len(boxes),
                  input_shape=list(tensor.shape), coordinate_scale=scale.tolist(),
                  embedding_shape=list(embedding.shape), seconds=time.monotonic()-started,
                  checkpoint_sha256=EXPECTED, gt_read=False, cuda_initialized=False,
                  previous_check_preserved='real_input_check.json',
                  issue='Previous technical check used shared R50 1024 resize; official BDD config uses 1333x800 bound',
                  full_official_feature_parity=False, hota_measured=False, scientific_verdict=None,
                  source_sha256={p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in [
                      'tracking/masa_detic_bdd_input.py',
                      'runs/continuation_20261028/reid_candidates_preparation/masa/reference_bdd_dataset.py']})
    target=out/'bdd_input_check.json'
    assert not target.exists()
    target.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False))

if __name__ == '__main__': main()
