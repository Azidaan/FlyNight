"""MASA exemplar embedding snapped to the observed detection at the exemplar frame (z13_hybrid_identity_v2 input).

The exemplar box of z13_masa_regions_20261026 is reconstructed from the hex eye mask, so it is coarse. Here the box is
replaced by the raw detection (conf >= 0.25) with the highest IoU to that mask box at the same frame when IoU >= 0.3;
otherwise the mask box is kept (flag stored). Detections are observable tracker inputs; no ground truth is read.
Output per video: runs/z13_masa_exemplar_snap_20261026/videos/<id>.npz (uid, exemplar[N,256], snapped[N], iou[N]).
"""
from pathlib import Path
from datetime import datetime
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.z13_candidate_reserve_20261021 import read, write, sha, now

RUN = ROOT / 'runs/z13_masa_exemplar_snap_20261026'
VIDEOS = ROOT / 'data/bdd100k/videos/train'
DEADLINE = datetime.fromisoformat('2026-10-27T07:30:00+03:00')


def status(stage, **kw):
    if (RUN / 'stop.request').exists() or now() >= DEADLINE: raise TimeoutError('Stop request or author weekday deadline')
    write(RUN / 'status.json', dict(state='running', stage=stage, updated=now().isoformat(), **kw))


def iou(a, b):
    lt = np.maximum(a[:2], b[:, :2]); rb = np.minimum(a[2:], b[:, 2:]); wh = np.clip(rb - lt, 0, None); inter = wh[:, 0] * wh[:, 1]
    return inter / np.maximum((a[2] - a[0]) * (a[3] - a[1]) + (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) - inter, 1e-9)


def plan_video(v, queries, geom):
    from tracking.z13_readout import to_frame
    with np.load(ROOT / f'runs/train_cache/{v}/detections/raw_video.npz') as z:
        det = {k: z[k] for k in ('xyxy', 'conf', 'frame_idx')}
    with np.load(ROOT / f'runs/z13_data/{v}.npz') as z:
        off, exm, frames, crops = z['ep_offset'], z['ex_mask'], z['fr_frame'], z['fr_crop']
    need = {}; uids = []; info = {}
    for q in queries[v]:
        e = q['episode_local']; kx = int(off[e]) + 3; pts = geom.points[exm[e]]; f = int(frames[kx])
        lo, hi = pts.min(0) - geom.spacing / 2, pts.max(0) + geom.spacing / 2
        a = to_frame(lo, crops[kx], geom.eye_size); b = to_frame(hi, crops[kx], geom.eye_size); box = np.asarray([a[0], a[1], b[0], b[1]], np.float32)
        ix = (det['frame_idx'] == f) & (det['conf'] >= .25); cand = det['xyxy'][ix]; best = 0.
        if len(cand):
            o = iou(box, cand); j = int(o.argmax()); best = float(o[j])
            if best >= .3: box = cand[j].astype(np.float32)
        uids.append(q['uid']); info[q['uid']] = (best >= .3, best); need.setdefault(f, []).append((q['uid'], box))
    return need, uids, info


def run():
    import cv2
    import torch
    from tracking.masa_r50 import build, preprocess
    from scripts.z13_region_assignment_recovery2_20261025 import namespace
    p = read(RUN / 'protocol.json')
    for k, h in p['sha256'].items(): assert sha(ROOT / k) == h, k
    ns = namespace(); queries, _ = ns['queries_and_crops'](); geom = ns['geometry']()
    model = build(); (RUN / 'videos').mkdir(exist_ok=True); began = time.monotonic()
    for i, v in enumerate(p['videos']):
        out = RUN / 'videos' / (v + '.npz')
        if out.exists(): continue
        status('embeddings', video=v, done=i, total=len(p['videos']), seconds=time.monotonic() - began)
        need, uids, info = plan_video(v, queries, geom); index = {u: n for n, u in enumerate(uids)}
        ex = np.zeros((len(uids), 256), np.float16); got = set()
        cap = cv2.VideoCapture(str(VIDEOS / f'{v}.mov')); f = 0; last = max(need) if need else -1
        with torch.no_grad():
            while f <= last:
                ok, fr = cap.read()
                if not ok: break
                if f in need:
                    items = need[f]; x, s = preprocess(fr)
                    emb = torch.nn.functional.normalize(model.embed_boxes(model.features(x), torch.tensor(np.stack([b for _, b in items]), device='cuda'), s), dim=1).cpu().numpy()
                    for (u, _), ee in zip(items, emb): ex[index[u]] = ee
                    got.add(f)
                f += 1
        cap.release(); assert got == set(need), (v, len(set(need) - got))
        np.savez_compressed(out, uid=np.asarray(uids, np.int64), exemplar=ex, snapped=np.asarray([info[u][0] for u in uids]), iou=np.asarray([info[u][1] for u in uids], np.float32))
    write(RUN / 'report.json', dict(complete=True, videos=len(p['videos']), seconds=time.monotonic() - began, ground_truth_read=False))
    write(RUN / 'status.json', dict(state='completed', stage='done', updated=now().isoformat()))


def prepare():
    assert not (RUN / 'protocol.json').exists(); RUN.mkdir(parents=True, exist_ok=True)
    p = read(ROOT / 'runs/z13_region_assignment_20261025/recovery2/protocol.json')
    sources = [Path(__file__), ROOT / 'tracking/masa_r50.py', ROOT / 'tracking/z13_readout.py']
    write(RUN / 'protocol.json', dict(id='z13_masa_exemplar_snap_v1', created=now().isoformat(), timezone='Europe/Moscow', device='cuda', deadline=DEADLINE.isoformat(),
                                      videos=p['fit84'] + p['fit20'] + p['evaluate26'], snap='highest-IoU raw detection (conf>=0.25) at the exemplar frame if IoU>=0.3, else mask box',
                                      sha256={q.relative_to(ROOT).as_posix(): sha(q) for q in sources}))
    write(RUN / 'status.json', dict(state='ready', stage='protocol', updated=now().isoformat()))


if __name__ == '__main__':
    if sys.argv[1] == '--prepare': prepare()
    elif sys.argv[1] == '--run':
        from tracking.gpu_lock import GpuLock
        try:
            with GpuLock('flyvis_fovea_gpu'): run()
        except Exception as error:
            write(RUN / 'failure.json', dict(at=now().isoformat(), error=repr(error))); raise
