"""MASA-R50 appearance embeddings for every real region of every z13 query and for each episode exemplar.

Base branch of the hybrid identity readout (z13_hybrid_identity_v1). Region boxes are the same real
detections that the FlyVis readout uses (tracking/z13_region_assignment.regions, choices checked
against the saved label rows). The exemplar box is the observed track box at the exemplar frame,
recovered from the saved observed eye mask (ex_mask) and crop; no ground truth is read.
Output per video: runs/z13_masa_regions_20261026/videos/<id>.npz (uid, region[N,K,256], valid[N,K], exemplar[N,256]).
"""
from pathlib import Path
from datetime import datetime
import os
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.z13_candidate_reserve_20261021 import read, write, sha, now

RUN = ROOT / 'runs/z13_masa_regions_20261026'
BASE = ROOT / 'runs/z13_region_assignment_20261025'
VIDEOS = ROOT / 'data/bdd100k/videos/train'
DEADLINE = datetime.fromisoformat('2026-10-27T07:30:00+03:00')


def status(stage, **kw):
    if (RUN / 'stop.request').exists() or now() >= DEADLINE: raise TimeoutError('Stop request or author weekday deadline')
    write(RUN / 'status.json', dict(state='running', stage=stage, updated=now().isoformat(), **kw))


def plan_video(v, queries, labelrows, geom):
    """frame -> list of (uid or -1-episode, slot, box); exemplar uses slot -1."""
    from tracking.z13_region_assignment import regions
    from tracking.z13_readout import to_frame
    need = {}; uids = []; widths = {}
    with np.load(ROOT / f'runs/train_cache/{v}/detections/raw_video.npz') as z:
        det = {k: z[k] for k in ('xyxy', 'conf', 'cls', 'frame_idx')}
    with np.load(ROOT / f'runs/z13_data/{v}.npz') as z:
        off, exm, frames, crops = z['ep_offset'], z['ex_mask'], z['fr_frame'], z['fr_crop']
    for q in queries[v]:
        l = q['label']; ix = (det['frame_idx'] == q['frame']) & (det['conf'] >= .25); row = labelrows[q['uid']]
        choices, masks, _ = regions(geom, l['crop'], l['pred'], det['xyxy'][ix], det['conf'][ix], det['cls'][ix], row['regions']['track_class'])
        assert choices == row['regions']['choices'][1:]
        boxes = det['xyxy'][ix][choices] if choices else np.zeros((0, 4))
        uids.append(q['uid']); widths[q['uid']] = len(choices)
        for k, b in enumerate(boxes): need.setdefault(int(q['frame']), []).append((q['uid'], k, b.astype(np.float32)))
        e = q['episode_local']; kx = int(off[e]) + 3; pts = geom.points[exm[e]]
        lo, hi = pts.min(0) - geom.spacing / 2, pts.max(0) + geom.spacing / 2
        a = to_frame(lo, crops[kx], geom.eye_size); b = to_frame(hi, crops[kx], geom.eye_size)
        need.setdefault(int(frames[kx]), []).append((q['uid'], -1, np.asarray([a[0], a[1], b[0], b[1]], np.float32)))
    return need, uids, widths


def run():
    import cv2
    import torch
    from tracking.masa_r50 import build, preprocess
    from scripts.z13_region_assignment_recovery2_20261025 import namespace
    p = read(RUN / 'protocol.json')
    for k, h in p['sha256'].items(): assert sha(ROOT / k) == h, k
    ns = namespace(); queries, _ = ns['queries_and_crops'](); geom = ns['geometry']()
    videos = p['videos']; model = build(); (RUN / 'videos').mkdir(exist_ok=True); began = time.monotonic()
    for i, v in enumerate(videos):
        out = RUN / 'videos' / (v + '.npz')
        if out.exists(): continue
        status('embeddings', video=v, done=i, total=len(videos), seconds=time.monotonic() - began)
        labelrows = {r['uid']: r for r in read(BASE / 'labels/videos' / (v + '.json'))['rows']}
        need, uids, widths = plan_video(v, queries, labelrows, geom)
        K = max(max(widths.values()), 1); index = {u: n for n, u in enumerate(uids)}
        region = np.zeros((len(uids), K, 256), np.float16); valid = np.zeros((len(uids), K), bool); exemplar = np.zeros((len(uids), 256), np.float16)
        cap = cv2.VideoCapture(str(VIDEOS / f'{v}.mov')); f = 0; last = max(need) if need else -1; got = set()
        with torch.no_grad():
            while f <= last:
                ok, fr = cap.read()
                if not ok: break
                if f in need:
                    items = need[f]; x, s = preprocess(fr)
                    boxes = torch.tensor(np.stack([b for _, _, b in items]), device='cuda')
                    emb = torch.nn.functional.normalize(model.embed_boxes(model.features(x), boxes, s), dim=1).cpu().numpy()
                    for (u, k, _), ee in zip(items, emb):
                        n = index[u]
                        if k < 0: exemplar[n] = ee
                        else: region[n, k] = ee; valid[n, k] = True
                    got.add(f)
                f += 1
        cap.release()
        assert got == set(need), (v, len(set(need) - got))
        np.savez_compressed(out, uid=np.asarray(uids, np.int64), region=region, valid=valid, exemplar=exemplar)
    write(RUN / 'report.json', dict(complete=True, videos=len(videos), seconds=time.monotonic() - began, ground_truth_read=False))
    write(RUN / 'status.json', dict(state='completed', stage='done', updated=now().isoformat()))


def prepare():
    assert not (RUN / 'protocol.json').exists(); RUN.mkdir(parents=True, exist_ok=True)
    p = read(BASE / 'recovery2/protocol.json')
    sources = [Path(__file__), ROOT / 'tracking/masa_r50.py', ROOT / 'tracking/z13_region_assignment.py', ROOT / 'tracking/z13_readout.py']
    write(RUN / 'protocol.json', dict(id='z13_masa_regions_v1', created=now().isoformat(), timezone='Europe/Moscow', device='cuda', deadline=DEADLINE.isoformat(),
                                      videos=p['fit84'] + p['fit20'] + p['evaluate26'], weights='tracking/masa_r50.py default MASA-R50 weights',
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
