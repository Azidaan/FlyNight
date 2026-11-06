"""Контроль «тот же вход для DIS» (карточка dis_retina_control_v1).

1. DIS (тот же пресет, шаг сетки и разрешение, что dis_quarter) на кадрах после сетчатки-ламины.
   Сетчатка применяется к кадру в разрешении DIS (×1/4), окна пересчитаны (σ 24→6, 12→3 px),
   сетка точек и векторы приводятся к полному разрешению — сетка совпадает с dis_quarter.
2. Признак «рассогласование DIS» в строках фовеального потока (ламина) пересчитывается одной и той
   же функцией из исходного DIS и из DIS-на-сетчатке: dis_box(t) = перенос out_box(t−1) того же трека.
   Строки без out_box(t−1) получают NaN в обоих вариантах (одинаковое правило).
Выход: runs/dis_retina/{orig,lamina}/<id>/fovea_flow.npz и runs/dis_retina/motion/<id>/dis_quarter.npz.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

VAL7 = ["b1c66a42-6f7d68ca", "b1c81faa-3df17267", "b1cac6a7-04e33135", "b1cebfb7-284f5117", "b232c7c9-d251d9ee", "b1d22449-15fb948f", "b251064f-e7a165fd"]
OUT = PROJECT_ROOT / "runs" / "dis_retina"


def dis_on_retina(task: tuple) -> str:
    video_id, video_path = task
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import cv2
    from tracking.classical_motion import OpenCVMotionEstimator
    from tracking.retina import make_retina

    cv2.setNumThreads(1)
    path = OUT / "motion" / video_id / "dis_quarter.npz"
    if path.exists():
        return video_id
    cap = cv2.VideoCapture(str(video_path)); fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    retina = make_retina("lamina", fps, sigma_adapt=6.0, sigma_lamina=3.0)
    est = OpenCVMotionEstimator("dis", processing_scale=1.0, sample_step=8)
    vectors, points = [], None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
        field = est.observe(np.clip(retina.process(small), 0, 255).astype(np.uint8))
        if points is None:
            points = field.points_xy.copy() * 4.0
        vectors.append(field.vectors_xy.copy() * 4.0)
    cap.release()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, points=points.astype(np.float32), vectors=np.stack(vectors).astype(np.float32))
    return video_id


def rewrite(task: tuple) -> str:
    video_id, flow_npz, orig_dis = task
    from scripts.evaluate_flyvis_qmatch_gate import propagate_box
    from scripts.run_bdd_video_rate_experiment import load_motion_cache

    z = dict(np.load(flow_npz))
    key = {(int(f), int(t)): i for i, (f, t) in enumerate(zip(z["frame"], z["track_id"]))}
    sources = {"orig": load_motion_cache(Path(orig_dis), 1.0), "lamina": load_motion_cache(OUT / "motion" / video_id / "dis_quarter.npz", 1.0)}
    for name, (pts, vec) in sources.items():
        dis = np.full_like(z["dis_box"], np.nan)
        for i, (f, t) in enumerate(zip(z["frame"], z["track_id"])):
            j = key.get((int(f) - 1, int(t)))
            if j is None or f == 0:
                continue
            prev = z["out_box"][j].astype(np.float64)
            if not np.isfinite(prev).all() or prev[2] - prev[0] < 1 or prev[3] - prev[1] < 1:
                continue
            dis[i] = propagate_box(prev, pts, vec, int(f) - 1, int(f))[0]
        out = OUT / name / video_id / "fovea_flow.npz"
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, **{**z, "dis_box": dis.astype(np.float32)})
    return video_id


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=14)
    args = parser.parse_args()
    sel = json.loads((PROJECT_ROOT / "runs/train_cache/selection.json").read_text(encoding="utf-8"))["videos"]
    train_ids = [v for s in ("train", "dev", "test") for v in sel[s]]
    t1 = [(v, PROJECT_ROOT / "data/bdd100k/videos/train" / f"{v}.mov") for v in train_ids] + [(v, PROJECT_ROOT / "data/bdd100k/videos/val" / f"{v}.mov") for v in VAL7]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for k, _ in enumerate(pool.map(dis_on_retina, t1), 1):
            if k % 25 == 0:
                print(f"[DIS] {k}/{len(t1)}", flush=True)
    t2 = [(v, PROJECT_ROOT / "runs/train_cache_retina_lamina4" / v / "fovea_flow.npz", PROJECT_ROOT / "runs/train_cache" / v / "motion/dis_quarter.npz") for v in train_ids]
    t2 += [(v, PROJECT_ROOT / "runs/flyvis_fovea_flow_retina_lamina4" / v / "fovea_flow.npz", PROJECT_ROOT / "runs/video_rate" / v / "motion/dis_quarter.npz") for v in VAL7]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for k, _ in enumerate(pool.map(rewrite, t2), 1):
            if k % 25 == 0:
                print(f"[FEAT] {k}/{len(t2)}", flush=True)
    print(f"[OK] {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
