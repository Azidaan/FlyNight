"""Н11: сигнатуры состояния фовеального глаза FlyVis для повторного опознания объекта (карточка fly_signature_reid_v1).

Трекер стенда (XYWHByteTrack) и глаз на каждый подтверждённый трек (EyeBank, сетчатка-ламина),
как в ступени B. Для каждой «жизни» глаза (от рождения до потери трека) сохраняются средние за
первые и последние K шагов:

* муха: активность 34 выходных типов, пулированная по 19 макроколонкам гексагональной решётки
  (k-means по координатам колонок, фиксированный сид) — 34 × 19;
* вход глаза (после сетчатки), 721 фасетка — контроль «что муха видит»;
* уменьшенная серая картинка рамки 16 × 16 и гистограмма HSV (8 × 4 оттенок-насыщенность + 8 яркость)
  — контроли внешности.

Плюс все выходы трекера по кадрам (для сшивки и HOTA). Разметку сборщик не читает.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.freeze_bench import XYWH_CONFIG  # noqa: E402
from scripts.run_bdd_video_rate_experiment import detections_for_frame, load_detection_cache, make_motion_estimator, save_json  # noqa: E402
from tracking.foveated_eyes import EyeBank  # noqa: E402
from tracking.offline_association import _DetSlice, make_tracker_args  # noqa: E402
from tracking.retina import make_retina  # noqa: E402

K = 5
MACRO = 19


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-id", action="append", default=[])
    parser.add_argument("--videos", type=Path, default=PROJECT_ROOT / "data" / "bdd100k" / "videos" / "train")
    parser.add_argument("--detections-root", type=Path, default=PROJECT_ROOT / "runs" / "train_cache")
    parser.add_argument("--flow-root", type=Path, default=PROJECT_ROOT / "runs" / "train_cache")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "runs" / "fly_signatures")
    parser.add_argument("--retina", default="lamina")
    parser.add_argument("--retina-downscale", type=int, default=4)
    parser.add_argument("--max-eyes", type=int, default=16)
    # совместимость с run_fovea_jobs.py (эти флаги ему передаются всегда)
    parser.add_argument("--decode-flow", action="store_true")
    parser.add_argument("--gpu-mem-fraction", type=float, default=0.0)
    return parser.parse_args()


def macro_cells(points: np.ndarray) -> np.ndarray:
    from sklearn.cluster import KMeans

    return KMeans(n_clusters=MACRO, n_init=10, random_state=0).fit_predict(points.astype(np.float64))


def appearance(gray: np.ndarray, hsv: np.ndarray, box: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = gray.shape
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    x1, y1 = max(0, min(x1, w - 2)), max(0, min(y1, h - 2)); x2, y2 = max(x1 + 2, min(x2, w)), max(y1 + 2, min(y2, h))
    thumb = cv2.resize(gray[y1:y2, x1:x2], (16, 16), interpolation=cv2.INTER_AREA).astype(np.float32).ravel() / 255.0
    patch = hsv[y1:y2, x1:x2]
    hs = cv2.calcHist([patch], [0, 1], None, [8, 4], [0, 180, 0, 256]).ravel()
    v = cv2.calcHist([patch], [2], None, [8], [0, 256]).ravel()
    hist = np.concatenate([hs / max(hs.sum(), 1.0), v / max(v.sum(), 1.0)]).astype(np.float32)
    return thumb, hist


class Segment:
    def __init__(self, tid: int, start: int, cls: int) -> None:
        self.tid, self.start, self.cls = tid, start, cls
        self.first: list[tuple] = []
        self.last: deque = deque(maxlen=K)
        self.end = start
        self.first_box = self.last_box = None

    def add(self, frame: int, box: np.ndarray, item: tuple) -> None:
        if len(self.first) < K:
            self.first.append(item)
            if self.first_box is None:
                self.first_box = box.copy()
        self.last.append(item)
        self.last_box = box.copy(); self.end = frame


def mean_parts(items) -> list[np.ndarray]:
    return [np.mean([it[k] for it in items], axis=0).astype(np.float32) for k in range(4)]


def build_video(video_id: str, args: argparse.Namespace) -> dict:
    from tracking.xywh_bytetrack import XYWHByteTrack
    from ultralytics.trackers.basetrack import BaseTrack

    out_path = args.output / video_id / "signatures.npz"
    if out_path.exists():
        print(f"[CACHE] {video_id}", flush=True)
        return {}
    detections = load_detection_cache(args.detections_root / video_id / "detections" / "raw_video.npz")
    fps = float(json.loads((args.flow_root / video_id / "motion" / "flyvis_tiled.json").read_text(encoding="utf-8"))["fps"])
    estimator = make_motion_estimator("flyvis_tiled", fps, 0.02)
    threads = int(os.environ.get("FOVEA_THREADS", "0"))
    if threads > 0:
        estimator.torch.set_num_threads(threads); cv2.setNumThreads(threads)
    bank = EyeBank(estimator, max_batch=args.max_eyes)
    retina = make_retina(args.retina, fps, blur_downscale=args.retina_downscale)
    cells = macro_cells(bank.points_local)
    onehot = np.zeros((721, MACRO), np.float32); onehot[np.arange(721), cells] = 1.0; onehot /= onehot.sum(axis=0, keepdims=True)
    BaseTrack.reset_id()
    tracker = XYWHByteTrack(make_tracker_args(**XYWH_CONFIG))
    capture = cv2.VideoCapture(str(args.videos / f"{video_id}.mov"))
    segments: dict[int, Segment] = {}
    done: list[Segment] = []
    outputs: list[np.ndarray] = []
    prev_tracked: list = []
    t0 = time.perf_counter(); frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY); hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        eye_frame = retina.process(gray)
        xyxy, conf, classes = detections_for_frame(detections, frame_index)
        predictions = {int(t.track_id): tuple((t.mean[:4] + t.mean[4:8]).tolist()) for t in prev_tracked if int(t.track_id) in bank.track_ids}
        activity = bank.step(eye_frame, predictions)
        for tid, act in activity.items():
            cx, cy, w, h = predictions[tid]
            box = np.asarray([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], np.float32)
            fly = (act @ onehot).astype(np.float32)  # (34, 19)
            hexin = bank._last_hex[tid].detach().reshape(-1).cpu().numpy().astype(np.float32)  # (721,)
            thumb, hist = appearance(gray, hsv, box)
            segments[tid].add(frame_index, box, (fly, hexin, thumb, hist))
        output = tracker.update(_DetSlice(xyxy, conf, classes))
        if len(output):
            outputs.append(np.column_stack([np.full(len(output), frame_index), np.asarray(output)[:, :7]]).astype(np.float32))
        tracked = list(tracker.tracked_stracks)
        current = {int(t.track_id) for t in tracked}
        births = {int(t.track_id): tuple(t.mean[:4].tolist()) for t in tracked if int(t.track_id) not in bank.track_ids}
        bank.birth(eye_frame, births)
        for t in tracked:
            tid = int(t.track_id)
            if tid in births:
                if tid in segments and segments[tid].first:
                    done.append(segments[tid])
                segments[tid] = Segment(tid, frame_index, int(t.cls))
        lost = set(bank.track_ids) - current
        for tid in lost:
            if tid in segments and segments[tid].first:
                done.append(segments.pop(tid))
            else:
                segments.pop(tid, None)
        bank.remove(lost)
        prev_tracked = tracked
        frame_index += 1
        if frame_index % 100 == 0 and estimator.torch.cuda.is_available():
            estimator.torch.cuda.empty_cache()  # кэш PyTorch растёт до ~1,5 ГБ на процесс; 7 процессов забивали 12 ГБ и драйвер выгружал память
        if frame_index % 25 == 0:
            print(f"[{video_id}] {frame_index} кадров, глаз {len(bank.track_ids)}, жизней {len(done)}", flush=True)
    capture.release()
    done += [s for s in segments.values() if s.first]
    if not done:
        raise RuntimeError(f"{video_id}: ни одной жизни глаза")
    firsts = [mean_parts(s.first) for s in done]; lasts = [mean_parts(list(s.last)) for s in done]
    args.output.joinpath(video_id).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        track_id=np.asarray([s.tid for s in done], np.int32), start=np.asarray([s.start for s in done], np.int32), end=np.asarray([s.end for s in done], np.int32),
        cls=np.asarray([s.cls for s in done], np.int16), first_box=np.stack([s.first_box for s in done]), last_box=np.stack([s.last_box for s in done]),
        n_first=np.asarray([len(s.first) for s in done], np.int16), n_last=np.asarray([len(s.last) for s in done], np.int16),
        **{f"first_{n}": np.stack([f[k] for f in firsts]).astype(np.float16) for k, n in enumerate(("fly", "hexin", "thumb", "hist"))},
        **{f"last_{n}": np.stack([f[k] for f in lasts]).astype(np.float16) for k, n in enumerate(("fly", "hexin", "thumb", "hist"))},
        outputs=np.concatenate(outputs) if outputs else np.zeros((0, 8), np.float32), frame_count=np.asarray(frame_index))
    save_json(args.output / video_id / "signatures.json", {"video_id": video_id, "frames": frame_index, "lives": len(done), "retina": retina.describe(), "macro_cells": MACRO, "K": K, "seconds": time.perf_counter() - t0})
    print(f"[OK] {video_id}: {frame_index} кадров, {len(done)} жизней глаз, {1000 * (time.perf_counter() - t0) / max(frame_index, 1):.0f} мс/кадр", flush=True)
    return {}


def main() -> int:
    args = parse_args()
    for v in args.video_id:
        build_video(v, args)
    print(f"[OK] {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
