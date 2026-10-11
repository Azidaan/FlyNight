"""Модернизация FlyVis: фовеальный глаз сохраняется при потере трека, остаточный поток поправляет прогноз положения
(карточка fly_persistent_eye_v1; протокол и критерии — в карточке).

Трекер стенда (XYWHByteTrack, буфер 30) на сохранённых детекциях; глаз — EyeBank (сетчатка-ламина, decode_flow) как у
сигнатур стенда. Варианты:
  carry  — глаз потерянного трека остаётся, каждый кадр смотрит на вырезку вокруг прогноза, состояние непрерывно;
  reinit — контроль: в момент потери глаз заново разгоняется на текущем кадре в вырезке прогноза (состояние не переносится).
Считывание (как flyvis_fovea_motion_v1): остаточный декодированный поток, среднее по колонкам внутри прогнозной рамки,
× сторона вырезки / размер ядра → px, ось y вниз; δ = g · среднее; подавление при < 3 колонках, |δ| ≤ 0,5 · max(w, h).
Действие: только потерянные треки, после прогноза Калмана перед первым сопоставлением центр сдвигается на δ.
Режим --fit (g = 0, траектории стенда): строки для МНК — сырое среднее (до g) и цель (центр своей уверенной детекции
минус центр прогноза; своя — по интерполированной разметке, association_log) на кадрах потерянных треков.

Запуск: python scripts/fly_persistent_eye.py --variant carry --video-id ID [--g G | --fit] --out DIR
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

FRAME = (1280, 720)


def fit_gain(rows: np.ndarray) -> dict:
    """МНК g по приращениям за кадр: поток глаза меряет движение за один кадр, а поправка накапливается в прогнозе,
    поэтому цель — изменение смещения «своя детекция − прогноз» между соседними кадрами одного потерянного трека
    (строки: кадр, трек, сырой поток x, y, смещение x, y, w, h)."""
    X, Y = [], []
    rows = rows[np.lexsort((rows[:, 0], rows[:, 1]))]
    for a_, b_ in zip(rows[:-1], rows[1:]):
        if a_[1] == b_[1] and b_[0] == a_[0] + 1:
            X.append(b_[2:4]); Y.append(b_[4:6] - a_[4:6])
    X, Y = np.asarray(X), np.asarray(Y)
    if not len(X):
        return {"g": 0.0, "n": 0}
    g = float((X * Y).sum() / max((X * X).sum(), 1e-12))
    return {"g": g, "n": len(X), "corr_x": float(np.corrcoef(X[:, 0], Y[:, 0])[0, 1]), "corr_y": float(np.corrcoef(X[:, 1], Y[:, 1])[0, 1]),
            "median_inc": float(np.median(np.linalg.norm(Y, axis=1))), "median_inc_after_g": float(np.median(np.linalg.norm(Y - g * X, axis=1))), "median_raw": float(np.median(np.linalg.norm(X, axis=1)))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=("carry", "reinit"), required=True)
    ap.add_argument("--video-id", required=True)
    ap.add_argument("--g", type=float, default=0.0)
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--split", default="train"); ap.add_argument("--root", type=Path, default=PROJECT_ROOT / "runs/train_cache")
    ap.add_argument("--max-eyes", type=int, default=16); ap.add_argument("--gpu-mem-fraction", type=float, default=0.0)
    a = ap.parse_args()
    from ultralytics.trackers.basetrack import BaseTrack, TrackState
    from datasets.bdd100k_loader import CLASS_NAMES
    from scripts.association_log import gt_index, label
    from scripts.freeze_bench import XYWH_CONFIG
    from scripts.output_filter_oracle import VALID, _iou
    from scripts.run_bdd_video_rate_experiment import detections_for_frame, load_detection_cache, make_motion_estimator, save_json, scalabel_frame
    from tracking.foveated_eyes import EyeBank, box_to_crop_frame, crop_box
    from tracking.offline_association import _DetSlice, make_tracker_args
    from tracking.retina import make_retina
    from tracking.xywh_bytetrack import XYWHByteTrack
    vid = a.video_id; t0 = time.perf_counter()
    det = load_detection_cache(a.root / vid / "detections" / "raw_video.npz")
    fps = float(json.loads((a.root / vid / "motion" / "flyvis_tiled.json").read_text(encoding="utf-8"))["fps"])
    est = make_motion_estimator("flyvis_tiled", fps, 0.02)
    if a.gpu_mem_fraction > 0 and est.torch.cuda.is_available():
        est.torch.cuda.set_per_process_memory_fraction(a.gpu_mem_fraction)
    bank = EyeBank(est, max_batch=a.max_eyes, decode_flow=True)
    retina = make_retina("lamina", fps, blur_downscale=4)
    G = None
    if a.fit:
        matches = json.loads((a.root / vid / "alignment.json").read_text(encoding="utf-8"))["matches"]
        truth = json.loads((PROJECT_ROOT / "data/bdd100k/labels/box_track_20" / a.split / f"{vid}.json").read_text(encoding="utf-8"))
        G = gt_index(truth, [int(m["video_frame_index"]) for m in matches], VALID)
    delta: dict[int, np.ndarray] = {}
    stats = {"frames": 0, "eye_steps_lost": 0, "applied": 0, "suppressed": 0, "reinit": 0, "guard_tracked": 0, "abs_shift_px": 0.0, "capped": 0}
    fit_rows = []

    class BT(XYWHByteTrack):
        def _pre_first_associate(self, pool, unconfirmed, img, results_high):
            super()._pre_first_associate(pool, unconfirmed, img, results_high)
            for t in pool:
                if int(t.track_id) in delta:
                    if t.state == TrackState.Tracked:   # страж: поправка — только потерянным трекам
                        stats["guard_tracked"] += 1; continue
                    t.mean[0] += float(delta[int(t.track_id)][0]); t.mean[1] += float(delta[int(t.track_id)][1]); stats["applied"] += 1
                    stats["abs_shift_px"] += float(np.linalg.norm(delta[int(t.track_id)]))

    BaseTrack.reset_id()
    tr = BT(make_tracker_args(**XYWH_CONFIG))
    cap = cv2.VideoCapture(str(PROJECT_ROOT / "data/bdd100k/videos" / a.split / f"{vid}.mov"))
    by_frame, prev_tracked, own = {}, [], {}
    f = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        eye_frame = retina.process(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        xyxy, conf, cls = detections_for_frame(det, f)
        lab_f = None
        if a.fit and G is not None and len(xyxy) and G.get(f):
            lab_f = label(list(xyxy), [CLASS_NAMES[int(c)] for c in cls], G[f])   # разметка детекций кадра — один раз (только подбор g)
        lost_ids = {int(t.track_id) for t in tr.lost_stracks}
        pool = [t for t in prev_tracked] + [t for t in tr.lost_stracks]
        # прогноз глаза = прогноз трекера на этот кадр: BOTrack.multi_predict обнуляет скорости ширины и высоты у нетрекаемых треков
        def pred(t):
            v = t.mean[4:8].copy()
            if int(t.track_id) in lost_ids:
                v[2:] = 0.0
            return tuple((t.mean[:4] + v).tolist())
        preds = {int(t.track_id): pred(t) for t in pool if int(t.track_id) in bank.track_ids}
        bank.step(eye_frame, preds)
        delta.clear()
        for tid in preds:
            if tid not in lost_ids:
                continue
            stats["eye_steps_lost"] += 1
            crop = bank.crops[tid]; side = crop[2] - crop[0]
            cx, cy, w, h = preds[tid]; pb = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
            loc = box_to_crop_frame(pb, crop, bank.eye_size)
            px, py = bank.points_local[:, 0], bank.points_local[:, 1]
            ins = (px >= loc[0]) & (px <= loc[2]) & (py >= loc[1]) & (py <= loc[3])
            if ins.sum() < 3:
                stats["suppressed"] += 1; continue
            m = bank.last_flow(tid)[ins].mean(axis=0); sc = side / est.kernel_area
            raw = np.array([m[0] * sc, -m[1] * sc])
            if lab_f is not None and own.get(tid) is not None:
                x1, y1, x2, y2 = crop_box(preds[tid], FRAME)
                for j, g_ in enumerate(lab_f):
                    if g_ == own[tid] and conf[j] >= 0.25:
                        c = (np.asarray(xyxy[j][:2]) + np.asarray(xyxy[j][2:])) / 2
                        if x1 <= c[0] <= x2 and y1 <= c[1] <= y2:
                            fit_rows.append([f, tid, raw[0], raw[1], c[0] - cx, c[1] - cy, w, h])   # смещение «своя детекция − прогноз»; цель подбора — его приращение за кадр (fit_gain)
                        break
            d = a.g * raw; lim = 0.5 * max(w, h); nrm = float(np.linalg.norm(d))
            if nrm > lim:
                d = d * (lim / nrm); stats["capped"] += 1
            delta[tid] = d
        o = tr.update(_DetSlice(xyxy, conf, cls))
        if len(o):
            by_frame[f] = np.asarray(o, np.float64)[:, :7]
        tracked = list(tr.tracked_stracks); cur = {int(t.track_id) for t in tracked}; lost_now = {int(t.track_id) for t in tr.lost_stracks}
        if lab_f is not None:
            for t in tracked:
                if t.frame_id == tr.frame_id:
                    j = int(np.argmax(_iou([t.xyxy], np.asarray(xyxy))[0])); own[int(t.track_id)] = lab_f[j]
        births = {int(t.track_id): tuple(t.mean[:4].tolist()) for t in tracked if int(t.track_id) not in bank.track_ids}
        keep = {}
        if a.variant == "reinit":   # контроль: в момент потери — новый глаз (разгон на текущем кадре), состояние не переносится
            newly = [t for t in tr.lost_stracks if int(t.track_id) in bank.track_ids and int(t.track_id) in {int(x.track_id) for x in prev_tracked}]
            if newly:
                keep = {int(t.track_id): (bank._baseline_flow[int(t.track_id)], bank._baseline[int(t.track_id)]) for t in newly}
                bank.remove(list(keep)); stats["reinit"] += len(newly)
                births.update({int(t.track_id): tuple(t.mean[:4].tolist()) for t in newly})
        bank.birth(eye_frame, births)
        for tid, (bf, bl) in keep.items():   # точка отсчёта считывания — прежняя (от рождения трека), как у carry: отличается только состояние глаза
            bank._baseline_flow[tid] = bf; bank._baseline[tid] = bl
        bank.remove(set(bank.track_ids) - cur - lost_now)   # трек удалён (потеря > буфера) — глаз удаляется
        prev_tracked = tracked; f += 1; stats["frames"] = f
        if f % 100 == 0:
            if est.torch.cuda.is_available():
                est.torch.cuda.empty_cache()
            print(f"[{vid}] {f} кадров, глаз {len(bank.track_ids)} (потерянных {len(lost_now & set(bank.track_ids))}), {(time.perf_counter() - t0) / f * 1000:.0f} мс/кадр", flush=True)
    cap.release()
    a.out.mkdir(parents=True, exist_ok=True)
    stats["seconds"] = time.perf_counter() - t0
    if a.fit:
        np.save(a.out / f"{vid}_fit.npy", np.asarray(fit_rows, np.float64).reshape(-1, 8))
    else:
        matches = json.loads((a.root / vid / "alignment.json").read_text(encoding="utf-8"))["matches"]
        truth = json.loads((PROJECT_ROOT / "data/bdd100k/labels/box_track_20" / a.split / f"{vid}.json").read_text(encoding="utf-8"))
        rows = [scalabel_frame(vid, truth[i], by_frame.get(int(m["video_frame_index"]), np.zeros((0, 7)))) for i, m in enumerate(matches)]
        save_json(a.out / f"{vid}.json", rows)
    save_json(a.out / f"{vid}_stats.json", stats)
    print(f"[OK] {vid} {a.variant}: {stats}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
