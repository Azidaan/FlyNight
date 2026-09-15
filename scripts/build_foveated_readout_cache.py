"""Фовеальные считывания FlyVis: глаз на каждый трек XYWHByteTrack (карточка flyvis_foveation_readout_v1).

На кадре t глаза продвигаются к предсказаниям треков (без мутации трекера),
для каждой детекции кадра t, пересекающей вырезку глаза, считаются признаки
T4/T5 и all34 в координатах глаза; затем трекер обновляется, рождаются глаза
новых треков, удаляются глаза потерянных. Разметка не читается.
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

from scripts.evaluate_flyvis_readout_gate import _BOX_ALL34, _BOX_T4T5, READOUT_ALL34_FEATURES, READOUT_PAIR_FEATURES  # noqa: E402
from scripts.freeze_bench import XYWH_CONFIG  # noqa: E402
from scripts.run_bdd_video_rate_experiment import detections_for_frame, distribution, load_detection_cache, make_motion_estimator, save_json  # noqa: E402
from tracking.flyvis_activity import box_readout_all_types, box_readout_t4t5, column_spacing  # noqa: E402
from tracking.foveated_eyes import EyeBank, box_to_crop_frame, original_detection_index  # noqa: E402
from tracking.offline_association import _DetSlice, make_tracker_args  # noqa: E402

VIDEO_IDS = (
    "b1c66a42-6f7d68ca", "b1c81faa-3df17267", "b1cac6a7-04e33135", "b1cebfb7-284f5117",
    "b232c7c9-d251d9ee", "b1d22449-15fb948f", "b251064f-e7a165fd",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-id", action="append", default=[])
    parser.add_argument("--videos", type=Path, default=PROJECT_ROOT / "data" / "bdd100k" / "videos" / "val")
    parser.add_argument("--detections-root", type=Path, default=PROJECT_ROOT / "runs" / "video_rate")
    parser.add_argument("--flow-root", type=Path, default=PROJECT_ROOT / "runs" / "flyvis_stress_gate")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "runs" / "flyvis_fovea")
    parser.add_argument("--context", type=float, default=2.0)
    parser.add_argument("--min-side", type=int, default=64)
    parser.add_argument("--fade-in", type=float, default=1.0)
    parser.add_argument("--ring-scale", type=float, default=1.5)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def pair_terms(prev: dict, cur: dict) -> list[float]:
    a = np.asarray([prev["inside_net_x"], prev["inside_net_y"]])
    b = np.asarray([cur["inside_net_x"], cur["inside_net_y"]])
    cosine = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
    return [cosine, abs(prev["inside_entropy"] - cur["inside_entropy"]), abs(prev["inside_energy_total"] - cur["inside_energy_total"])]


def intersects(box_xyxy: np.ndarray, crop: tuple[int, int, int, int]) -> bool:
    x1, y1, x2, y2 = crop
    return bool(box_xyxy[2] > x1 and box_xyxy[0] < x2 and box_xyxy[3] > y1 and box_xyxy[1] < y2)


def build_video(video_id: str, args: argparse.Namespace) -> dict:
    from tracking.xywh_bytetrack import XYWHByteTrack

    output = args.output / video_id
    data_path, meta_path = output / "fovea_readout.npz", output / "fovea_readout.json"
    if data_path.exists() and meta_path.exists() and not args.force:
        print(f"[CACHE] {video_id}: {data_path}", flush=True)
        return json.loads(meta_path.read_text(encoding="utf-8"))

    detections = load_detection_cache(args.detections_root / video_id / "detections" / "raw_video.npz")
    fps = float(json.loads((args.flow_root / video_id / "motion" / "flyvis_tiled.json").read_text(encoding="utf-8"))["fps"])
    estimator = make_motion_estimator("flyvis_tiled", fps, 0.02)
    bank = EyeBank(estimator, context=args.context, min_side=args.min_side, fade_in_seconds=args.fade_in)
    spacing = column_spacing(bank.points_local)
    tracker = XYWHByteTrack(make_tracker_args(**XYWH_CONFIG))
    high, low = XYWH_CONFIG["track_high_thresh"], XYWH_CONFIG["track_low_thresh"]

    capture = cv2.VideoCapture(str(args.videos / f"{video_id}.mov"))
    if not capture.isOpened():
        raise RuntimeError(f"Не удалось открыть видео {video_id}")
    rows_meta, rows_t4t5, rows_all34 = [], [], []
    prev_tracked: list = []
    prev_det_index: dict[int, int] = {}
    eyes_per_frame, frame_seconds = [], []
    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        tick = time.perf_counter()
        xyxy, conf, classes = detections_for_frame(detections, frame_index)
        # 1. Глаза существующих треков смотрят туда, где трек предсказан на этом кадре.
        predictions = {int(t.track_id): tuple((t.mean[:4] + t.mean[4:8]).tolist()) for t in prev_tracked if int(t.track_id) in bank.track_ids}
        activity = bank.step(frame, predictions)
        for tid, act in activity.items():
            crop = bank.crops[tid]
            base = bank.baseline(tid)
            self_box = box_to_crop_frame(_xywh_to_xyxy(predictions[tid]), crop, bank.eye_size)
            kwargs = dict(ring_scale=args.ring_scale, min_margin=spacing)
            self_t4 = box_readout_t4t5(act, base, bank.points_local, self_box, **kwargs)
            self_34 = box_readout_all_types(act, base, bank.points_local, self_box, **kwargs)
            candidates = [j for j in range(len(xyxy)) if intersects(xyxy[j], crop)]
            for j in [-1] + candidates:
                if j < 0:
                    cand_t4, cand_34 = self_t4, self_34
                else:
                    local = box_to_crop_frame(xyxy[j], crop, bank.eye_size)
                    cand_t4 = box_readout_t4t5(act, base, bank.points_local, local, **kwargs)
                    cand_34 = box_readout_all_types(act, base, bank.points_local, local, **kwargs)
                rows_meta.append((frame_index, tid, j, prev_det_index.get(tid, -1), crop[2] - crop[0]))
                rows_t4t5.append([self_t4[n] for n in _BOX_T4T5] + [cand_t4[n] for n in _BOX_T4T5] + pair_terms(self_t4, cand_t4))
                rows_all34.append([self_34[n] for n in _BOX_ALL34] + [cand_34[n] for n in _BOX_ALL34])
        # 2. Трекер обновляется на кадре t (его состояние глаза не трогали).
        tracker.update(_DetSlice(xyxy, conf, classes))
        tracked = list(tracker.tracked_stracks)
        current_ids = {int(t.track_id) for t in tracked}
        prev_det_index = {int(t.track_id): original_detection_index(conf, subset_index=int(t.idx), score=float(t.score), high=high, low=low) for t in tracked}
        # 3. Рождение глаз новых треков на текущем кадре, удаление потерянных.
        births = {int(t.track_id): tuple(t.mean[:4].tolist()) for t in tracked if int(t.track_id) not in bank.track_ids}
        bank.birth(frame, births)
        bank.remove(set(bank.track_ids) - current_ids)
        prev_tracked = tracked
        eyes_per_frame.append(len(current_ids))
        frame_seconds.append(time.perf_counter() - tick)
        frame_index += 1
        if frame_index % 100 == 0:
            print(f"[{video_id}] {frame_index} кадров, глаз {len(current_ids)}, строк {len(rows_meta)}", flush=True)
    capture.release()
    if frame_index != int(detections["frame_count"]):
        raise RuntimeError(f"{video_id}: {frame_index} кадров против {detections['frame_count']} в кэше детекций")
    output.mkdir(parents=True, exist_ok=True)
    meta = np.asarray(rows_meta, dtype=np.int32).reshape(-1, 5)
    np.savez_compressed(
        data_path,
        frame=meta[:, 0], track_id=meta[:, 1], detection_index=meta[:, 2], prev_detection_index=meta[:, 3], crop_side=meta[:, 4],
        t4t5=np.asarray(rows_t4t5, dtype=np.float32).reshape(-1, len(READOUT_PAIR_FEATURES)),
        all34=np.asarray(rows_all34, dtype=np.float32).reshape(-1, len(READOUT_ALL34_FEATURES)),
        t4t5_names=np.asarray(READOUT_PAIR_FEATURES), all34_names=np.asarray(READOUT_ALL34_FEATURES),
    )
    metadata = {
        "video_id": video_id, "frame_count": frame_index, "fps": fps, "rows": int(len(rows_meta)),
        "eye": {"context": args.context, "min_side": args.min_side, "fade_in_seconds": args.fade_in, "eye_size": bank.eye_size, "column_spacing_eye_px": spacing, "steps_per_frame": bank.steps, "dt": bank.dt},
        "tracker": {"class": "XYWHByteTrack", **XYWH_CONFIG},
        "eyes_per_frame": distribution(eyes_per_frame), "births": bank.stats["births"], "eye_steps": bank.stats["steps"],
        "frame_seconds": distribution(frame_seconds), "bytes": data_path.stat().st_size,
    }
    save_json(meta_path, metadata)
    print(f"[OK] {video_id}: {frame_index} кадров, {len(rows_meta)} строк, глаз/кадр ≈ {np.mean(eyes_per_frame):.1f}, {np.mean(frame_seconds)*1000:.0f} мс/кадр", flush=True)
    return metadata


def _xywh_to_xyxy(xywh) -> np.ndarray:
    cx, cy, w, h = xywh
    return np.asarray([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dtype=np.float32)


def main() -> int:
    args = parse_args()
    summary = {video_id: build_video(video_id, args) for video_id in (args.video_id or list(VIDEO_IDS))}
    save_json(args.output / "summary.json", summary)
    print(f"[OK] {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
