"""Построить кэш внутренней активности FlyVis (34 типа клеток) по исходному видео.

Тот же рендер, режим tiled 2x3 и частота, что у опубликованного кэша потока
``runs/flyvis_stress_gate/<video>/motion/flyvis_tiled.*``; попутно проверяется,
что декодированный поток нового прохода совпадает с сохранённым (детерминизм
сети). Разметка не читается. Карточка: ``flyvis_internal_readout_v1``.
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

from scripts.run_bdd_video_rate_experiment import (  # noqa: E402
    distribution,
    make_motion_estimator,
    save_json,
)
from tracking.flyvis_activity import (  # noqa: E402
    OUTPUT_CELL_TYPES,
    mean_activity_by_type,
    save_activity_cache,
    type_column_indices,
)

VIDEO_IDS = (
    "b1c66a42-6f7d68ca",
    "b1c81faa-3df17267",
    "b1cac6a7-04e33135",
    "b1cebfb7-284f5117",
    "b232c7c9-d251d9ee",
    "b1d22449-15fb948f",
    "b251064f-e7a165fd",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-id", action="append", default=[])
    parser.add_argument("--videos", type=Path, default=PROJECT_ROOT / "data" / "bdd100k" / "videos" / "val")
    parser.add_argument("--flow-root", type=Path, default=PROJECT_ROOT / "runs" / "flyvis_stress_gate")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "runs" / "flyvis_activity")
    parser.add_argument("--source", default="flyvis_tiled")
    parser.add_argument("--max-dt", type=float, default=0.02)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def build_video(video_id: str, args: argparse.Namespace) -> dict:
    output = args.output / video_id
    data_path = output / f"{args.source}_activity.npz"
    meta_path = output / f"{args.source}_activity.json"
    if data_path.exists() and meta_path.exists() and not args.force:
        print(f"[CACHE] {video_id}: {data_path}", flush=True)
        return json.loads(meta_path.read_text(encoding="utf-8"))

    flow_meta = json.loads((args.flow_root / video_id / "motion" / f"{args.source}.json").read_text(encoding="utf-8"))
    with np.load(args.flow_root / video_id / "motion" / f"{args.source}.npz") as flow:
        flow_points = flow["points"]
        flow_vectors = flow["vectors"]
    fps = float(flow_meta["fps"])

    started = time.perf_counter()
    estimator = make_motion_estimator(args.source, fps, args.max_dt)
    node_types = estimator.network.connectome.nodes.type[:]
    indices = type_column_indices(node_types, OUTPUT_CELL_TYPES)
    initialization_seconds = time.perf_counter() - started

    capture = cv2.VideoCapture(str(args.videos / f"{video_id}.mov"))
    if not capture.isOpened():
        raise RuntimeError(f"Не удалось открыть видео {video_id}")
    frames, observation_seconds, flow_differences = [], [], []
    baseline = None
    points = None
    frame_count = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        tick = time.perf_counter()
        field = estimator.observe(frame)
        if estimator.last_activity is None:
            # Первый кадр: движения ещё нет, активность = baseline после fade-in.
            baseline = mean_activity_by_type(estimator.baseline_activity[:, None, :], indices)
            frames.append(baseline)
            points = field.points_xy.copy()
        else:
            frames.append(mean_activity_by_type(estimator.last_activity, indices))
        observation_seconds.append(time.perf_counter() - tick)
        if frame_count < len(flow_vectors):
            flow_differences.append(float(np.abs(field.vectors_xy - flow_vectors[frame_count]).max()))
        frame_count += 1
        if frame_count % 100 == 0:
            print(f"[{video_id}] {frame_count} кадров", flush=True)
    capture.release()
    if baseline is None or points is None:
        raise RuntimeError(f"{video_id}: видео не содержит кадров")
    if frame_count != int(flow_meta["frame_count"]):
        raise RuntimeError(f"{video_id}: {frame_count} кадров против {flow_meta['frame_count']} в кэше потока")
    if not np.allclose(points, flow_points):
        raise RuntimeError(f"{video_id}: координаты колонок не совпали с кэшем потока")

    # Порядок колонок внутри типа совпадает с порядком рецепторов BoxEye
    # (проверено 16.09.2026 по (u, v) для R1 и всех 34 типов).
    save_activity_cache(
        data_path,
        frames=np.stack(frames),
        baseline=baseline,
        points=points,
        cell_types=OUTPUT_CELL_TYPES,
        observation_seconds=np.asarray(observation_seconds, dtype=np.float64),
    )
    metadata = {
        "video_id": video_id,
        "source": args.source,
        "frame_count": frame_count,
        "fps": fps,
        "tiles": int(baseline.shape[0]),
        "cell_types": list(OUTPUT_CELL_TYPES),
        "columns_per_tile": int(baseline.shape[2]),
        "dtype": "float16, среднее по внутренним шагам; frames[0] = baseline",
        "integration": {
            "steps_per_video_frame": estimator.integration_steps,
            "integration_dt_seconds": estimator.dt,
            "temporal_resampling": estimator.temporal_resampling,
        },
        "flow_reproduction": {
            "compared_frames": len(flow_differences),
            "max_abs_difference_px": max(flow_differences) if flow_differences else None,
            "mean_abs_difference_px": float(np.mean(flow_differences)) if flow_differences else None,
        },
        "initialization_seconds_excluded": initialization_seconds,
        "observation": distribution(observation_seconds),
        "estimator_stats": estimator.stats,
        "bytes": data_path.stat().st_size,
    }
    save_json(meta_path, metadata)
    print(f"[OK] {video_id}: {frame_count} кадров, поток Δmax={metadata['flow_reproduction']['max_abs_difference_px']}", flush=True)
    return metadata


def main() -> int:
    args = parse_args()
    video_ids = args.video_id or list(VIDEO_IDS)
    summary = {video_id: build_video(video_id, args) for video_id in video_ids}
    save_json(args.output / "summary.json", summary)
    print(f"[OK] {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
