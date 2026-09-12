"""Проверка официального FlyVis на синтетическом движении и BDD100K.

Этот стенд не запускает трекер и не считает HOTA/AssA. Он отвечает на более
ранний вопрос: загружаются ли официальный коннектом, checkpoint и декодер;
реагирует ли модель на противоположные направления; может ли она обработать
реальные последовательности BDD100K; сколько это занимает времени и памяти.

Пример:
    .venv/Scripts/python.exe scripts/flyvis_probe.py --frames 6
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.bdd100k_loader import DATA_ROOT  # noqa: E402
from tracking.flyvis_adapter import activate_flyvis  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed(callable_):
    synchronize()
    start = time.perf_counter()
    value = callable_()
    synchronize()
    return value, time.perf_counter() - start


def load_gray_frames(paths: list[Path], height: int, width: int) -> torch.Tensor:
    frames = []
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise RuntimeError(f"Не удалось прочитать кадр: {path}")
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        frames.append(image.astype(np.float32) / 255.0)
    return torch.from_numpy(np.stack(frames, axis=0)).unsqueeze(0)


def temporal_interpolate(movie: torch.Tensor, source_fps: float, dt: float) -> tuple[torch.Tensor, int]:
    """Линейно привести редкие наблюдения к шагу интегрирования FlyVis."""
    source_dt = 1.0 / source_fps
    steps = int(round(source_dt / dt))
    if not math.isclose(steps * dt, source_dt, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(
            f"Интервал {source_dt:g} с нельзя представить целым числом шагов dt={dt:g}"
        )
    if steps == 1:
        return movie, steps

    segments = [movie[:, :1]]
    for index in range(movie.shape[1] - 1):
        left = movie[:, index : index + 1]
        right = movie[:, index + 1 : index + 2]
        alpha = torch.arange(1, steps + 1, device=movie.device, dtype=movie.dtype)
        alpha = (alpha / steps).view(1, steps, 1, 1)
        segments.append(left * (1.0 - alpha) + right * alpha)
    return torch.cat(segments, dim=1), steps


def make_bar_movie(
    height: int,
    width: int,
    frames: int,
    axis: str = "x",
    reverse: bool = False,
    static: bool = False,
) -> torch.Tensor:
    movie = np.full((1, frames, height, width), 0.5, dtype=np.float32)
    span = width if axis == "x" else height
    positions = np.linspace(span * 0.25, span * 0.75, frames)
    if reverse:
        positions = positions[::-1]
    if static:
        positions[:] = span * 0.5
    half_width = max(3, span // 40)
    for frame_index, center in enumerate(positions.astype(int)):
        left = max(0, center - half_width)
        right = min(span, center + half_width)
        if axis == "x":
            movie[0, frame_index, :, left:right] = 1.0
        elif axis == "y":
            movie[0, frame_index, left:right, :] = 1.0
        else:
            raise ValueError(f"Неизвестная ось движения: {axis}")
    return torch.from_numpy(movie)


def run_model(network, decoder, stimulus: torch.Tensor, dt: float) -> tuple[torch.Tensor, torch.Tensor, float, float]:
    stimulus = stimulus.to(torch.float32)
    initial_state, warmup_s = timed(
        lambda: network.fade_in_state(1.0, dt, stimulus[:, 0])
    )
    activity, network_s = timed(
        lambda: network.simulate(stimulus, dt, initial_state=initial_state)
    )
    flow, decoder_s = timed(lambda: decoder(activity))
    return activity, flow, warmup_s + network_s, decoder_s


def finite_summary(array: torch.Tensor) -> dict:
    finite = torch.isfinite(array)
    nonfinite_count = int(torch.count_nonzero(~finite).item())
    return {
        "all_finite": nonfinite_count == 0,
        "nonfinite_count": nonfinite_count,
        "finite_fraction": float(finite.to(torch.float64).mean().item()),
    }


def flow_summary(flow: torch.Tensor) -> dict:
    xy = flow.mean(dim=(0, 1, 3)).detach().cpu().numpy()
    magnitude = torch.linalg.vector_norm(flow, dim=2)
    return {
        "mean_xy": [float(xy[0]), float(xy[1])],
        "mean_magnitude": float(magnitude.mean().item()),
        "max_magnitude": float(magnitude.max().item()),
        "finite": finite_summary(flow),
    }


def choose_video(split: str, requested: str | None) -> tuple[str, Path]:
    root = DATA_ROOT / "images" / "track" / split
    if requested:
        video_dir = root / requested
        if not video_dir.is_dir():
            raise FileNotFoundError(f"Видео BDD100K не найдено: {video_dir}")
        return requested, video_dir
    candidates = sorted(path for path in root.iterdir() if path.is_dir())
    if not candidates:
        raise FileNotFoundError(f"В {root} нет каталогов видео")
    return candidates[0].name, candidates[0]


def save_flow_figure(
    output_path: Path,
    receptor_centers: torch.Tensor,
    panels: list[tuple[str, torch.Tensor]],
) -> None:
    """Сохранить сопоставимую карту среднего потока по 721 омматидию."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    centers = receptor_centers.detach().cpu().numpy()
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    magnitudes = []
    means = []
    for _, flow in panels:
        mean_flow = flow.mean(dim=(0, 1)).detach().cpu().numpy()
        means.append(mean_flow)
        magnitudes.append(np.linalg.norm(mean_flow, axis=0))
    vmax = max(float(magnitude.max()) for magnitude in magnitudes) or 1.0

    scatter = None
    for axis, (title, _), mean_flow, magnitude in zip(
        axes.flat, panels, means, magnitudes
    ):
        scatter = axis.scatter(
            centers[:, 1], centers[:, 0], c=magnitude, s=18,
            cmap="viridis", vmin=0.0, vmax=vmax,
        )
        axis.quiver(
            centers[:, 1], centers[:, 0], mean_flow[0], -mean_flow[1],
            color="white", alpha=0.7, scale=max(vmax * 18, 1.0), width=0.003,
        )
        axis.set_title(title)
        axis.set_aspect("equal")
        axis.invert_yaxis()
        axis.set_xticks([])
        axis.set_yticks([])
    assert scatter is not None
    fig.colorbar(scatter, ax=axes, label="Модуль декодированного движения")
    fig.suptitle("FlyVis 1.2.0: проверка направленного ответа")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="flow/0000/000")
    parser.add_argument("--split", default="val")
    parser.add_argument("--video-id", default=None)
    parser.add_argument("--frames", type=int, default=6)
    parser.add_argument("--source-fps", type=float, default=5.0)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--synthetic-frames", type=int, default=16)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "runs" / "flyvis" / "probe",
    )
    args = parser.parse_args()
    if args.frames < 2 or args.synthetic_frames < 2:
        parser.error("Нужно минимум два реальных и два синтетических кадра")

    args.output.mkdir(parents=True, exist_ok=True)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    init_start = time.perf_counter()
    flyvis = activate_flyvis()
    view = flyvis.NetworkView(args.model)
    network = view.init_network()
    decoders = view.init_decoder()
    decoder = decoders["flow"]
    network.eval().requires_grad_(False)
    decoder.eval().requires_grad_(False)
    init_s = time.perf_counter() - init_start

    extent = int(view.dir.config.network.connectome.extent)
    kernel_size = int(view.dir.config.task.dataset.boxfilter.kernel_size)
    from flyvis.datasets.rendering import BoxEye

    eye = BoxEye(extent=extent, kernel_size=kernel_size)
    eye.conv = eye.conv.to(flyvis.device)
    height, width = (int(value) for value in eye.min_frame_size.tolist())

    left_to_right = make_bar_movie(height, width, args.synthetic_frames, axis="x")
    right_to_left = make_bar_movie(
        height, width, args.synthetic_frames, axis="x", reverse=True
    )
    top_to_bottom = make_bar_movie(height, width, args.synthetic_frames, axis="y")
    bottom_to_top = make_bar_movie(
        height, width, args.synthetic_frames, axis="y", reverse=True
    )
    static_bar = make_bar_movie(
        height, width, args.synthetic_frames, axis="x", static=True
    )
    synthetic_input, render_lr_s = timed(lambda: eye(left_to_right.to(flyvis.device)))
    synthetic_reverse, render_rl_s = timed(lambda: eye(right_to_left.to(flyvis.device)))
    synthetic_down, render_down_s = timed(lambda: eye(top_to_bottom.to(flyvis.device)))
    synthetic_up, render_up_s = timed(lambda: eye(bottom_to_top.to(flyvis.device)))
    synthetic_static, render_static_s = timed(lambda: eye(static_bar.to(flyvis.device)))
    with torch.inference_mode():
        activity_lr, flow_lr, net_lr_s, dec_lr_s = run_model(
            network, decoder, synthetic_input, args.dt
        )
        activity_rl, flow_rl, net_rl_s, dec_rl_s = run_model(
            network, decoder, synthetic_reverse, args.dt
        )
        activity_down, flow_down, net_down_s, dec_down_s = run_model(
            network, decoder, synthetic_down, args.dt
        )
        activity_up, flow_up, net_up_s, dec_up_s = run_model(
            network, decoder, synthetic_up, args.dt
        )
        activity_static, flow_static, net_static_s, dec_static_s = run_model(
            network, decoder, synthetic_static, args.dt
        )

    video_id, video_dir = choose_video(args.split, args.video_id)
    frame_paths = sorted(video_dir.glob("*.jpg"))[: args.frames]
    if len(frame_paths) < args.frames:
        raise RuntimeError(
            f"В {video_dir} только {len(frame_paths)} кадров, запрошено {args.frames}"
        )
    cartesian = load_gray_frames(frame_paths, height, width).to(flyvis.device)
    bdd_hex, bdd_render_s = timed(lambda: eye(cartesian))
    bdd_input, internal_steps = temporal_interpolate(bdd_hex, args.source_fps, args.dt)
    with torch.inference_mode():
        bdd_activity, bdd_flow, bdd_network_s, bdd_decoder_s = run_model(
            network, decoder, bdd_input, args.dt
        )
    sample_indices = torch.arange(0, bdd_flow.shape[1], internal_steps)
    sampled_flow = bdd_flow[:, sample_indices]

    checkpoint = PROJECT_ROOT / "runs" / "flyvis" / "results" / Path(args.model) / "best_chkpt"
    archive = PROJECT_ROOT / "runs" / "flyvis" / "results_pretrained_models.zip"
    result = {
        "status": "ok",
        "scope": "FlyVis functionality probe; tracking metrics are not computed",
        "tracking_metrics_computed": False,
        "versions": {
            "flyvis": flyvis.__version__,
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "opencv": cv2.__version__,
            "numpy": np.__version__,
        },
        "device": str(flyvis.device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "model": {
            "name": args.model,
            "archive_sha256": sha256(archive),
            "checkpoint_sha256": sha256(checkpoint),
            "network_parameters": sum(p.numel() for p in network.parameters()),
            "decoder_parameters": sum(p.numel() for p in decoder.parameters()),
            "activity_shape": list(activity_lr.shape),
            "flow_shape": list(flow_lr.shape),
            "eye_extent": extent,
            "eye_kernel_size": kernel_size,
            "eye_frame_size": [height, width],
            "ommatidia": int(synthetic_input.shape[-1]),
        },
        "synthetic": {
            "left_to_right": flow_summary(flow_lr),
            "right_to_left": flow_summary(flow_rl),
            "top_to_bottom": flow_summary(flow_down),
            "bottom_to_top": flow_summary(flow_up),
            "static": flow_summary(flow_static),
            "activity_finite_lr": finite_summary(activity_lr),
            "activity_finite_rl": finite_summary(activity_rl),
            "activity_finite_down": finite_summary(activity_down),
            "activity_finite_up": finite_summary(activity_up),
            "activity_finite_static": finite_summary(activity_static),
            "opposite_horizontal_response": bool(
                flow_lr.mean(dim=(0, 1, 3))[0].item()
                * flow_rl.mean(dim=(0, 1, 3))[0].item()
                < 0
            ),
            "opposite_vertical_response": bool(
                flow_down.mean(dim=(0, 1, 3))[1].item()
                * flow_up.mean(dim=(0, 1, 3))[1].item()
                < 0
            ),
            "render_seconds": (
                render_lr_s + render_rl_s + render_down_s + render_up_s + render_static_s
            ),
            "network_seconds": (
                net_lr_s + net_rl_s + net_down_s + net_up_s + net_static_s
            ),
            "decoder_seconds": (
                dec_lr_s + dec_rl_s + dec_down_s + dec_up_s + dec_static_s
            ),
        },
        "bdd100k": {
            "split": args.split,
            "video_id": video_id,
            "frames": [path.name for path in frame_paths],
            "source_fps_assumption": args.source_fps,
            "integration_dt_seconds": args.dt,
            "interpolation_steps_per_observation": internal_steps,
            "internal_frames": int(bdd_input.shape[1]),
            "sampled_flow_shape": list(sampled_flow.shape),
            "activity_finite": finite_summary(bdd_activity),
            "flow": flow_summary(sampled_flow),
            "render_seconds": bdd_render_s,
            "network_seconds": bdd_network_s,
            "decoder_seconds": bdd_decoder_s,
        },
        "timing": {
            "initialization_seconds": init_s,
            "peak_gpu_memory_mib": (
                torch.cuda.max_memory_allocated() / 1024**2
                if torch.cuda.is_available()
                else None
            ),
        },
    }
    output_json = args.output / f"{video_id}_{args.model.replace('/', '_')}.json"
    output_npz = args.output / f"{video_id}_{args.model.replace('/', '_')}.npz"
    output_png = args.output / f"{video_id}_{args.model.replace('/', '_')}.png"
    output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez_compressed(
        output_npz,
        synthetic_left_to_right=flow_lr.detach().cpu().numpy(),
        synthetic_right_to_left=flow_rl.detach().cpu().numpy(),
        synthetic_top_to_bottom=flow_down.detach().cpu().numpy(),
        synthetic_bottom_to_top=flow_up.detach().cpu().numpy(),
        synthetic_static=flow_static.detach().cpu().numpy(),
        bdd_sampled_flow=sampled_flow.detach().cpu().numpy(),
        bdd_frame_names=np.array([path.name for path in frame_paths]),
    )
    save_flow_figure(
        output_png,
        eye.receptor_centers,
        [
            ("Слева направо", flow_lr),
            ("Справа налево", flow_rl),
            ("Сверху вниз", flow_down),
            ("Снизу вверх", flow_up),
            ("Неподвижная полоса", flow_static),
            (f"BDD100K: {video_id}", sampled_flow),
        ],
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"[OK] отчёт: {output_json}")
    print(f"[OK] массивы: {output_npz}")
    print(f"[OK] рисунок: {output_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
