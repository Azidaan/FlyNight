"""Проверить все официальные checkpoint одного ансамбля FlyVis.

Для каждой модели загружаются нейронная сеть и декодер, затем выполняются два
противоположных горизонтальных движения. Отчёт фиксирует ошибки загрузки,
некорректные числа и знак усреднённого ответа. Это проверка целостности и
работоспособности ансамбля, а не оценка качества трекинга.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.flyvis_probe import (  # noqa: E402
    finite_summary,
    make_bar_movie,
    run_model,
    sha256,
    timed,
)
from tracking.flyvis_adapter import DEFAULT_DATA_DIR, activate_flyvis  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ensemble", default="flow/0000")
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "runs" / "flyvis" / "ensemble_probe.json",
    )
    args = parser.parse_args()

    flyvis = activate_flyvis()
    ensemble = flyvis.Ensemble(args.ensemble)
    if not ensemble.names:
        raise RuntimeError(f"В ансамбле нет моделей: {args.ensemble}")

    first = ensemble[0]
    extent = int(first.dir.config.network.connectome.extent)
    kernel_size = int(first.dir.config.task.dataset.boxfilter.kernel_size)
    from flyvis.datasets.rendering import BoxEye

    eye = BoxEye(extent=extent, kernel_size=kernel_size)
    eye.conv = eye.conv.to(flyvis.device)
    height, width = (int(value) for value in eye.min_frame_size.tolist())
    lr_movie = make_bar_movie(height, width, args.frames, axis="x")
    rl_movie = make_bar_movie(height, width, args.frames, axis="x", reverse=True)
    lr_input, _ = timed(lambda: eye(lr_movie.to(flyvis.device)))
    rl_input, _ = timed(lambda: eye(rl_movie.to(flyvis.device)))

    rows = []
    network = None
    decoder_dict = None
    started = time.perf_counter()
    for index, name in enumerate(ensemble.names):
        view = ensemble[name]
        row = {"index": index, "name": name, "status": "error"}
        item_start = time.perf_counter()
        try:
            network = view.init_network(network=network)
            decoder_dict = view.init_decoder(decoder=decoder_dict)
            decoder = decoder_dict["flow"]
            network.eval().requires_grad_(False)
            decoder.eval().requires_grad_(False)
            with torch.inference_mode():
                activity_lr, flow_lr, _, _ = run_model(network, decoder, lr_input, args.dt)
                activity_rl, flow_rl, _, _ = run_model(network, decoder, rl_input, args.dt)
            x_lr = float(flow_lr.mean(dim=(0, 1, 3))[0].item())
            x_rl = float(flow_rl.mean(dim=(0, 1, 3))[0].item())
            checkpoint = Path(view.dir.path) / "best_chkpt"
            row.update(
                {
                    "status": "ok",
                    "checkpoint_sha256": sha256(checkpoint),
                    "x_left_to_right": x_lr,
                    "x_right_to_left": x_rl,
                    "opposite_horizontal_response": x_lr * x_rl < 0,
                    "activity_lr": finite_summary(activity_lr),
                    "activity_rl": finite_summary(activity_rl),
                    "flow_lr": finite_summary(flow_lr),
                    "flow_rl": finite_summary(flow_rl),
                }
            )
        except Exception as error:  # отчёт должен сохранить все отказы ансамбля
            row["error"] = f"{type(error).__name__}: {error}"
        row["seconds"] = time.perf_counter() - item_start
        rows.append(row)
        print(
            f"[{index + 1:02d}/{len(ensemble.names):02d}] {name}: {row['status']} "
            f"({row['seconds']:.2f} с)",
            flush=True,
        )

    ok_rows = [row for row in rows if row["status"] == "ok"]
    finite_rows = [
        row
        for row in ok_rows
        if all(
            row[key]["all_finite"]
            for key in ("activity_lr", "activity_rl", "flow_lr", "flow_rl")
        )
    ]
    opposite_rows = [row for row in ok_rows if row["opposite_horizontal_response"]]
    result = {
        "scope": "FlyVis ensemble integrity and directional-response probe",
        "tracking_metrics_computed": False,
        "ensemble": args.ensemble,
        "expected_models": len(ensemble.names),
        "loaded_models": len(ok_rows),
        "all_finite_models": len(finite_rows),
        "opposite_horizontal_models": len(opposite_rows),
        "all_checkpoints_loaded": len(ok_rows) == len(ensemble.names),
        "all_outputs_finite": len(finite_rows) == len(ensemble.names),
        "all_direction_checks_passed": len(opposite_rows) == len(ensemble.names),
        "total_seconds": time.perf_counter() - started,
        "models": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "models"}, ensure_ascii=False, indent=2))
    print(f"[OK] отчёт: {args.output}")
    return 0 if result["all_checkpoints_loaded"] and result["all_outputs_finite"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

