"""Обучить голову надёжности предсказания для трекера (tracking/reliability_head.py).

Событие — как в Н2: ошибка центра предсказания Калмана на опорном кадре, нормированная на
sqrt(площади GT), выше τ_e (≈25 % положительных на train). Признаки — те, что трекер
считает онлайн (tracking.reliability_head.base_features / fly_features). Обучение —
train-сплит кэшей (130 роликов), отчёт AP — dev и test.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_fovea_motion import collect  # noqa: E402
from scripts.run_bdd_video_rate_experiment import save_json  # noqa: E402
from tracking.reliability_head import BASE_FEATURES, FLY_FEATURES, base_features, fly_features  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", type=Path, default=PROJECT_ROOT / "runs" / "train_cache")
    parser.add_argument("--fly-root", type=Path, default=PROJECT_ROOT / "runs" / "train_cache", help="где лежат <id>/fovea_flow.npz (режим сетчатки)")
    parser.add_argument("--labels", type=Path, default=PROJECT_ROOT / "data" / "bdd100k" / "labels" / "box_track_20" / "train")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--positive-rate", type=float, default=0.25)
    return parser.parse_args()


def matrices(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pb = np.asarray([r["pred_box"] for r in rows], dtype=np.float64)
    pred_xywh = np.concatenate([(pb[:, :2] + pb[:, 2:]) / 2.0, pb[:, 2:] - pb[:, :2]], axis=1)
    dis_c = np.asarray([r["dis"] for r in rows], dtype=np.float64)
    dis_xyxy = np.concatenate([dis_c - 0.5, dis_c + 0.5], axis=1)
    Xb = base_features(pred_xywh, dis_xyxy)
    Xf = fly_features(np.asarray([r["delta"] for r in rows]), np.asarray([r["magnitude"] for r in rows]), np.asarray([r["n_inside"] for r in rows]))
    err = np.asarray([np.linalg.norm(r["gt"] - r["pred"]) / r["scale"] for r in rows])
    return Xb, Xf, err


def fit(X: np.ndarray, y: np.ndarray, names: list[str]) -> tuple[dict, StandardScaler, LogisticRegression]:
    scaler = StandardScaler().fit(X)
    scale = np.where(scaler.scale_ > 0, scaler.scale_, 1.0)
    model = LogisticRegression(C=1.0, max_iter=3000, random_state=42).fit((X - scaler.mean_) / scale, y)
    spec = {"features": names, "mean": scaler.mean_.tolist(), "scale": scale.tolist(), "coef": model.coef_[0].tolist(), "intercept": float(model.intercept_[0])}
    return spec, scaler, model


def main() -> int:
    args = parse_args()
    selection = json.loads((args.train_cache / "selection.json").read_text(encoding="utf-8"))["videos"]
    ns = argparse.Namespace(flow_root=args.fly_root, root=args.train_cache, labels=args.labels, min_iou=0.5)
    data = {}
    for split in ("train", "dev", "test"):
        parts = [matrices(r) for r in (collect(v, ns)["rows"] for v in selection[split]) if r]
        data[split] = tuple(np.concatenate([p[k] for p in parts]) for k in range(3))
        print(f"[DATA] {split}: {len(data[split][2])} наблюдений", flush=True)
    tau = float(np.quantile(data["train"][2], 1.0 - args.positive_rate))
    y = {s: (d[2] > tau).astype(np.int8) for s, d in data.items()}
    base_spec, *_ = fit(data["train"][0], y["train"], list(BASE_FEATURES))
    fly_spec, *_ = fit(np.concatenate([data["train"][0], data["train"][1]], axis=1), y["train"], list(BASE_FEATURES + FLY_FEATURES))
    from tracking.reliability_head import LogisticHead

    report = {}
    for name, spec, cols in (("base", base_spec, False), ("fly", fly_spec, True)):
        head = LogisticHead(spec)
        report[name] = {s: float(average_precision_score(y[s], head(np.concatenate([d[0], d[1]], axis=1) if cols else d[0]))) for s, d in data.items()}
        print(f"[AP] {name}: " + ", ".join(f"{s} {v:.4f}" for s, v in report[name].items()), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_json(args.output, {"tau": tau, "positive_rate": {s: float(v.mean()) for s, v in y.items()}, "fly_root": str(args.fly_root), "base": base_spec, "fly": fly_spec, "average_precision": report})
    print(f"[OK] {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
