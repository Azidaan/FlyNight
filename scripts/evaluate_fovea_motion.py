"""Н1: остаточный поток фовеального глаза как поправка к предсказанию фильтра (карточка flyvis_fovea_motion_v1).

На опорных кадрах рамка трека после обновления сопоставляется с GT; цель —
центр GT-рамки. Сравниваются: предсказание фильтра (контроль), предсказание +
g·δ(FlyVis), перенос DIS с t−1, рамка сопоставленной детекции (оракул).
g подбирается на обучающих видео методом наименьших квадратов.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_flyvis_qmatch_gate import TEST_VIDEO_IDS, TRAIN_VIDEO_IDS, truth_arrays  # noqa: E402
from scripts.run_bdd_video_rate_experiment import DEFAULT_GAINS, save_json  # noqa: E402
from tracking.flyvis_tracker import box_iou  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flow-root", type=Path, default=PROJECT_ROOT / "runs" / "flyvis_fovea_flow")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT / "runs" / "video_rate")
    parser.add_argument("--labels", type=Path, default=PROJECT_ROOT / "data" / "bdd100k" / "labels" / "box_track_20" / "val")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "runs" / "flyvis_fovea_motion")
    parser.add_argument("--min-iou", type=float, default=0.5)
    return parser.parse_args()


def center(box: np.ndarray) -> np.ndarray:
    return (box[..., :2] + box[..., 2:]) / 2.0


def collect(video_id: str, args: argparse.Namespace) -> dict:
    with np.load(args.flow_root / video_id / "fovea_flow.npz") as z:
        flow = {k: z[k] for k in z.files}
    matches = json.loads((args.root / video_id / "alignment.json").read_text(encoding="utf-8"))["matches"]
    ref_of_raw = {int(m["video_frame_index"]): i for i, m in enumerate(matches)}
    truth = json.loads((args.labels / f"{video_id}.json").read_text(encoding="utf-8"))
    rows = []
    for i in range(len(flow["frame"])):
        t = int(flow["frame"][i])
        if t not in ref_of_raw or np.isnan(flow["out_box"][i, 0]):
            continue
        tb, tc, tids, _ = truth_arrays(truth[ref_of_raw[t]])
        if not len(tb):
            continue
        ov = box_iou(flow["out_box"][i][None, :], tb)[0]
        k = int(ov.argmax())
        if ov[k] < args.min_iou:
            continue
        gt_c = center(tb[k]); pred_c = center(flow["pred_box"][i]); out_c = center(flow["out_box"][i])
        dis_c = center(flow["dis_box"][i]) if not np.isnan(flow["dis_box"][i, 0]) else pred_c
        rows.append({"frame": t, "track_id": int(flow["track_id"][i]), "side": int(flow["crop_side"][i]), "gt": gt_c, "pred": pred_c, "delta": flow["delta"][i].astype(np.float64), "dis": dis_c, "oracle": out_c, "magnitude": float(flow["magnitude"][i]), "pred_box": flow["pred_box"][i].astype(np.float64), "n_inside": int(flow["n_inside"][i]), "scale": float(np.sqrt(max((tb[k][2] - tb[k][0]) * (tb[k][3] - tb[k][1]), 1.0)))})
    return {"rows": rows, "flow_rows": int(len(flow["frame"]))}


def fit_gain(rows: list[dict]) -> float:
    """g = argmin Σ |gt − pred − g·δ|² (один скаляр по обеим осям)."""
    residual = np.concatenate([(r["gt"] - r["pred"]) for r in rows])
    delta = np.concatenate([r["delta"] for r in rows])
    denominator = float(delta @ delta)
    return float(residual @ delta / denominator) if denominator > 0 else 0.0


def errors(rows: list[dict], gain: float) -> dict:
    def stats(values: np.ndarray) -> dict:
        return {"mean": float(values.mean()), "median": float(np.median(values)), "p90": float(np.percentile(values, 90))}
    e_pred = np.asarray([np.linalg.norm(r["gt"] - r["pred"]) for r in rows])
    e_fly = np.asarray([np.linalg.norm(r["gt"] - r["pred"] - gain * r["delta"]) for r in rows])
    e_dis = np.asarray([np.linalg.norm(r["gt"] - r["dis"]) for r in rows])
    e_oracle = np.asarray([np.linalg.norm(r["gt"] - r["oracle"]) for r in rows])
    n_pred = e_pred / np.asarray([r["scale"] for r in rows]); n_fly = e_fly / np.asarray([r["scale"] for r in rows])
    mag = np.asarray([r["magnitude"] for r in rows])
    return {"n": int(len(rows)), "pred": stats(e_pred), "fly": stats(e_fly), "dis": stats(e_dis), "oracle": stats(e_oracle),
            "normalized_pred": stats(n_pred), "normalized_fly": stats(n_fly),
            "fly_improves_fraction": float((e_fly < e_pred).mean()),
            "spearman_magnitude_vs_pred_error": float(spearmanr(mag, e_pred).correlation) if len(rows) > 5 else None,
            "spearman_delta_norm_vs_pred_error": float(spearmanr(np.linalg.norm(np.stack([r["delta"] for r in rows]), axis=1), e_pred).correlation) if len(rows) > 5 else None}


def main() -> int:
    args = parse_args()
    data = {v: collect(v, args) for v in TRAIN_VIDEO_IDS + TEST_VIDEO_IDS}
    train_rows = [r for v in TRAIN_VIDEO_IDS for r in data[v]["rows"]]
    gain = fit_gain(train_rows)
    gains = {"fitted_on_train": gain, "tiled_default": float(DEFAULT_GAINS["flyvis_tiled"])}
    by_side = {}
    for lo, hi in ((0, 96), (96, 160), (160, 256), (256, 10000)):
        sub = [r for r in train_rows if lo <= r["side"] < hi]
        by_side[f"{lo}-{hi}"] = {"n": len(sub), "gain": fit_gain(sub) if len(sub) > 20 else None}
    report = {"protocol": {"gain_rule": "МНК на обучающих видео по всем строкам; отдельно по корзинам стороны вырезки (диагностика)", "truth": f"опорные кадры, рамка после обновления ↔ GT IoU ≥ {args.min_iou}", "baselines": ["предсказание фильтра", "DIS-перенос рамки с t−1", "оракул: рамка после обновления"]},
              "gain": gains, "gain_by_crop_side_train": by_side, "flow_rows": {v: data[v]["flow_rows"] for v in data}, "per_video": {}, "aggregate": {}}
    for label, g in gains.items():
        report["per_video"][label] = {v: errors(data[v]["rows"], g) for v in data if data[v]["rows"]}
        report["aggregate"][label] = {"train": errors(train_rows, g), "test": errors([r for v in TEST_VIDEO_IDS for r in data[v]["rows"]], g)}
    test = report["per_video"]["fitted_on_train"]
    acceptance = {
        "fly_mean_below_pred_every_test_video": all(test[v]["fly"]["mean"] < test[v]["pred"]["mean"] for v in TEST_VIDEO_IDS),
        "fly_median_below_pred_every_test_video": all(test[v]["fly"]["median"] < test[v]["pred"]["median"] for v in TEST_VIDEO_IDS),
        "magnitude_correlates_with_error_test": (report["aggregate"]["fitted_on_train"]["test"]["spearman_magnitude_vs_pred_error"] or 0) > 0.2,
    }
    report["acceptance"] = acceptance
    args.output.mkdir(parents=True, exist_ok=True)
    save_json(args.output / "report.json", report)
    for label in gains:
        a = report["aggregate"][label]["test"]
        print(f"[{label}] g={gains[label]:.3f} TEST n={a['n']} pred {a['pred']['mean']:.2f}/{a['pred']['median']:.2f} | fly {a['fly']['mean']:.2f}/{a['fly']['median']:.2f} | dis {a['dis']['mean']:.2f}/{a['dis']['median']:.2f} | oracle {a['oracle']['mean']:.2f} | improves {a['fly_improves_fraction']:.3f} | ρ(|flow|,err) {a['spearman_magnitude_vs_pred_error']}")
    print("by video (fitted):", {v[:8]: (round(test[v]["pred"]["mean"], 2), round(test[v]["fly"]["mean"], 2), round(test[v]["dis"]["mean"], 2)) for v in TEST_VIDEO_IDS})
    print(json.dumps(acceptance, ensure_ascii=False))
    print(f"[OK] {args.output / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
