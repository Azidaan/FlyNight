"""Н9/М1: фронт-энд «фоторецептор + ламина» перед глазом FlyVis — ступень A на val-семёрке (карточка flyvis_retina_frontend_v1).

Для каждого режима сетчатки (none = прежний кэш `runs/flyvis_fovea_flow`, norm, photoreceptor,
lamina) по тем же строкам фовеального потока: ρ(|δ|, ошибка предсказания) по видео (мера Н1),
ошибка центра после поправки g·δ (g по МНК на train-четвёрке), и голова надёжности
«геометрия + муха без DIS» с обучением по схеме «исключить одно видео» (7 сгибов) —
AP на исключённом видео. Отдельно — тёмное видео b1d22449 и три held-out.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_flyvis_qmatch_gate import TEST_VIDEO_IDS, TRAIN_VIDEO_IDS  # noqa: E402
from scripts.evaluate_fovea_motion import collect, errors, fit_gain  # noqa: E402
from scripts.evaluate_motion_reliability import DIS, FLY, GEOMETRY, NAMES, rows_to_matrix  # noqa: E402
from scripts.run_bdd_video_rate_experiment import save_json  # noqa: E402

DARK_VIDEO = "b1d22449-15fb948f"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modes", default="none,norm,photoreceptor,lamina")
    parser.add_argument("--flow-roots", default="none=runs/flyvis_fovea_flow,norm=runs/flyvis_fovea_flow_retina_norm,photoreceptor=runs/flyvis_fovea_flow_retina_photo,lamina=runs/flyvis_fovea_flow_retina_lamina")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT / "runs" / "video_rate")
    parser.add_argument("--labels", type=Path, default=PROJECT_ROOT / "data" / "bdd100k" / "labels" / "box_track_20" / "val")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "runs" / "retina_frontend")
    parser.add_argument("--positive-rate", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_mode(flow_root: Path, args: argparse.Namespace, video_ids: list[str]) -> dict[str, dict]:
    ns = argparse.Namespace(flow_root=flow_root, root=args.root, labels=args.labels, min_iou=0.5)
    out = {}
    for v in video_ids:
        rows = collect(v, ns)["rows"]
        X, err = rows_to_matrix(rows)
        out[v] = {"rows": rows, "X": X, "err": err}
    return out


def leave_one_video_out(data: dict[str, dict], cols: list[int], tau: float, rng: np.random.Generator, shuffle_fly: bool = False) -> dict[str, float]:
    result = {}
    fly_idx = [NAMES.index(n) for n in FLY]
    for held in data:
        Xtr = np.concatenate([data[v]["X"] for v in data if v != held]); ytr = np.concatenate([(data[v]["err"] > tau) for v in data if v != held]).astype(np.int8)
        Xte = data[held]["X"].copy(); yte = (data[held]["err"] > tau).astype(np.int8)
        if shuffle_fly:
            Xte[:, fly_idx] = Xte[rng.permutation(len(Xte))][:, fly_idx]
        if 0 < yte.sum() < len(yte):
            model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000, random_state=42)).fit(Xtr[:, cols], ytr)
            result[held] = float(average_precision_score(yte, model.predict_proba(Xte[:, cols])[:, 1]))
    return result


def main() -> int:
    args = parse_args()
    roots = {k: PROJECT_ROOT / v for k, v in (item.split("=") for item in args.flow_roots.split(","))}
    modes = [m for m in args.modes.split(",") if (roots[m] / DARK_VIDEO / "fovea_flow.npz").exists()]
    video_ids = list(TRAIN_VIDEO_IDS + TEST_VIDEO_IDS)
    rng = np.random.default_rng(args.seed)
    report = {"modes": {}, "videos": video_ids, "dark_video": DARK_VIDEO}
    geo_cols = [NAMES.index(n) for n in GEOMETRY]
    fly_cols = geo_cols + [NAMES.index(n) for n in FLY]
    dis_cols = geo_cols + [NAMES.index(n) for n in DIS]
    both_cols = dis_cols + [NAMES.index(n) for n in FLY]
    tau = None
    for mode in modes:
        data = load_mode(roots[mode], args, video_ids)
        if tau is None:  # общий порог события для всех режимов: по режиму none на train-четвёрке
            tau = float(np.quantile(np.concatenate([data[v]["err"] for v in TRAIN_VIDEO_IDS]), 1.0 - args.positive_rate))
        gain = fit_gain([r for v in TRAIN_VIDEO_IDS for r in data[v]["rows"]])
        per_video = {}
        for v in video_ids:
            rows = data[v]["rows"]
            mag = np.asarray([r["magnitude"] for r in rows]); err = data[v]["err"]
            e = errors(rows, gain)
            per_video[v] = {"n": len(rows), "rho_mag_err": float(np.corrcoef(mag, err)[0, 1]) if len(rows) > 2 else float("nan"),
                            "err_pred_mean": e["pred"]["mean"], "err_fly_mean": e["fly"]["mean"], "err_dis_mean": e["dis"]["mean"],
                            "fly_mag_median": float(np.median(mag))}
        heads = {"geometry": leave_one_video_out(data, geo_cols, tau, rng), "geometry_fly": leave_one_video_out(data, fly_cols, tau, rng),
                 "geometry_fly_shuffled": leave_one_video_out(data, fly_cols, tau, rng, shuffle_fly=True),
                 "geometry_dis": leave_one_video_out(data, dis_cols, tau, rng), "geometry_dis_fly": leave_one_video_out(data, both_cols, tau, rng)}
        summary = {h: {"mean_ap": float(np.mean(list(r.values()))), "dark_ap": r.get(DARK_VIDEO), "heldout_mean_ap": float(np.mean([r[v] for v in TEST_VIDEO_IDS if v in r]))} for h, r in heads.items()}
        report["modes"][mode] = {"gain": gain, "tau": tau, "per_video": per_video, "heads_by_video": heads, "heads": summary,
                                 "rho_mean": float(np.nanmean([per_video[v]["rho_mag_err"] for v in video_ids])), "rho_dark": per_video[DARK_VIDEO]["rho_mag_err"]}
        s = summary
        print(f"[{mode:13s}] ρ mean {report['modes'][mode]['rho_mean']:+.3f} dark {report['modes'][mode]['rho_dark']:+.3f} | LOVO AP mean: geo {s['geometry']['mean_ap']:.4f} +fly {s['geometry_fly']['mean_ap']:.4f} (shuf {s['geometry_fly_shuffled']['mean_ap']:.4f}) +dis {s['geometry_dis']['mean_ap']:.4f} +dis+fly {s['geometry_dis_fly']['mean_ap']:.4f} | dark: geo {s['geometry']['dark_ap']:.4f} +fly {s['geometry_fly']['dark_ap']:.4f} +dis {s['geometry_dis']['dark_ap']:.4f} +dis+fly {s['geometry_dis_fly']['dark_ap']:.4f}", flush=True)
    if "none" in report["modes"]:
        base = report["modes"]["none"]["heads"]
        report["acceptance"] = {}
        for mode in modes:
            if mode == "none":
                continue
            s = report["modes"][mode]["heads"]
            fly_gain_dark = s["geometry_fly"]["dark_ap"] - s["geometry"]["dark_ap"]
            base_gain_dark = base["geometry_fly"]["dark_ap"] - base["geometry"]["dark_ap"]
            report["acceptance"][mode] = {
                "fly_gain_dark_over_geometry": fly_gain_dark, "fly_gain_dark_none": base_gain_dark,
                "dark_improves": fly_gain_dark > base_gain_dark,
                "fly_beats_shuffled_dark": s["geometry_fly"]["dark_ap"] > s["geometry_fly_shuffled"]["dark_ap"],
                "mean_not_worse": s["geometry_fly"]["mean_ap"] >= base["geometry_fly"]["mean_ap"] - 0.005,
                "rho_dark_improves": report["modes"][mode]["rho_dark"] > report["modes"]["none"]["rho_dark"],
            }
        print(json.dumps(report["acceptance"], ensure_ascii=False, indent=1))
    args.output.mkdir(parents=True, exist_ok=True)
    save_json(args.output / "report_val7.json", report)
    print(f"[OK] {args.output / 'report_val7.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
