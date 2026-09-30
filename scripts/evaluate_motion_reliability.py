"""Н2: голова достоверности движения по остаточному потоку фовеального глаза (карточка motion_reliability_v1).

Событие — нормированная ошибка центра предсказания фильтра на опорном кадре выше
τ_e (τ_e: ≈25 % положительных на train). Лестница логистических голов:
geometry → + DIS-рассогласование → + FlyVis → + оба; контроль shuffled FlyVis.
Обучение на train-сплите кэшей ступени 1; отчёт на dev, test и val-семёрке.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_event_heads import ece_score, f1_threshold  # noqa: E402
from scripts.evaluate_flyvis_qmatch_gate import TEST_VIDEO_IDS, TRAIN_VIDEO_IDS, paired_video_bootstrap  # noqa: E402
from scripts.evaluate_fovea_motion import collect  # noqa: E402
from scripts.run_bdd_video_rate_experiment import save_json  # noqa: E402

GEOMETRY = ("box_w", "box_h", "box_area", "center_x", "center_y", "crop_side")
DIS = ("dis_disagreement",)
FLY = ("fly_norm", "fly_dx", "fly_dy", "fly_magnitude", "fly_n_inside")
SHUFFLED = tuple(f"shuffled_{n}" for n in FLY)
NAMES = GEOMETRY + DIS + FLY + SHUFFLED
GROUPS = {"geometry": GEOMETRY, "geometry_dis": GEOMETRY + DIS, "geometry_fly": GEOMETRY + FLY, "geometry_dis_fly": GEOMETRY + DIS + FLY, "geometry_dis_shuffled": GEOMETRY + DIS + SHUFFLED}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", type=Path, default=PROJECT_ROOT / "runs" / "train_cache")
    parser.add_argument("--train-flow", type=Path, default=None, help="где лежат <id>/fovea_flow.npz обучающих роликов (по умолчанию = --train-cache)")
    parser.add_argument("--dark-threshold", type=float, default=60.0, help="порог средней яркости ролика (0..255) для тёмной подвыборки")
    parser.add_argument("--train-videos", type=Path, default=PROJECT_ROOT / "data" / "bdd100k" / "videos" / "train")
    parser.add_argument("--val-videos", type=Path, default=PROJECT_ROOT / "data" / "bdd100k" / "videos" / "val")
    parser.add_argument("--train-labels", type=Path, default=PROJECT_ROOT / "data" / "bdd100k" / "labels" / "box_track_20" / "train")
    parser.add_argument("--val-flow", type=Path, default=PROJECT_ROOT / "runs" / "flyvis_fovea_flow")
    parser.add_argument("--val-root", type=Path, default=PROJECT_ROOT / "runs" / "video_rate")
    parser.add_argument("--val-labels", type=Path, default=PROJECT_ROOT / "data" / "bdd100k" / "labels" / "box_track_20" / "val")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "runs" / "motion_reliability")
    parser.add_argument("--positive-rate", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def rows_to_matrix(rows: list[dict], frame_size=(1280.0, 720.0)) -> tuple[np.ndarray, np.ndarray]:
    X, err = [], []
    w, h = frame_size
    for r in rows:
        pb = r["pred_box"]; bw, bh = pb[2] - pb[0], pb[3] - pb[1]
        X.append([bw / w, bh / h, bw * bh / (w * h), (pb[0] + pb[2]) / 2 / w, (pb[1] + pb[3]) / 2 / h, r["side"] / w,
                  float(np.linalg.norm(r["dis"] - r["pred"])) / r["scale"],
                  float(np.linalg.norm(r["delta"])), float(r["delta"][0]), float(r["delta"][1]), r["magnitude"], r["n_inside"] / 721.0])
        err.append(float(np.linalg.norm(r["gt"] - r["pred"])) / r["scale"])
    return np.asarray(X, np.float32), np.asarray(err, np.float64)


def collect_split(flow_root: Path, root: Path, labels: Path, video_ids: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    Xs, errs, vids = [], [], []
    ns = argparse.Namespace(flow_root=flow_root, root=root, labels=labels, min_iou=0.5)
    for k, v in enumerate(video_ids):
        if not (flow_root / v / "fovea_flow.npz").exists():
            continue  # кэш мухи для ролика ещё не посчитан
        rows = collect(v, ns)["rows"]
        if not rows:
            continue
        X, e = rows_to_matrix(rows)
        Xs.append(X); errs.append(e); vids.append(np.full(len(e), k, np.int32))
    return np.concatenate(Xs), np.concatenate(errs), np.concatenate(vids)


def video_luminance(video_roots: dict, selection: dict, val_ids: list[str], cache: Path) -> dict[str, float]:
    """Средняя яркость ролика (0..255) по 6 равномерно взятым кадрам; кэш в JSON."""
    import cv2

    values = json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else {}
    todo = [(v, video_roots["train"]) for s in ("train", "dev", "test") for v in selection[s]] + [(v, video_roots["val"]) for v in val_ids]
    for v, root in todo:
        if v in values:
            continue
        cap = cv2.VideoCapture(str(Path(root) / f"{v}.mov")); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); vals = []
        for t in np.linspace(0, max(n - 1, 0), 6).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(t)); ok, frame = cap.read()
            if ok:
                vals.append(float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean()))
        cap.release()
        values[v] = float(np.mean(vals)) if vals else float("nan")
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(values, indent=1), encoding="utf-8")
    return values


def main() -> int:
    args = parse_args()
    selection = json.loads((args.train_cache / "selection.json").read_text(encoding="utf-8"))["videos"]
    data = {}
    for split in ("train", "dev", "test"):
        data[split] = collect_split(args.train_flow or args.train_cache, args.train_cache, args.train_labels, selection[split]) + (selection[split],)
        print(f"[DATA] {split}: {len(data[split][1])} наблюдений на {len(selection[split])} видео", flush=True)
    val_ids = list(TRAIN_VIDEO_IDS + TEST_VIDEO_IDS)
    data["val7"] = collect_split(args.val_flow, args.val_root, args.val_labels, val_ids) + (val_ids,)
    print(f"[DATA] val7: {len(data['val7'][1])} наблюдений", flush=True)
    tau = float(np.quantile(data["train"][1], 1.0 - args.positive_rate))
    labels = {s: (d[1] > tau).astype(np.int8) for s, d in data.items()}
    print(f"[TAU] τ_e = {tau:.4f} (нормированная ошибка); доля положительных: " + ", ".join(f"{s} {labels[s].mean():.3f}" for s in labels), flush=True)
    rng = np.random.default_rng(args.seed)
    features = {}
    for s, d in data.items():
        X = d[0].copy(); sh = X[:, len(GEOMETRY) + len(DIS): len(GEOMETRY) + len(DIS) + len(FLY)].copy()
        for v in np.unique(d[2]):
            idx = np.flatnonzero(d[2] == v); sh[idx] = sh[rng.permutation(idx)]
        features[s] = np.concatenate([X, sh], axis=1)
    results = {}
    for name, cols in GROUPS.items():
        idx = [NAMES.index(c) for c in cols]
        model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000, random_state=42)).fit(features["train"][:, idx], labels["train"])
        threshold = f1_threshold(labels["train"], model.predict_proba(features["train"][:, idx])[:, 1])
        results[name] = {"features": list(cols)}
        for s in ("train", "dev", "test", "val7"):
            p = model.predict_proba(features[s][:, idx])[:, 1]; y = labels[s]
            results[name][s] = {"samples": int(len(y)), "positives": int(y.sum()), "average_precision": float(average_precision_score(y, p)), "brier": float(brier_score_loss(y, p)), "ece": ece_score(y, p)}
            by_video = {}
            for k, vid in enumerate(data[s][3]):
                m = data[s][2] == k
                if m.sum() > 20 and 0 < y[m].sum() < m.sum():
                    by_video[vid] = {"average_precision": float(average_precision_score(y[m], p[m])), "brier": float(brier_score_loss(y[m], p[m]))}
            results[name][f"{s}_by_video"] = by_video
        print(f"[FIT] {name:22s} " + " | ".join(f"{s}: AP={results[name][s]['average_precision']:.4f} Brier={results[name][s]['brier']:.4f} ECE={results[name][s]['ece']:.4f}" for s in ("train", "dev", "test", "val7")), flush=True)

    def boot(a: str, b: str, s: str) -> dict:
        wrap = lambda n: {"test_by_video": results[n][f"{s}_by_video"]}
        return {m: paired_video_bootstrap(wrap(a), wrap(b), m) for m in ("average_precision", "brier")}

    luminance = video_luminance({"train": args.train_videos, "val": args.val_videos}, selection, list(TRAIN_VIDEO_IDS + TEST_VIDEO_IDS), args.output / "luminance.json")
    strata = {}
    for group, test in (("dark", lambda L: L < args.dark_threshold), ("bright", lambda L: L >= args.dark_threshold)):
        vids = [v for s in ("dev", "test") for v in data[s][3] if test(luminance[v])]
        strata[group] = {"videos": len(vids)}
        for name in GROUPS:
            aps = [results[name][f"{s}_by_video"][v]["average_precision"] for s in ("dev", "test") for v in data[s][3] if v in results[name][f"{s}_by_video"] and test(luminance[v])]
            strata[group][name] = float(np.mean(aps)) if aps else None
        print(f"[STRATA] {group} ({len(vids)} видео dev+test): " + ", ".join(f"{n} {strata[group][n]:.4f}" for n in GROUPS if strata[group][n] is not None), flush=True)
    comparisons = {f"fly_vs_dis_{s}": boot("geometry_dis", "geometry_dis_fly", s) for s in ("dev", "test", "val7")}
    comparisons.update({f"fly_vs_shuffled_{s}": boot("geometry_dis_shuffled", "geometry_dis_fly", s) for s in ("test", "val7")})
    ap = lambda n, s: results[n][s]["average_precision"]; br = lambda n, s: results[n][s]["brier"]
    acceptance = {
        "ap_up_dev_test_val7": all(ap("geometry_dis_fly", s) > ap("geometry_dis", s) for s in ("dev", "test", "val7")),
        "brier_not_worse_dev_test_val7": all(br("geometry_dis_fly", s) <= br("geometry_dis", s) for s in ("dev", "test", "val7")),
        "beats_shuffled_test_val7": all(ap("geometry_dis_fly", s) > ap("geometry_dis_shuffled", s) for s in ("test", "val7")),
        "ece_not_worse_test": results["geometry_dis_fly"]["test"]["ece"] <= results["geometry_dis"]["test"]["ece"],
        "ci_excludes_zero_test": comparisons["fly_vs_dis_test"]["average_precision"]["ci95_percentile"][0] > 0,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    save_json(args.output / "report.json", {"strata": strata, "dark_threshold": args.dark_threshold, "protocol": {"event": f"нормированная ошибка центра предсказания > τ_e = {tau:.4f} (≈{args.positive_rate:.0%} положительных на train)", "model": "StandardScaler + LogisticRegression(C=1), порог F1 на train", "splits": {s: len(d[3]) for s, d in data.items()}}, "tau": tau, "positive_rate": {s: float(labels[s].mean()) for s in labels}, "groups": results, "comparisons": comparisons, "acceptance": acceptance})
    print(json.dumps(acceptance, ensure_ascii=False))
    print(f"[OK] {args.output / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
