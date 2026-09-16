"""Головы событийной уверенности без FlyVis (цель 4): ранжирование, калибровка, рабочая точка.

Для каждого события — лестница: контроль (монотонное преобразование исходной
оценки), base (все признаки без DIS), base + DIS. Обучение и порог — только на
обучающих видео; метрики — на отложенных, повидео, поклассово и по стратам.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, precision_recall_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_flyvis_qmatch_gate import TEST_VIDEO_IDS, TRAIN_VIDEO_IDS, paired_video_bootstrap  # noqa: E402
from scripts.run_bdd_video_rate_experiment import CLASS_NAMES, save_json  # noqa: E402

EVENTS = ("q_det", "q_match", "q_birth", "q_alive")
CONTROL_FEATURES = {"q_det": ("conf",), "q_match": ("pair_iou", "det_conf"), "q_birth": ("conf",), "q_alive": ("track_age",)}
RAW_SCORE = {"q_det": "conf", "q_match": "pair_iou", "q_birth": "conf", "q_alive": "track_hits"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "runs" / "event_dataset")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "runs" / "event_heads")
    parser.add_argument("--min-class-support", type=int, default=20)
    return parser.parse_args()


def ece_score(target: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    if len(target) < bins:
        return float("nan")
    order = np.argsort(probability)
    chunks = np.array_split(order, bins)
    total = 0.0
    for chunk in chunks:
        if not len(chunk):
            continue
        total += len(chunk) / len(target) * abs(probability[chunk].mean() - target[chunk].mean())
    return float(total)


def f1_threshold(target: np.ndarray, probability: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(target, probability)
    if not len(thresholds):
        return 0.5
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[int(np.argmax(f1))])


def metrics(target: np.ndarray, probability: np.ndarray, threshold: float | None = None) -> dict:
    out = {"samples": int(len(target)), "positives": int(target.sum())}
    if len(target) == 0 or target.min() == target.max():
        out.update({"average_precision": None, "brier": None, "log_loss": None, "ece": None})
        return out
    out.update({
        "average_precision": float(average_precision_score(target, probability)),
        "brier": float(brier_score_loss(target, probability)),
        "log_loss": float(log_loss(target, np.clip(probability, 1e-6, 1 - 1e-6))),
        "ece": ece_score(target, probability),
    })
    if threshold is not None:
        decided = probability >= threshold
        tp = int((decided & (target == 1)).sum()); fp = int((decided & (target == 0)).sum()); fn = int((~decided & (target == 1)).sum())
        out.update({"threshold": threshold, "tp": tp, "fp": fp, "fn": fn, "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1)})
    return out


def load_event(dataset: Path, event: str, video_ids: list[str]) -> dict:
    features, meta, video = [], [], []
    names = None
    for index, video_id in enumerate(video_ids):
        with np.load(dataset / video_id / f"{event}.npz") as data:
            names = [str(n) for n in data["feature_names"]]
            meta_names = [str(n) for n in data["meta_names"]]
            features.append(data["features"]); meta.append(data["meta"]); video.append(np.full(len(data["meta"]), index, np.int16))
    return {"features": np.concatenate(features), "meta": np.concatenate(meta), "video": np.concatenate(video), "names": names, "meta_names": meta_names}


def fit_group(data: dict, feature_names: tuple[str, ...], train_count: int, video_ids: list[str], *, min_class_support: int) -> dict:
    cols = [data["names"].index(n) for n in feature_names]
    X = data["features"][:, cols]
    y = data["meta"][:, data["meta_names"].index("label")].astype(np.int8)
    classes = data["meta"][:, data["meta_names"].index("class_id")]
    train = data["video"] < train_count
    test = ~train
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000, random_state=42))
    model.fit(X[train], y[train])
    p_train = model.predict_proba(X[train])[:, 1]
    threshold = f1_threshold(y[train], p_train)
    p_test = model.predict_proba(X[test])[:, 1]
    result = {"features": list(feature_names), "train": metrics(y[train], p_train, threshold), "test": metrics(y[test], p_test, threshold), "test_by_video": {}, "test_by_class": {}, "test_by_stratum": {}}
    for index in range(train_count, len(video_ids)):
        mask = data["video"][test] == index
        result["test_by_video"][video_ids[index]] = metrics(y[test][mask], p_test[mask], threshold)
    for c in sorted(set(classes[test].tolist())):
        mask = classes[test] == c
        if y[test][mask].sum() >= min_class_support and (y[test][mask] == 0).sum() >= min_class_support:
            result["test_by_class"][CLASS_NAMES[int(c)]] = metrics(y[test][mask], p_test[mask], threshold)
    for stratum in ("occluded", "det_unmatched", "track_lost"):
        if stratum in data["meta_names"]:
            flag = data["meta"][:, data["meta_names"].index(stratum)][test] == 1
        elif stratum in data["names"]:
            flag = data["features"][:, data["names"].index(stratum)][test] == 1
        else:
            continue
        result["test_by_stratum"][stratum] = {"true": metrics(y[test][flag], p_test[flag], threshold), "false": metrics(y[test][~flag], p_test[~flag], threshold)}
    raw = data["features"][:, data["names"].index(RAW_SCORE_FOR[data["event"]])][test]
    result["spearman_vs_raw_score"] = float(spearmanr(raw, p_test).correlation) if len(raw) > 2 else None
    result["_probability_test"] = p_test
    return result


RAW_SCORE_FOR = RAW_SCORE


def main() -> int:
    args = parse_args()
    video_ids = list(TRAIN_VIDEO_IDS + TEST_VIDEO_IDS)
    summary = json.loads((args.dataset / "summary.json").read_text(encoding="utf-8"))
    report = {"protocol": {"train_video_ids": list(TRAIN_VIDEO_IDS), "test_video_ids": list(TEST_VIDEO_IDS), "model": "StandardScaler + LogisticRegression(C=1, max_iter=3000); порог по F1 на обучающих видео", "ece": "10 равночастотных корзин", "dataset_rules": summary["rules"]}, "events": {}}
    for event in EVENTS:
        data = load_event(args.dataset, event, video_ids); data["event"] = event
        all_names = tuple(data["names"]); dis = tuple(summary["dis_feature_names"][event])
        groups = {"control": CONTROL_FEATURES[event], "base": tuple(n for n in all_names if n not in dis), "base_dis": all_names}
        results = {}
        for name, feats in groups.items():
            print(f"[FIT] {event}/{name} ({len(feats)} признаков)", flush=True)
            results[name] = fit_group(data, feats, len(TRAIN_VIDEO_IDS), video_ids, min_class_support=args.min_class_support)
        comparisons = {}
        for a, b in (("control", "base"), ("base", "base_dis"), ("control", "base_dis")):
            comparisons[f"{b}_vs_{a}"] = {m: paired_video_bootstrap(results[a], results[b], m) for m in ("average_precision", "brier")}
        y_test = data["meta"][:, data["meta_names"].index("label")][data["video"] >= len(TRAIN_VIDEO_IDS)]
        report["events"][event] = {
            "units": int(len(data["meta"])), "positives": int(data["meta"][:, data["meta_names"].index("label")].sum()),
            "test_units": int(len(y_test)), "test_positives": int(y_test.sum()),
            "groups": {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in results.items()},
            "comparisons": comparisons,
        }
        line = " | ".join(f"{k}: AP={v['test']['average_precision']:.4f} Brier={v['test']['brier']:.4f} ECE={v['test']['ece']:.4f}" if v['test']['average_precision'] is not None else f"{k}: n/a" for k, v in results.items())
        print(f"[{event}] test {len(y_test)} ({int(y_test.sum())}+) :: {line}", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    save_json(args.output / "report.json", report)
    print(f"[OK] {args.output / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
