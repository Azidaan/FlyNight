"""Диагностика: нелинейная модель (HistGradientBoosting) на признаках гейта считывания FlyVis.

Не критерий карточек flyvis_internal_readout_v1 / flyvis_foveation_readout_v1
(там — логистическая регрессия по протоколу), а проверка возражения «линейная
модель слишком слаба, чтобы увидеть сигнал». Те же пары, то же разбиение.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, brier_score_loss

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_flyvis_qmatch_gate import BASE_FEATURES, FEATURE_NAMES, MOTION_FEATURES, TRAIN_VIDEO_IDS, TEST_VIDEO_IDS  # noqa: E402
from scripts.run_bdd_video_rate_experiment import save_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--fovea-features", type=Path, required=True, help="fovea_features.npz гейта фовеации")
    parser.add_argument("--tiled-features", type=Path, required=True, help="readout_features.npz tiled-гейта")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with np.load(args.pairs) as data:
        base = data["features"].astype(np.float32)
        targets = data["targets"].astype(np.int8)
        video = data["video_indices"]
        video_ids = [str(v) for v in data["video_ids"]]
    with np.load(args.fovea_features) as data:
        fovea_t4t5, fovea_all34, has_eye, fovea_shuffled = data["t4t5"], data["all34"], data["has_eye"], data["shuffled"]
    with np.load(args.tiled_features) as data:
        tiled_t4t5, tiled_all34 = data["t4t5"], data["all34"]
    names = list(FEATURE_NAMES)
    dis = [names.index(n) for n in MOTION_FEATURES[0:3]]
    decoded = [names.index(n) for n in MOTION_FEATURES[6:9]]
    base_idx = [names.index(n) for n in BASE_FEATURES]
    eye = has_eye[:, None].astype(np.float32)
    blocks = {
        "base_dis_eye": [base[:, base_idx], base[:, dis], eye],
        "base_dis_eye_decoded": [base[:, base_idx], base[:, dis], eye, base[:, decoded]],
        "base_dis_eye_fovea_t4t5": [base[:, base_idx], base[:, dis], eye, fovea_t4t5],
        "base_dis_eye_fovea_all34": [base[:, base_idx], base[:, dis], eye, fovea_all34],
        "base_dis_eye_shuffled_fovea": [base[:, base_idx], base[:, dis], eye, fovea_shuffled],
        "base_dis_eye_tiled_t4t5": [base[:, base_idx], base[:, dis], eye, tiled_t4t5],
        "base_dis_eye_tiled_all34": [base[:, base_idx], base[:, dis], eye, tiled_all34],
        "base_dis_eye_fovea_all34_decoded": [base[:, base_idx], base[:, dis], eye, fovea_all34, base[:, decoded]],
    }
    train = video < len(TRAIN_VIDEO_IDS)
    results = {}
    for name, parts in blocks.items():
        X = np.concatenate(parts, axis=1)
        model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_leaf_nodes=31, l2_regularization=1.0, random_state=args.seed)
        model.fit(X[train], targets[train])
        prob = model.predict_proba(X[~train])[:, 1]
        by_video = {}
        for index in range(len(TRAIN_VIDEO_IDS), len(video_ids)):
            mask = video[~train] == index
            by_video[video_ids[index]] = {"average_precision": float(average_precision_score(targets[~train][mask], prob[mask])), "brier": float(brier_score_loss(targets[~train][mask], prob[mask]))}
        results[name] = {"features": int(X.shape[1]), "test": {"average_precision": float(average_precision_score(targets[~train], prob)), "brier": float(brier_score_loss(targets[~train], prob))}, "test_by_video": by_video}
        print(f"[HGB] {name:34s} AP={results[name]['test']['average_precision']:.6f} Brier={results[name]['test']['brier']:.6f} by video {[round(v['average_precision'],5) for v in by_video.values()]}", flush=True)

    def delta(a: str, b: str) -> dict:
        return {v: results[b]["test_by_video"][v]["average_precision"] - results[a]["test_by_video"][v]["average_precision"] for v in results[a]["test_by_video"]}

    comparisons = {
        "fovea_t4t5_vs_base_dis_eye": delta("base_dis_eye", "base_dis_eye_fovea_t4t5"),
        "fovea_all34_vs_base_dis_eye": delta("base_dis_eye", "base_dis_eye_fovea_all34"),
        "fovea_all34_vs_shuffled": delta("base_dis_eye_shuffled_fovea", "base_dis_eye_fovea_all34"),
        "tiled_all34_vs_base_dis_eye": delta("base_dis_eye", "base_dis_eye_tiled_all34"),
        "decoded_vs_base_dis_eye": delta("base_dis_eye", "base_dis_eye_decoded"),
        "fovea_all34_decoded_vs_decoded": delta("base_dis_eye_decoded", "base_dis_eye_fovea_all34_decoded"),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"protocol": {"model": "HistGradientBoostingClassifier(max_iter=300, lr=0.08, leaves=31, l2=1.0)", "role": "диагностика, не критерий", "train_video_ids": list(TRAIN_VIDEO_IDS), "test_video_ids": list(TEST_VIDEO_IDS)}, "feature_groups": results, "per_video_ap_deltas": comparisons}
    save_json(args.output / "report.json", report)
    print(json.dumps(comparisons, ensure_ascii=False, indent=1))
    print(f"[OK] {args.output / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
