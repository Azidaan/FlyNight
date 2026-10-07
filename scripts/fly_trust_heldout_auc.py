"""Различение правоты мухи на отложенных примерах (дополнение к fly_motion_trust_learned_v1).

В fly_motion_trust_learned_v1 точность модулей доверия приводилась на обучающих примерах. Здесь — на отложенных:
на 18 видео подбора и на 24 видео проверки (17 dev + val7) трекер схемы 5 Гц идёт с чистым Калманом и пишет признаки и
метку «поправка мухи ближе к эталону, чем Калман» (scripts/fly_trust_learned.py); модули обучаются на подборе, AUC и точность
считаются на проверке, по наборам. Сравнение: с признаками мухи / без них / только уверенность поля.
Если AUC с признаками мухи на отложенных не выше ≈ 0,6 и не выше варианта без признаков — устранять расхождение условий
обучения и применения бессмысленно (различать нечего).

Запуск: python scripts/fly_trust_heldout_auc.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import scripts.fly_trust_learned as L  # noqa: E402

OUT = PROJECT_ROOT / "runs/motion_scheme_search"


def collect(vids):
    data = []
    for v, root, sp in vids:
        before = len(data); L.run_video(v, root, sp, gate=None, collect=data)
        print(f"[COLLECT] {v}: +{len(data) - before}", flush=True)
    return np.array([x for x, _ in data]), np.array([y for _, y in data])


def main() -> int:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    split = json.loads((OUT / "trust_split.json").read_text(encoding="utf-8"))
    tc, vr = PROJECT_ROOT / "runs/train_cache", PROJECT_ROOT / "runs/video_rate"
    Xtr, ytr = collect([(v, tc, "train") for v in split["tune"]])
    sets = {"dev17": [(v, tc, "train") for v in split["test"]], "val7": [(v, vr, "val") for v in L.M.VAL7]}
    held = {k: collect(v) for k, v in sets.items()}
    held["all24"] = (np.concatenate([held["dev17"][0], held["val7"][0]]), np.concatenate([held["dev17"][1], held["val7"][1]]))
    res = {"n_train": int(len(ytr)), "pos_train": float(ytr.mean())}
    for name, cols in (("with_fly", L.FLY_FEATS), ("no_fly", L.NOFLY_FEATS), ("conf_only", [0])):
        mdl = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(Xtr[:, cols], ytr)
        res[name] = {"train_auc": float(roc_auc_score(ytr, mdl.predict_proba(Xtr[:, cols])[:, 1]))}
        for k, (X, y) in held.items():
            pr = mdl.predict_proba(X[:, cols])[:, 1]
            res[name][k] = {"auc": float(roc_auc_score(y, pr)), "acc": float(((pr > 0.5) == y).mean()), "majority_acc": float(max(y.mean(), 1 - y.mean())), "n": int(len(y)), "pos": float(y.mean())}
        print(f"[AUC] {name}: обучение {res[name]['train_auc']:.3f} | " + " | ".join(f"{k} AUC {res[name][k]['auc']:.3f} (точность {res[name][k]['acc']:.3f}, большинство {res[name][k]['majority_acc']:.3f})" for k in held), flush=True)
    (OUT / "trust_heldout_auc.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("[OK]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
