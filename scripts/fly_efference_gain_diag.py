"""Диагностика после непройденной технической проверки fly_efference_eye_v1 (12.10): переносится ли калибровка между
изображениями. Новая версия (EfferenceEyeBank), C_new из runs/fly_efference_eye/calib.json; на 14 изображениях train
(12 калибровочных + 2 проверочных) сцена движется 2 px/кадр в 8 направлениях, глаз неподвижен, рамка 90 px, 12 кадров:
усиление = |Σ оценки| / путь, косинус; контраст — ст. отклонение яркости участка 240 × 240 вокруг рамки (после сетчатки
не берётся — исходный серый кадр). Только описание; постановку и критерии не меняет. Выход — runs/fly_efference_eye/gain_diag.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> int:
    from scipy.stats import spearmanr
    from scripts.fly_efference_eye import DIRS, episode, read_frame
    from scripts.run_bdd_video_rate_experiment import make_motion_estimator
    cal = json.loads((PROJECT_ROOT / "runs/fly_efference_eye/calib.json").read_text(encoding="utf-8"))
    C = {k: np.asarray(v) for k, v in cal["C"].items()}
    est = make_motion_estimator("flyvis_tiled", 30.0, 0.02)
    rows = []
    for v, fi in [tuple(x) for x in cal["calib_images"]] + [tuple(x) for x in cal["test_images"]]:
        img = read_frame(v, int(fi)); c = (640.0, 400.0)
        o = episode(est, img, c, 90.0, DIRS * 2.0, np.zeros((8, 2)), C=C)
        path = DIRS * 24.0
        S = o["new"]["est"].sum(1)
        gain = float(np.mean(np.linalg.norm(S, axis=1)) / 24.0); cos = float(np.mean([S[d] @ path[d] / (np.linalg.norm(S[d]) * 24 + 1e-9) for d in range(8)]))
        contrast = float(img[int(c[1]) - 120:int(c[1]) + 120, int(c[0]) - 120:int(c[0]) + 120].std()); mean = float(img[int(c[1]) - 120:int(c[1]) + 120, int(c[0]) - 120:int(c[0]) + 120].mean())
        rows.append({"image": f"{v}:{fi}", "gain": gain, "cos": cos, "contrast_std": contrast, "mean_brightness": mean})
        print(f"{v}:{fi} усиление {gain:.2f} косинус {cos:+.2f} контраст {contrast:.1f} яркость {mean:.0f}", flush=True)
    g = np.array([r["gain"] for r in rows]); k = np.array([r["contrast_std"] for r in rows])
    rep = {"rows": rows, "gain_min": float(g.min()), "gain_max": float(g.max()), "gain_cv": float(g.std() / g.mean()), "spearman_gain_contrast": float(spearmanr(g, k).correlation)}
    (PROJECT_ROOT / "runs/fly_efference_eye/gain_diag.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[DIAG] усиление {rep['gain_min']:.2f}–{rep['gain_max']:.2f}, коэффициент вариации {rep['gain_cv']:.2f}, Спирмен с контрастом {rep['spearman_gain_contrast']:+.2f}\n[OK]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
