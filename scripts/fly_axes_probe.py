"""Проверка осей потока мухи стенда: связан ли средний поток мухи с известным сдвигом кадра линейно (2×2).

Кадры BDD (dev) сдвигаются целиком с постоянной скоростью (как в fly_finetune_k1_synthetic.py), штатная муха
(декодер Sintel, преобразование стенда tracking/flyvis_motion.py:_points_and_vectors при усилении 1) даёт поле;
берётся среднее по центральным колонкам каждой плитки. Подгоняется v_true ≈ A · v_fly (МНК) и сообщается R²,
матрица A (поворот/отражение/масштаб) и корреляция по каждой оси. Отдельно — по сырым каналам декодера
до преобразования (на случай, если перепутаны каналы, а не знак).

Запуск: python scripts/fly_axes_probe.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import scripts.fly_finetune_k1 as K  # noqa: E402
from scripts.fly_finetune_k1_synthetic import segments  # noqa: E402

OUT = PROJECT_ROOT / "runs" / "fly_finetune_k1" / "axes_probe.json"


def main() -> int:
    fps = json.loads((K.TC / "0001542f-5ce3cf52/motion/dis_quarter.json").read_text(encoding="utf-8"))["fps"]
    fly = K.Fly(fps)
    t = fly.torch; dec = fly.base.decoder
    sel = json.loads((K.TC / "selection.json").read_text(encoding="utf-8"))["videos"]
    segs = segments(fly, sel["dev"][:10], "train", np.random.default_rng(0), 6)
    pts, sc = fly.points_scale()
    cy = fly.base._local_centers_yx
    central = np.flatnonzero(np.hypot(cy[:, 0], cy[:, 1]) < 0.6 * np.abs(cy).max())   # отступ от края плитки
    X, Xraw, Y, lag = [], [], [], []
    with t.no_grad():
        for frames, tgt in segs:
            cart = t.from_numpy(frames).to(fly.dev, t.float32).unsqueeze(2) / 255.0
            h0 = fly.render(cart[0]); st = fly.fade_in(h0); last = h0
            base = dec(st.nodes.activity.unsqueeze(1))[:, 0]
            for j in range(1, K.WARM + K.LEN):
                cur = fly.render(cart[j]); st, act = fly.step(st, last, cur); last = cur
                raw = (dec(act).mean(1) - base).cpu().numpy()                     # (6, 2, 721) каналы декодера
                if j >= K.WARM:
                    px = K.to_px(t.from_numpy(raw).to(fly.dev), t.tensor(sc, device=fly.dev) / fly.base.flow_gain, t.zeros((), device=fly.dev)).cpu().numpy()
                    px = px.reshape(6, 721, 2)[:, central].mean(1)                  # по плиткам
                    for k in range(6):
                        X.append(px[k]); Xraw.append(raw[k][:, central].mean(1)); Y.append(tgt[0, 0]); lag.append(j - K.WARM)
    X, Xraw, Y, lag = map(np.asarray, (X, Xraw, Y, lag))
    res = {}
    for name, F in (("stand_px", X), ("raw_channels", Xraw)):
        Fa = np.c_[F, np.ones(len(F))]
        A, *_ = np.linalg.lstsq(Fa, Y, rcond=None)
        pred = Fa @ A
        r2 = 1 - ((Y - pred) ** 2).sum(0) / ((Y - Y.mean(0)) ** 2).sum(0)
        corr = np.corrcoef(np.c_[F, Y].T)[:2, 2:]
        res[name] = {"A": A[:2].T.tolist(), "bias": A[2].tolist(), "R2_xy": r2.tolist(), "corr[fly_i, true_j]": corr.tolist()}
        print(f"[{name}] R² по x/y: {r2.round(3)} | матрица true≈A·fly: {np.round(A[:2].T, 3).tolist()} | corr(fly_i,true_j): {np.round(corr, 3).tolist()}", flush=True)
    # то же для поздних кадров отрезка (после переходного процесса)
    m = lag >= 6
    Fa = np.c_[X[m], np.ones(m.sum())]; A, *_ = np.linalg.lstsq(Fa, Y[m], rcond=None)
    r2 = 1 - ((Y[m] - Fa @ A) ** 2).sum(0) / ((Y[m] - Y[m].mean(0)) ** 2).sum(0)
    res["stand_px_late"] = {"A": A[:2].T.tolist(), "R2_xy": r2.tolist()}
    print(f"[stand_px, кадры 6–11 отрезка] R² {r2.round(3)} | A {np.round(A[:2].T, 3).tolist()}")
    res["n"] = int(len(Y))
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[OK] {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
