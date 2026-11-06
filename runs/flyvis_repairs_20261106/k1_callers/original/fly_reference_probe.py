"""Точка отсчёта движения мухи (карточка fly_reference_point_v1).

В стенде (tracking/flyvis_motion.py) и в К1 из выхода декодера вычитается отклик сразу после fade-in — а fade-in
не установившееся состояние (контраст первого кадра нарастает от 0 за 1 с). Сравниваются точки отсчёта:
  A fadein  — как в стенде: dec(t) − dec(сразу после fade-in);
  B absolute — dec(t) без вычитания;
  C static  — dec(t) − средний dec за последние 5 кадров неподвижной сцены после 2 с покоя (идеальная проверка);
  D ema     — dec(t) − скользящее среднее dec с τ = 1 с (до текущего кадра; применимо к живому видео).
Протокол один для всех: fade-in на первом кадре → STATIC кадров неподвижной сцены → LEN кадров равномерного сдвига.
Сдвиг — cv2.warpAffine с BORDER_CONSTANT; учитываются только колонки, у которых весь путь изображения за отрезок
лежит внутри исходного кадра с запасом MARGIN px (отражённых и пустых краёв нет). Входы raw и lcn24
(scripts/fly_noise_probe.py), сетка стенда 2×3, каждый кадр, штатная муха 000, усиление 1.
Мера на точку отсчёта: ложный поток на неподвижной сцене (разброс проекции на скорости 0), наклон отклика
(проекция / скорость), шум, ошибка при лучшем общем масштабе / нулевое (масштаб на чётных отрезках, проверка на
нечётных; скорости > 0).

Запуск: python scripts/fly_reference_probe.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import scripts.fly_finetune_k1 as K  # noqa: E402
from scripts.fly_noise_probe import lcn  # noqa: E402
from scripts.fly_scale_probe import make_fly  # noqa: E402

OUT = PROJECT_ROOT / "runs" / "fly_reference_probe"
SPEEDS = [0.0, 1.0, 2.0, 4.0, 8.0]
STATIC, LEN = 60, 8
MARGIN = 24
REFS = ("fadein", "absolute", "static", "ema")


def main() -> int:
    import cv2
    OUT.mkdir(parents=True, exist_ok=True)
    fps = json.loads((K.TC / "0001542f-5ce3cf52/motion/dis_quarter.json").read_text(encoding="utf-8"))["fps"]
    sel = json.loads((K.TC / "selection.json").read_text(encoding="utf-8"))["videos"]
    rng = np.random.default_rng(0)
    frames = []
    for v in sel["dev"][:8]:
        g = [x for _, x in zip(range(300), K.read_gray(PROJECT_ROOT / "data/bdd100k/videos/train" / f"{v}.mov"))]
        frames += [g[i] for i in rng.choice(len(g), 3, replace=False)]
    angs = rng.uniform(0, 2 * np.pi, len(frames))
    fly = make_fly(fps, "2x3", 1)
    t = fly.torch; dec = fly.base.decoder
    a_ema = 1 - np.exp(-1.0 / (1.0 * fps))
    res = {}
    t0 = time.time()
    for inp in ("raw", "lcn24"):
        tf = (lambda g: g.astype(np.float32)) if inp == "raw" else lcn(24.0)
        res[inp] = {}
        store = {r: {"F": [], "Y": [], "seg": [], "sp": []} for r in REFS}
        for sp in SPEEDS:
            for si, (g, a) in enumerate(zip(frames, angs)):
                u = np.array([np.cos(a), np.sin(a)]); vel = u * sp
                H, W = g.shape
                seq = [g] * (1 + STATIC) + [cv2.warpAffine(g, np.float32([[1, 0, vel[0] * j], [0, 1, vel[1] * j]]), (W, H), borderMode=cv2.BORDER_CONSTANT) for j in range(1, LEN + 1)]
                tiles = [fly.tiles(tf(x)) for x in seq]
                pts, sc = fly.points_scale(); sct = t.tensor(sc, device=fly.dev)
                # колонка годна, если источник изображения на всём пути внутри кадра с запасом
                src_end = pts - vel * LEN
                ok = ((pts[:, 0] > MARGIN) & (pts[:, 0] < W - MARGIN) & (pts[:, 1] > MARGIN) & (pts[:, 1] < H - MARGIN)
                      & (src_end[:, 0] > MARGIN) & (src_end[:, 0] < W - MARGIN) & (src_end[:, 1] > MARGIN) & (src_end[:, 1] < H - MARGIN))
                with t.no_grad():
                    cart = lambda x: t.from_numpy(x).to(fly.dev, t.float32).unsqueeze(1) / 255.0
                    cur = fly.render(cart(tiles[0])); state = fly.fade_in(cur); last = cur
                    d_fade = dec(state.nodes.activity.unsqueeze(1))[:, 0]
                    ema = d_fade.clone(); static_hist = []
                    outs = {r: [] for r in REFS}
                    for j in range(1, len(tiles)):
                        cur = fly.render(cart(tiles[j])); state, act = fly.step(state, last, cur); last = cur
                        d = dec(act).mean(1)
                        if j <= STATIC:
                            if j > STATIC - 5:
                                static_hist.append(d)
                            ema = (1 - a_ema) * ema + a_ema * d
                            continue
                        d_static = t.stack(static_hist).mean(0)
                        base = {"fadein": d_fade, "absolute": t.zeros_like(d), "static": d_static, "ema": ema}
                        for r in REFS:
                            outs[r].append(K.to_px(d - base[r], sct, t.zeros((), device=fly.dev)).cpu().numpy()[ok])
                        ema = (1 - a_ema) * ema + a_ema * d
                for r in REFS:
                    for fl in outs[r]:
                        store[r]["F"].append(fl); store[r]["Y"].append(np.broadcast_to(vel, fl.shape)); store[r]["seg"].append(si); store[r]["sp"].append(sp)
            print(f"[REF] {inp} скорость {sp}: готово | {time.time() - t0:.0f} с", flush=True)
            K.status(stage="reference", input=inp, speed=sp)
        for r in REFS:
            S = store[r]
            per = {}
            for sp in SPEEDS:
                idx = [i for i, x in enumerate(S["sp"]) if x == sp]
                if not idx:
                    continue
                par = np.concatenate([S["F"][i] @ (S["Y"][i][0] / max(sp, 1e-9)) if sp > 0 else S["F"][i][:, 0] for i in idx])
                if sp == 0:
                    allv = np.concatenate([S["F"][i] for i in idx])
                    per["0.0"] = {"false_flow_mean_abs": float(np.linalg.norm(allv, axis=1).mean()), "false_flow_sd": float(allv.std())}
                else:
                    per[str(sp)] = {"parallel": float(par.mean()), "parallel_sd": float(par.std()), "slope": float(par.mean() / sp)}
            mv = [i for i, x in enumerate(S["sp"]) if x > 0]
            F = np.concatenate([S["F"][i] for i in mv]); Y = np.concatenate([S["Y"][i] for i in mv])
            seg = np.concatenate([np.full(len(S["F"][i]), S["seg"][i]) for i in mv])
            fit, ev = seg % 2 == 0, seg % 2 == 1
            g = float((F[fit] * Y[fit]).sum() / (F[fit] ** 2).sum())
            per["epe_ratio_col"] = float(np.linalg.norm(g * F[ev] - Y[ev], axis=1).mean() / np.linalg.norm(Y[ev], axis=1).mean())
            per["gain"] = g
            res[inp][r] = per
            print(f"[REF] {inp} {r:8s}: ложный поток {per['0.0']['false_flow_mean_abs']:.3f} px | наклон " +
                  " ".join(f"{sp:g}:{per[str(sp)]['slope']:.3f}" for sp in SPEEDS if sp > 0) +
                  f" | ошибка/ноль {per['epe_ratio_col']:.3f} (масштаб {g:.2f})", flush=True)
        (OUT / "reference.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[OK] {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
