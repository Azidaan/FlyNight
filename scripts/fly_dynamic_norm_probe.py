"""Динамическая нормализация сигнала перед мухой против lcn24 (карточка fly_dynamic_norm_v1; план автора 06.10, шаг 5).

По мотивам Drews et al. (динамическое сжатие сигнала в путях перед детекторами движения мухи): усиление
контраста подстраивается во времени под оценку локального контраста. Реализация своя, параметры не из статьи:
  dyn — y = 0,5 + 0,2·(x − m) / (c_t + floor), где m — локальное среднее (σ = 24 px, мгновенно, как в lcn24),
        c_t — скользящее во времени (τ = 0,3 с) среднее |x − m|, размытое по большой области (σ = 48 px);
        floor = 0,03 (≈ 8 уровней яркости из 255) не даёт раздувать шум квантования бледных кадров.
Сравнение с lcn24 (scripts/fly_noise_probe.py) и raw. Искусственный сдвиг без отражённых краёв, протокол стенда
(fade-in → STATIC кадров покоя → LEN кадров сдвига, точка отсчёта — после fade-in, как в fly_input_norm_v1),
скорость 1 / 2 / 4 / 8 px/кадр × контраст ×0,25 / ×0,5 / ×1. Мера — ошибка при лучшем общем масштабе / нулевое
движение на колонку (масштаб на чётных отрезках всех контрастов, проверка на нечётных), отдельно по контрастам.

Запуск: python scripts/fly_dynamic_norm_probe.py
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

OUT = PROJECT_ROOT / "runs" / "fly_dynamic_norm"
SPEEDS = [1.0, 2.0, 4.0, 8.0]
CONTRASTS = [0.25, 0.5, 1.0]
STATIC, LEN, MARGIN = 30, 8, 24


class DynNorm:
    """Состояние живёт в пределах одного видео/отрезка."""

    def __init__(self, fps, sigma_m=24.0, sigma_c=48.0, tau=0.3, floor=0.03):
        self.sm, self.sc, self.floor = sigma_m, sigma_c, floor
        self.a = 1 - np.exp(-1.0 / (tau * fps))
        self.c = None

    def __call__(self, g):
        import cv2
        x = g.astype(np.float32) / 255.0
        m = cv2.GaussianBlur(x, (0, 0), self.sm)
        dev = cv2.GaussianBlur(np.abs(x - m), (0, 0), self.sc)
        self.c = dev if self.c is None else (1 - self.a) * self.c + self.a * dev
        y = 0.5 + 0.2 * (x - m) / (self.c + self.floor)
        return (np.clip(y, 0, 1) * 255).astype(np.float32)


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
    res = {}; t0 = time.time()
    for inp in ("raw", "lcn24", "dyn"):
        F, Y, seg, cts = [], [], [], []
        for ct in CONTRASTS:
            for sp in SPEEDS:
                for si, (g, a) in enumerate(zip(frames, angs)):
                    tf = (lambda im: im.astype(np.float32)) if inp == "raw" else (lcn(24.0) if inp == "lcn24" else DynNorm(fps))
                    u = np.array([np.cos(a), np.sin(a)]); vel = u * sp
                    m = float(g.mean()); gc = np.clip(m + ct * (g.astype(np.float32) - m), 0, 255).astype(np.uint8)
                    H, W = g.shape
                    seq = [gc] * (1 + STATIC) + [cv2.warpAffine(gc, np.float32([[1, 0, vel[0] * j], [0, 1, vel[1] * j]]), (W, H), borderMode=cv2.BORDER_CONSTANT) for j in range(1, LEN + 1)]
                    tiles = [fly.tiles(tf(x)) for x in seq]
                    pts, sc = fly.points_scale(); sct = t.tensor(sc, device=fly.dev)
                    end = pts - vel * LEN
                    ok = ((pts[:, 0] > MARGIN) & (pts[:, 0] < W - MARGIN) & (pts[:, 1] > MARGIN) & (pts[:, 1] < H - MARGIN)
                          & (end[:, 0] > MARGIN) & (end[:, 0] < W - MARGIN) & (end[:, 1] > MARGIN) & (end[:, 1] < H - MARGIN))
                    with t.no_grad():
                        cart = lambda x: t.from_numpy(x).to(fly.dev, t.float32).unsqueeze(1) / 255.0
                        cur = fly.render(cart(tiles[0])); state = fly.fade_in(cur); last = cur
                        base = dec(state.nodes.activity.unsqueeze(1))[:, 0]
                        for j in range(1, len(tiles)):
                            cur = fly.render(cart(tiles[j])); state, act = fly.step(state, last, cur); last = cur
                            if j > STATIC:
                                fl = K.to_px(dec(act).mean(1) - base, sct, t.zeros((), device=fly.dev)).cpu().numpy()[ok]
                                F.append(fl); Y.append(np.broadcast_to(vel, fl.shape)); seg.append(np.full(len(fl), si)); cts.append(np.full(len(fl), ct))
                print(f"[DYN] {inp} контраст {ct} скорость {sp}: готово | {time.time() - t0:.0f} с", flush=True)
                K.status(stage="dynamic_norm", input=inp, contrast=ct, speed=sp)
        F, Y, seg, cts = (np.concatenate(x) for x in (F, Y, seg, cts))
        fit, ev = seg % 2 == 0, seg % 2 == 1
        g = float((F[fit] * Y[fit]).sum() / (F[fit] ** 2).sum())
        ratio = lambda msk: float(np.linalg.norm(g * F[msk] - Y[msk], axis=1).mean() / np.linalg.norm(Y[msk], axis=1).mean())
        r = {"gain": g, "epe_ratio_col": ratio(ev)}
        for ct in CONTRASTS:
            r[f"epe_ratio_c{ct}"] = ratio(ev & (cts == ct))
        res[inp] = r
        print(f"[DYN] {inp}: ошибка/ноль {r['epe_ratio_col']:.3f} | по контрасту " + " ".join(f"{ct}:{r[f'epe_ratio_c{ct}']:.3f}" for ct in CONTRASTS) + f" | масштаб {g:.2f}", flush=True)
        (OUT / "synthetic.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[OK] {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
