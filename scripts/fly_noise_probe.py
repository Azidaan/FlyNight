"""Источник шума в потоке мухи и проба нормализации контраста перед глазом (карточка fly_noise_probe_v1).

Искусственный равномерный сдвиг кадров BDD dev (как fly_scale_probe.py), сетка стенда 2×3, каждый кадр, штатная муха 000.
Скорости 0 (неподвижный кадр) / 2 / 4 px/кадр. Варианты входа (без обучения):
  raw    — штатный gray/255;
  lcn8   — нормализация локального контраста: (x − blur(x)) / (std_loc + ε), σ = 8 px кадра, отображение 0,5 ± 0,2·z (обрезка 0–1);
  lcn24  — то же, σ = 24 px;
  retina — фоторецептор и ламина проекта (tracking/retina.py, режим lamina, параметры по умолчанию).
Мера: проекция потока (усиление 1) на истинное направление (сигнал), её разброс и |поперёк| (шум), ошибка при лучшем общем
масштабе / нулевое движение на колонку (масштаб на чётных отрезках, проверка на нечётных; по скоростям 2 и 4).

Запуск: python scripts/fly_noise_probe.py [--warm 60 --variants raw lcn24]
  --warm — кадров разгона после fade-in до замера (не переходный ли процесс сети даёт шум)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import scripts.fly_finetune_k1 as K  # noqa: E402
from scripts.fly_scale_probe import make_fly, run_frames, LEN  # noqa: E402
from tracking.flyvis_geometry_protocol import add_geometry_arguments, ensure_geometry_output, resolve_output

OUT = PROJECT_ROOT / "runs" / "fly_scale_probe" / "noise_probe.json"
SPEEDS = [0.0, 2.0, 4.0]


def lcn(sigma):
    import cv2
    def f(g):
        x = g.astype(np.float32) / 255.0
        m = cv2.GaussianBlur(x, (0, 0), sigma)
        sd = np.sqrt(np.maximum(cv2.GaussianBlur(x * x, (0, 0), sigma) - m * m, 0))
        z = (x - m) / (sd + 0.02)
        return (np.clip(0.5 + 0.2 * z, 0, 1) * 255).astype(np.float32)
    return f


def main() -> int:
    import argparse
    import cv2
    ap = argparse.ArgumentParser(); ap.add_argument("--warm", type=int, default=4); ap.add_argument("--variants", nargs="+", default=["raw", "lcn8", "lcn24", "retina"])
    add_geometry_arguments(ap)
    a = ap.parse_args(); WARM = a.warm
    out = resolve_output(a.out, OUT if WARM == 4 else OUT.with_name(f"noise_probe_warm{WARM}.json"), PROJECT_ROOT)
    ensure_geometry_output(out, a.geometry_version, file=True)
    from tracking.retina import make_retina
    fps = json.loads((K.TC / "0001542f-5ce3cf52/motion/dis_quarter.json").read_text(encoding="utf-8"))["fps"]
    sel = json.loads((K.TC / "selection.json").read_text(encoding="utf-8"))["videos"]
    rng = np.random.default_rng(0)
    frames = []
    for v in sel["dev"][:8]:
        g = [x for _, x in zip(range(300), K.read_gray(PROJECT_ROOT / "data/bdd100k/videos/train" / f"{v}.mov"))]
        frames += [g[i] for i in rng.choice(len(g), 4, replace=False)]
    angs = rng.uniform(0, 2 * np.pi, len(frames))
    fly = make_fly(fps, "2x3", 1, geometry_version=a.geometry_version)
    res = {}
    for name in a.variants:
        res[name] = {}
        F, Y, seg = [], [], []
        for sp in SPEEDS:
            par, perp = [], []
            for si, (g, a) in enumerate(zip(frames, angs)):
                u = np.array([np.cos(a), np.sin(a)]); vel = u * sp
                if name == "retina":
                    ret = make_retina("lamina", fps)
                    tf = lambda im: ret.process(im).astype(np.float32)   # уже 0–255, с состоянием адаптации
                elif name == "raw":
                    tf = lambda im: im.astype(np.float32)
                else:
                    tf = lcn(8.0 if name == "lcn8" else 24.0)
                seq = []
                for j in range(WARM + LEN):
                    w = cv2.warpAffine(g, np.float32([[1, 0, vel[0] * j], [0, 1, vel[1] * j]]), (g.shape[1], g.shape[0]), borderMode=cv2.BORDER_REFLECT)
                    seq.append(fly.tiles(tf(w)))
                for fl in run_frames(fly, iter(seq))[WARM:]:
                    par.append(float((fl @ u).mean())); perp.append(float(np.abs(fl @ np.array([-u[1], u[0]])).mean()))
                    if sp > 0:
                        F.append(fl); Y.append(np.broadcast_to(vel, fl.shape)); seg.append(si)
            res[name][str(sp)] = {"parallel": float(np.mean(par)), "parallel_sd": float(np.std(par)), "perp_abs": float(np.mean(perp))}
            print(f"[NOISE] {name:6s} {sp:3.1f} px/кадр: сигнал {np.mean(par):+.3f} ± {np.std(par):.3f}, |поперёк| {np.mean(perp):.3f}", flush=True)
        F, Y, seg = np.stack(F), np.stack(Y), np.asarray(seg)
        fit, ev = seg % 2 == 0, seg % 2 == 1
        gk = float((F[fit] * Y[fit]).sum() / (F[fit] ** 2).sum())
        ratio = float(np.linalg.norm(gk * F[ev] - Y[ev], axis=-1).mean() / np.linalg.norm(Y[ev], axis=-1).mean())
        res[name]["epe_ratio_col"] = ratio; res[name]["gain"] = gk
        print(f"[NOISE] {name}: ошибка/ноль на колонку {ratio:.3f} (масштаб {gk:.2f})", flush=True)
        out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[OK] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
