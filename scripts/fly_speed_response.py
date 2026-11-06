"""Зависит ли величина потока мухи от скорости (карточка fly_scale_probe_v1, дополнительная диагностика).

Тот же искусственный равномерный сдвиг кадров BDD dev, что в fly_scale_probe.py (сетка стенда 2×3 и 3×5,
каждый кадр). Скорость задаётся фиксированными уровнями 0,5 / 1 / 2 / 4 / 8 / 16 px/кадр; для каждой —
средняя проекция потока мухи (усиление 1) на истинное направление и средняя перпендикулярная составляющая, по колонкам.
Если проекция почти не растёт со скоростью, муха кодирует направление, но не скорость.

Запуск: python scripts/fly_speed_response.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import scripts.fly_finetune_k1 as K  # noqa: E402
from scripts.fly_scale_probe import make_fly, run_frames, WARM, LEN  # noqa: E402
from tracking.flyvis_geometry_protocol import add_geometry_arguments, ensure_geometry_output, resolve_output

OUT = PROJECT_ROOT / "runs" / "fly_scale_probe" / "speed_response.json"
SPEEDS = [0.5, 1, 2, 4, 8, 16]


def main() -> int:
    global OUT
    import argparse
    import cv2
    parser = argparse.ArgumentParser(); add_geometry_arguments(parser); args = parser.parse_args()
    OUT = resolve_output(args.out, OUT, PROJECT_ROOT)
    ensure_geometry_output(OUT, args.geometry_version, file=True)
    fps = json.loads((K.TC / "0001542f-5ce3cf52/motion/dis_quarter.json").read_text(encoding="utf-8"))["fps"]
    sel = json.loads((K.TC / "selection.json").read_text(encoding="utf-8"))["videos"]
    rng = np.random.default_rng(0)
    frames = []
    for v in sel["dev"][:8]:
        g = [x for _, x in zip(range(300), K.read_gray(PROJECT_ROOT / "data/bdd100k/videos/train" / f"{v}.mov"))]
        frames += [g[i] for i in rng.choice(len(g), 3, replace=False)]
    angs = rng.uniform(0, 2 * np.pi, len(frames))
    res = {}
    for lay in ("2x3", "3x5"):
        fly = make_fly(fps, lay, 1, geometry_version=args.geometry_version)
        res[lay] = {}
        for sp in SPEEDS:
            par, perp = [], []
            for g, a in zip(frames, angs):
                u = np.array([np.cos(a), np.sin(a)]); vel = u * sp
                seq = (fly.tiles(cv2.warpAffine(g, np.float32([[1, 0, vel[0] * j], [0, 1, vel[1] * j]]), (g.shape[1], g.shape[0]), borderMode=cv2.BORDER_REFLECT)) for j in range(WARM + LEN))
                for fl in run_frames(fly, seq)[WARM:]:
                    par.append(float((fl @ u).mean())); perp.append(float(np.abs(fl @ np.array([-u[1], u[0]])).mean()))
            res[lay][str(sp)] = {"parallel": float(np.mean(par)), "parallel_sd": float(np.std(par)), "perp_abs": float(np.mean(perp))}
            print(f"[SPEED] {lay} {sp:5.1f} px/кадр: проекция {np.mean(par):+.3f} ± {np.std(par):.3f}, |поперёк| {np.mean(perp):.3f}", flush=True)
        del fly
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[OK] {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
