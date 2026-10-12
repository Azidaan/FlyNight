"""Техническая проверка перевода движения глаза в координаты кадра (закрытие fly_persistent_eye_v1, 12.10).

Синтетика на реальном кадре BDD (dev), глаз и считывание — как в scripts/fly_persistent_eye.py (EyeBank, сетчатка-ламина,
decode_flow; среднее остаточного потока по колонкам внутри прогнозной рамки × сторона / размер ядра, ось y вниз):
  (а) объект движется, вырезка неподвижна: изображение сдвигается на (s·cos, s·sin) px за кадр, прогноз рамки постоянен —
      ожидается сигнал того же направления (знак и коэффициент k = |сигнал| / s);
  (б) движется вырезка, сцена неподвижна: прогноз рамки сдвигается на ту же величину за кадр — если сигнал ≈ −(а), сдвиг
      вырезки попадает в измерение потока (самодвижение глаза не учтено).
s = 2 px/кадр, 8 направлений, 12 кадров движения после разгона; усреднение по кадрам 4–12. Выход — runs/fly_eye_coord_check/report.json.

Запуск: python scripts/fly_eye_coord_check.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
OUT = PROJECT_ROOT / "runs/fly_eye_coord_check"


def main() -> int:
    from scripts.run_bdd_video_rate_experiment import make_motion_estimator
    from tracking.foveated_eyes import EyeBank, box_to_crop_frame
    from tracking.retina import make_retina
    vid = "019efe88-bbc2a4ac"
    cap = cv2.VideoCapture(str(PROJECT_ROOT / "data/bdd100k/videos/train" / f"{vid}.mov")); cap.set(cv2.CAP_PROP_POS_FRAMES, 300); ok, fr = cap.read(); cap.release()
    gray0 = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    fps = 30.0; s = 2.0; steps = 12
    est = make_motion_estimator("flyvis_tiled", fps, 0.02)
    res = {}
    for mode in ("object", "crop"):
        rows = []
        for ang in np.arange(8) * np.pi / 4:
            d = np.array([np.cos(ang), np.sin(ang)]) * s
            bank = EyeBank(est, max_batch=4, decode_flow=True); retina = make_retina("lamina", fps, blur_downscale=4)
            box = (640.0, 360.0, 120.0, 120.0)
            bank.birth(retina.process(gray0), {1: box})
            sig = []
            for k in range(1, steps + 1):
                if mode == "object":
                    M = np.float32([[1, 0, d[0] * k], [0, 1, d[1] * k]]); img = cv2.warpAffine(gray0, M, (gray0.shape[1], gray0.shape[0]), borderMode=cv2.BORDER_REFLECT); pb = box
                else:
                    img = gray0; pb = (box[0] + d[0] * k, box[1] + d[1] * k, box[2], box[3])
                bank.step(retina.process(img), {1: pb})
                crop = bank.crops[1]; side = crop[2] - crop[0]
                cx, cy, w, h = pb; loc = box_to_crop_frame((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), crop, bank.eye_size)
                px, py = bank.points_local[:, 0], bank.points_local[:, 1]
                ins = (px >= loc[0]) & (px <= loc[2]) & (py >= loc[1]) & (py <= loc[3])
                m = bank.last_flow(1)[ins].mean(axis=0); sc = side / est.kernel_area
                sig.append([m[0] * sc, -m[1] * sc])
            mean = np.mean(sig[3:], axis=0)
            rows.append({"angle_deg": float(np.degrees(ang)), "true": d.tolist(), "signal": mean.tolist(), "cos": float(mean @ d / (np.linalg.norm(mean) * s + 1e-12)), "k": float(np.linalg.norm(mean) / s)})
        res[mode] = {"rows": rows, "mean_cos": float(np.mean([r["cos"] for r in rows])), "mean_k": float(np.mean([r["k"] for r in rows]))}
        print(f"[{mode}] средний косинус с истинным направлением {res[mode]['mean_cos']:+.3f}, |сигнал| / s {res[mode]['mean_k']:.3f}", flush=True)
        for r in rows:
            print(f"   {r['angle_deg']:5.0f}°: истина {np.round(r['true'], 2)}, сигнал {np.round(r['signal'], 3)}, cos {r['cos']:+.2f}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True); (OUT / "report.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("[OK]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
