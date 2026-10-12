"""fly_efference_eye_v1: калибровка 2 × 2 и техническая проверка (подкоманда calib), ступень 1 (подкоманда stage1).

Варианты считывания движения объекта (px кадра за кадр):
  old  — действующий глаз (EyeBank): сырой сигнал = движение изображения в вырезке; движение объекта = C_old · сырой
         + известное перемещение центра вырезки за кадр (эфферентная копия на выходе, вне FlyVis);
  new  — EfferenceEyeBank (перенос состояния и прошлого стимула в текущую вырезку внутри пути мухи); движение объекта = C_new · сырой;
  ctrl — контроль без FlyVis: та же геометрия (прошлый и текущий кадр после той же сетчатки в текущей вырезке, 128 × 128),
         классический поток DIS (OpenCV), среднее в рамке × сторона / 128; движение объекта = C_ctrl · сырой.
Сырой сигнал глаза — среднее остаточного декодированного потока по колонкам внутри рамки × сторона вырезки / размер ядра,
ось y вниз (как fly_persistent_eye). Калибровка (только train, зафиксирована до dev): 12 изображений (6 видео train, seed 0,
кадры 150 и 600), рамки 60 и 120 px, скорости 1, 2, 4 px/кадр, 8 направлений, сцена движется, глаз неподвижен; шаги 4–12;
C — МНК (истинное движение ≈ C · сырой). Техническая проверка (2 других кадра train, 2 px/кадр, 8 направлений, 12 кадров):
  T1 сцена неподвижна, глаз движется (сдвиг; сдвиг + рост рамки 1 %/кадр): накопленный перенос объекта |Σ оценка| ≤ 0,25 пути глаза;
  T2 сцена движется, глаз неподвижен: косинус Σ оценки с истиной ≥ 0,8, отношение длин 0,6–1,4;
  T3 сцена движется (направление A), глаз движется (A + 90°): те же пороги относительно движения сцены;
  T4 последовательные поправки: сцена движется, оценка c_t = c_{t−1} + оценка движения, вырезка — у c_{t−1}; ошибка через
     12 кадров ≤ 0,5 ошибки без поправки. Условие запуска ступени 1 — new проходит T1–T4 (средние по направлениям).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

OUT = PROJECT_ROOT / "runs/fly_efference_eye"
PATCH = 360
DIRS = np.array([[np.cos(a), np.sin(a)] for a in np.arange(8) * np.pi / 4])


def read_frame(vid, idx):
    cap = cv2.VideoCapture(str(PROJECT_ROOT / "data/bdd100k/videos/train" / f"{vid}.mov")); cap.set(cv2.CAP_PROP_POS_FRAMES, idx); ok, fr = cap.read(); cap.release()
    return cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).astype(np.float32)


def patch(img, center, shift):
    """Участок PATCH × PATCH вокруг center, сцена сдвинута на shift (px)."""
    M = np.float32([[1, 0, shift[0] - center[0] + PATCH / 2], [0, 1, shift[1] - center[1] + PATCH / 2]])
    return cv2.warpAffine(img, M, (PATCH, PATCH), borderMode=cv2.BORDER_REFLECT)


class Ctrl:
    """Контроль без FlyVis: классический поток на тех же стабилизированных вырезках."""
    def __init__(self):
        self.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM); self.prev = None

    def raw(self, img, crop, box):
        x1, y1, x2, y2 = crop; side = x2 - x1
        def cut(g):
            return cv2.resize(np.clip(g[y1:y2, x1:x2], 0, 255).astype(np.uint8), (128, 128), interpolation=cv2.INTER_AREA)
        if self.prev is None:
            return np.zeros(2)
        f = self.dis.calc(cut(self.prev), cut(img), None)
        cx, cy, w, h = box; s = 128 / side
        bx1, by1, bx2, by2 = int((cx - w / 2 - x1) * s), int((cy - h / 2 - y1) * s), int((cx + w / 2 - x1) * s), int((cy + h / 2 - y1) * s)
        region = f[max(by1, 0):max(by2, by1 + 1), max(bx1, 0):max(bx2, bx1 + 1)].reshape(-1, 2)
        return region.mean(0) * side / 128 if len(region) else np.zeros(2)


def eye_raw(bank, est, tid, box):
    from tracking.foveated_eyes import box_to_crop_frame
    crop = bank.crops[tid]; side = crop[2] - crop[0]
    cx, cy, w, h = box; loc = box_to_crop_frame((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), crop, bank.eye_size)
    px, py = bank.points_local[:, 0], bank.points_local[:, 1]
    ins = (px >= loc[0]) & (px <= loc[2]) & (py >= loc[1]) & (py <= loc[3])
    if ins.sum() < 3:
        return np.zeros(2)
    m = bank.last_flow(tid)[ins].mean(axis=0); sc = side / est.kernel_area
    return np.array([m[0] * sc, -m[1] * sc])


def episode(est, img, center, size, scene_v, eye_v, grow=0.0, steps=12, C=None, closed=False):
    """8 направлений пакетом. scene_v, eye_v — (8, 2) px/кадр. Возвращает по варианту массивы (8, steps, 2): сырой и оценку."""
    from tracking.efference_eyes import EfferenceEyeBank
    from tracking.foveated_eyes import EyeBank, crop_box
    from tracking.retina import make_retina
    canvas_w = PATCH * 8
    out = {}
    for var in ("old", "new", "ctrl"):
        bank = None if var == "ctrl" else (EyeBank if var == "old" else EfferenceEyeBank)(est, max_batch=8, decode_flow=True)
        ctr = [Ctrl() for _ in range(8)]
        retina = make_retina("lamina", 30.0, blur_downscale=4)
        def frame_at(k):
            return np.concatenate([patch(img, center, scene_v[d] * k) for d in range(8)], axis=1)
        base = [np.array([PATCH * d + PATCH / 2, PATCH / 2]) for d in range(8)]
        est_pos = [b.copy() for b in base]   # для T4: оценка положения объекта
        boxes = {d + 1: (base[d][0], base[d][1], size, size) for d in range(8)}
        f0 = retina.process(frame_at(0))
        if bank is not None:
            bank.birth(f0, boxes)
        prev_img = f0; prev_center = {d: base[d].copy() for d in range(8)}
        raws, ests = np.zeros((8, steps, 2)), np.zeros((8, steps, 2))
        for d in range(8):
            ctr[d].prev = f0
        for k in range(1, steps + 1):
            img_k = retina.process(frame_at(k))
            preds = {}
            for d in range(8):
                if closed:
                    c = est_pos[d]
                else:
                    c = base[d] + eye_v[d] * k
                sz = size * (1 + grow) ** k
                preds[d + 1] = (float(c[0]), float(c[1]), sz, sz)
            if bank is not None:
                bank.step(img_k, preds)
            for d in range(8):
                box = preds[d + 1]; cc = np.array(box[:2])
                if var == "ctrl":
                    crop = crop_box(box, (canvas_w, PATCH)); r = ctr[d].raw(img_k, crop, box); ctr[d].prev = img_k
                else:
                    r = eye_raw(bank, est, d + 1, box)
                raws[d, k - 1] = r
                if C is not None:
                    v = C[var] @ r
                    if var == "old":
                        v = v + (cc - prev_center[d])   # эфферентная копия на выходе: известное перемещение вырезки
                    ests[d, k - 1] = v
                    if closed:
                        est_pos[d] = est_pos[d] + v
                prev_center[d] = cc
            prev_img = img_k
        out[var] = {"raw": raws, "est": ests}
    return out


def calib(a) -> int:
    from scripts.run_bdd_video_rate_experiment import make_motion_estimator
    OUT.mkdir(parents=True, exist_ok=True); t0 = time.time()
    sel = json.loads((PROJECT_ROOT / "runs/train_cache/selection.json").read_text(encoding="utf-8"))["videos"]["train"]
    vids = list(np.random.default_rng(0).choice(sel, 8, replace=False))
    calib_imgs = [(v, f) for v in vids[:6] for f in (150, 600)]; test_imgs = [(vids[6], 900), (vids[7], 900)]
    est = make_motion_estimator("flyvis_tiled", 30.0, 0.02)
    rows = {v: [] for v in ("old", "new", "ctrl")}
    for i, (v, fi) in enumerate(calib_imgs):
        img = read_frame(v, fi); center = (640.0, 400.0)
        for size in (60.0, 120.0):
            for sp in (1.0, 2.0, 4.0):
                o = episode(est, img, center, size, DIRS * sp, np.zeros((8, 2)))
                for var in rows:
                    for d in range(8):
                        for k in range(3, 12):
                            rows[var].append(np.concatenate([o[var]["raw"][d, k], DIRS[d] * sp]))
        print(f"[CALIB] {i + 1}/{len(calib_imgs)} {v}:{fi} | {(time.time() - t0) / 60:.1f} мин", flush=True)
    C, fitq = {}, {}
    for var, rr in rows.items():
        R = np.asarray(rr); X, Y = R[:, :2], R[:, 2:]
        Ct, *_ = np.linalg.lstsq(X, Y, rcond=None); C[var] = Ct.T
        err = np.linalg.norm(Y - X @ Ct, axis=1); fitq[var] = {"median_err_px": float(np.median(err)), "median_true_px": float(np.median(np.linalg.norm(Y, axis=1))), "C": C[var].tolist(), "n": len(R)}
        print(f"[CALIB] {var}: C = {np.round(C[var], 3).tolist()}, медиана ошибки {fitq[var]['median_err_px']:.2f} px при медиане движения {fitq[var]['median_true_px']:.2f}", flush=True)
    # техническая проверка
    checks = {}
    def summ(est_arr, true_path):
        S = est_arr.sum(1)   # (8, 2)
        cos = [float(S[d] @ true_path[d] / (np.linalg.norm(S[d]) * np.linalg.norm(true_path[d]) + 1e-9)) for d in range(8)]
        ratio = [float(np.linalg.norm(S[d]) / (np.linalg.norm(true_path[d]) + 1e-9)) for d in range(8)]
        return float(np.mean(cos)), float(np.mean(ratio)), float(np.mean(np.linalg.norm(S, axis=1)))
    for v, fi in test_imgs:
        img = read_frame(v, fi); center = (640.0, 400.0); key = f"{v}:{fi}"
        sp = 2.0; path = DIRS * sp * 12
        t1 = episode(est, img, center, 90.0, np.zeros((8, 2)), DIRS * sp, C=C)
        t1s = episode(est, img, center, 90.0, np.zeros((8, 2)), DIRS * sp, grow=0.01, C=C)
        t2 = episode(est, img, center, 90.0, DIRS * sp, np.zeros((8, 2)), C=C)
        rot = np.stack([DIRS[(d + 2) % 8] for d in range(8)])
        t3 = episode(est, img, center, 90.0, DIRS * sp, rot * sp, C=C)
        t4 = episode(est, img, center, 90.0, DIRS * sp, np.zeros((8, 2)), C=C, closed=True)
        r = {}
        for var in ("old", "new", "ctrl"):
            e1 = float(np.mean(np.linalg.norm(t1[var]["est"].sum(1), axis=1))) / (sp * 12)
            e1s = float(np.mean(np.linalg.norm(t1s[var]["est"].sum(1), axis=1))) / (sp * 12)
            c2, q2, _ = summ(t2[var]["est"], path); c3, q3, _ = summ(t3[var]["est"], path)
            pos_err = float(np.mean(np.linalg.norm(t4[var]["est"].sum(1) - path, axis=1))); no_corr = float(np.mean(np.linalg.norm(path, axis=1)))
            r[var] = {"T1_drift_share": e1, "T1s_drift_share": e1s, "T2_cos": c2, "T2_ratio": q2, "T3_cos": c3, "T3_ratio": q3, "T4_err_px": pos_err, "T4_no_corr_px": no_corr}
            print(f"[CHECK {key}] {var}: T1 перенос/путь {e1:.2f} (с ростом {e1s:.2f}); T2 cos {c2:+.2f} длина {q2:.2f}; T3 cos {c3:+.2f} длина {q3:.2f}; T4 ошибка {pos_err:.1f} px (без поправки {no_corr:.1f})", flush=True)
        checks[key] = r
    agg = {var: {k: float(np.mean([checks[i][var][k] for i in checks])) for k in checks[next(iter(checks))][var]} for var in ("old", "new", "ctrl")}
    n = agg["new"]
    passed = {"T1": n["T1_drift_share"] <= 0.25 and n["T1s_drift_share"] <= 0.25, "T2": n["T2_cos"] >= 0.8 and 0.6 <= n["T2_ratio"] <= 1.4,
              "T3": n["T3_cos"] >= 0.8 and 0.6 <= n["T3_ratio"] <= 1.4, "T4": n["T4_err_px"] <= 0.5 * n["T4_no_corr_px"]}
    rep = {"calib_images": calib_imgs, "test_images": test_imgs, "C": {k: v.tolist() for k, v in C.items()}, "fit": fitq, "checks": checks, "agg": agg, "passed": passed, "all_passed": all(passed.values()), "minutes": (time.time() - t0) / 60}
    (OUT / "calib.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"[AGG] {json.dumps(agg, ensure_ascii=False)}\n[PASSED] {passed} → {rep['all_passed']}\n[OK] {(time.time() - t0) / 60:.1f} мин", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=("calib", "stage1")); a = ap.parse_args()
    return calib(a) if a.cmd == "calib" else stage1(a)


def stage1(a) -> int:
    raise SystemExit("ступень 1 — scripts/fly_efference_stage1.py")


if __name__ == "__main__":
    raise SystemExit(main())
