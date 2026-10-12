"""fly_efference_eye_v1, ступень 1: накопленная ошибка положения при последовательном прогнозе через потерю трека.

Эпизоды (общие для всех вариантов) — предварительный проход штатного трекера (XYWHByteTrack, буфер 30) на детекциях
dev 35: трек ведомый в кадре L−1 и потерян в L; эпизод — кадры L … min(L + 29, возврат или удаление трека).
Начальное состояние — центр, размер и скорость Калмана (x, y) трека после кадра L−1. Собственный объект — по реальной
разметке на последнем размеченном кадре a ∈ [L−12, L−1] (рамка выхода трека ↔ объект того же класса, IoU ≥ 0,5,
наибольший). Кадры сравнения — только реально размеченные кадры эпизода с этим объектом (интерполяция не используется).
Варианты (замкнутый цикл внутри эпизода): прогноз p_t = c_{t−1} + v_Калман; вырезка глаза — у p_t (размер рамки постоянен);
  none — c_t = p_t;  old / new / ctrl — c_t = c_{t−1} + оценка движения объекта (scripts/fly_efference_eye.py, C из
  runs/fly_efference_eye/calib.json; old — + известное перемещение вырезки). Глаза old и new рождаются у треков с эпизодами
  (как у сигнатур стенда), в ведомой фазе следуют прогнозу трекера, сохраняются при потере.
Ошибка — расстояние оценённого центра до центра объекта по разметке, px; по видео — среднее по кадрам сравнения.
Критерий (до прогона): new − old, new − none, new − ctrl по видео < 0, ДИ (бутстрэп 5000) ниже 0.

Запуск: python scripts/fly_efference_stage1.py run      (оркестратор: 3 процесса с блокировкой GPU, продолжение с готового)
        python scripts/fly_efference_stage1.py video ID  (одно видео)
        python scripts/fly_efference_stage1.py report
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

OUT = PROJECT_ROOT / "runs/fly_efference_eye/stage1"
ROOT = PROJECT_ROOT / "runs/train_cache"
VARS = ("none", "old", "new", "ctrl")


def episodes(vid):
    """Предварительный проход трекера (процессор): эпизоды потери с собственным объектом и кадрами сравнения."""
    from ultralytics.trackers.basetrack import BaseTrack, TrackState
    from datasets.bdd100k_loader import CLASS_NAMES
    from scripts.freeze_bench import XYWH_CONFIG
    from scripts.output_filter_oracle import _iou
    from scripts.run_bdd_video_rate_experiment import detections_for_frame, load_detection_cache
    from tracking.offline_association import _DetSlice, make_tracker_args
    from tracking.xywh_bytetrack import XYWHByteTrack
    det = load_detection_cache(ROOT / vid / "detections" / "raw_video.npz")
    matches = json.loads((ROOT / vid / "alignment.json").read_text(encoding="utf-8"))["matches"]
    truth = json.loads((PROJECT_ROOT / "data/bdd100k/labels/box_track_20/train" / f"{vid}.json").read_text(encoding="utf-8"))
    ann = {}
    for m, fr in zip(matches, truth):
        ann[int(m["video_frame_index"])] = [(str(l["id"]), l["category"], np.array([l["box2d"]["x1"], l["box2d"]["y1"], l["box2d"]["x2"], l["box2d"]["y2"]])) for l in fr.get("labels") or [] if l.get("box2d") and not (l.get("attributes") or {}).get("crowd")]
    n = int(np.load(PROJECT_ROOT / "runs/fly_signatures" / vid / "signatures.npz")["frame_count"])
    BaseTrack.reset_id(); tr = XYWHByteTrack(make_tracker_args(**XYWH_CONFIG))
    hist = {}   # tid → {frame: (state, mean, cls, out_box)}
    for f in range(n):
        o = tr.update(_DetSlice(*detections_for_frame(det, f)))
        outb = {int(r[4]): r[:4] for r in np.asarray(o)} if len(o) else {}
        for t in tr.tracked_stracks + tr.lost_stracks:
            hist.setdefault(int(t.track_id), {})[f] = ("T" if t.state == TrackState.Tracked else "L", t.mean.copy(), CLASS_NAMES[int(t.cls)], outb.get(int(t.track_id)))
    eps = []
    for tid, h in hist.items():
        fs = sorted(h)
        for f in fs:
            if h[f][0] == "L" and (f - 1) in h and h[f - 1][0] == "T":
                L = f; end = L
                while end + 1 in h and h[end + 1][0] == "L" and end + 1 - L < 30:
                    end += 1
                st, mean, cls, _ = h[L - 1]
                own = None
                for a in range(L - 1, L - 13, -1):
                    if a in ann and a in h and h[a][3] is not None:
                        cand = [(g, b) for g, c, b in ann[a] if c == cls]
                        if cand:
                            ious = _iou([h[a][3]], [b for _, b in cand])[0]
                            j = int(np.argmax(ious))
                            if ious[j] >= 0.5:
                                own = cand[j][0]
                        break
                if own is None:
                    continue
                ev = {}
                for t in range(L, end + 1):
                    if t in ann:
                        for g, c, b in ann[t]:
                            if g == own:
                                ev[t] = ((b[:2] + b[2:]) / 2).tolist()
                if ev:
                    eps.append({"tid": tid, "L": L, "end": end, "c0": mean[:2].tolist(), "wh": mean[2:4].tolist(), "vk": mean[4:6].tolist(), "own": own, "eval": {str(k): v for k, v in ev.items()}})
    return eps, n


def video(vid) -> int:
    from ultralytics.trackers.basetrack import BaseTrack, TrackState
    from scripts.fly_efference_eye import Ctrl, eye_raw
    from scripts.freeze_bench import XYWH_CONFIG
    from scripts.run_bdd_video_rate_experiment import detections_for_frame, load_detection_cache, make_motion_estimator, save_json
    from tracking.efference_eyes import EfferenceEyeBank
    from tracking.foveated_eyes import EyeBank, crop_box
    from tracking.offline_association import _DetSlice, make_tracker_args
    from tracking.retina import make_retina
    from tracking.xywh_bytetrack import XYWHByteTrack
    t0 = time.time()
    eps, n = episodes(vid)
    C = {k: np.asarray(v) for k, v in json.loads((PROJECT_ROOT / "runs/fly_efference_eye/calib.json").read_text(encoding="utf-8"))["C"].items()}
    by_tid = {}
    for i, e in enumerate(eps):
        by_tid.setdefault(e["tid"], []).append(i)
    start = {e["L"]: [] for e in eps}
    for i, e in enumerate(eps):
        start[e["L"]].append(i)
    det = load_detection_cache(ROOT / vid / "detections" / "raw_video.npz")
    est = make_motion_estimator("flyvis_tiled", 30.0, 0.02)
    if est.torch.cuda.is_available():
        est.torch.cuda.set_per_process_memory_fraction(0.3)
    banks = {"old": EyeBank(est, max_batch=16, decode_flow=True), "new": EfferenceEyeBank(est, max_batch=16, decode_flow=True)}
    retina = make_retina("lamina", 30.0, blur_downscale=4)
    BaseTrack.reset_id(); tr = XYWHByteTrack(make_tracker_args(**XYWH_CONFIG))
    cap = cv2.VideoCapture(str(PROJECT_ROOT / "data/bdd100k/videos/train" / f"{vid}.mov"))
    active = {}   # индекс эпизода → состояние по вариантам
    errs = {v: {} for v in VARS}   # индекс эпизода → {кадр: ошибка}
    prev_tracked, prev_img, f = [], None, 0
    while f < n:
        ok, frame = cap.read()
        if not ok:
            break
        img = retina.process(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        for i in start.get(f, []):   # начало эпизода: состояние после кадра L−1
            e = eps[i]
            active[i] = {"c": {v: np.array(e["c0"], float) for v in VARS}, "pc": {v: np.array(e["c0"], float) for v in ("old",)}, "ctrl": Ctrl()}
            active[i]["ctrl"].prev = prev_img
        lost_in_ep = {eps[i]["tid"]: i for i in active}
        # прогнозы глаз: ведомые треки с эпизодами — прогноз трекера; треки в эпизоде — свой замкнутый цикл по варианту
        preds = {"old": {}, "new": {}}
        for t in prev_tracked:
            tid = int(t.track_id)
            if tid in by_tid and tid not in lost_in_ep:
                p = tuple((t.mean[:4] + t.mean[4:8]).tolist())
                preds["old"][tid] = p; preds["new"][tid] = p
        for i in active:
            e = eps[i]; w, h = e["wh"]; vk = np.array(e["vk"])
            for var in ("old", "new"):
                pcen = active[i]["c"][var] + vk
                preds[var][e["tid"]] = (float(pcen[0]), float(pcen[1]), w, h)
        for var in ("old", "new"):
            banks[var].step(img, preds[var])
        for i in list(active):
            e = eps[i]; w, h = e["wh"]; vk = np.array(e["vk"]); tid = e["tid"]; A = active[i]
            for var in VARS:
                p = A["c"][var] + vk
                if var == "none":
                    A["c"][var] = p
                elif var == "ctrl":
                    box = (float(p[0]), float(p[1]), w, h); crop = crop_box(box, (img.shape[1], img.shape[0]))
                    r = A["ctrl"].raw(img, crop, box); A["ctrl"].prev = img
                    A["c"][var] = A["c"][var] + C["ctrl"] @ r
                else:
                    box = preds[var][tid]
                    r = eye_raw(banks[var], est, tid, box) if tid in banks[var].crops else np.zeros(2)
                    v = C[var] @ r
                    if var == "old":
                        v = v + (np.array(box[:2]) - A["pc"]["old"]); A["pc"]["old"] = np.array(box[:2])
                    A["c"][var] = A["c"][var] + v
                if str(f) in e["eval"]:
                    errs[var].setdefault(i, {})[f] = float(np.linalg.norm(A["c"][var] - np.array(e["eval"][str(f)])))
            if f >= e["end"]:
                del active[i]
        o = tr.update(_DetSlice(*detections_for_frame(det, f)))
        tracked = list(tr.tracked_stracks); cur = {int(t.track_id) for t in tracked}; alive = cur | {int(t.track_id) for t in tr.lost_stracks}
        births = {int(t.track_id): tuple(t.mean[:4].tolist()) for t in tracked if int(t.track_id) in by_tid and int(t.track_id) not in banks["old"].track_ids}
        for var in ("old", "new"):
            banks[var].birth(img, births)
            banks[var].remove(set(banks[var].track_ids) - alive)
        for var in ("old", "new"):   # у ведомых треков без эпизода в первом кадре потери глаз продолжает следовать трекеру
            pass
        prev_tracked, prev_img = tracked, img
        f += 1
        if f % 100 == 0:
            if est.torch.cuda.is_available():
                est.torch.cuda.empty_cache()
            print(f"[{vid}] {f}/{n} кадров, эпизодов активно {len(active)}, глаз {len(banks['old'].track_ids)}, {(time.time() - t0) / f * 1000:.0f} мс/кадр", flush=True)
    cap.release()
    per = {v: [x for i in errs[v] for x in errs[v][i].values()] for v in VARS}
    res = {"video": vid, "episodes": len(eps), "eval_frames": len(per["none"]), "mean_err": {v: (float(np.mean(per[v])) if per[v] else None) for v in VARS},
           "errs": {v: {str(i): errs[v][i] for i in errs[v]} for v in VARS}, "seconds": time.time() - t0}
    OUT.mkdir(parents=True, exist_ok=True); save_json(OUT / f"{vid}.json", res)
    print(f"[OK] {vid}: эпизодов {len(eps)}, кадров сравнения {res['eval_frames']}, ошибки {res['mean_err']}, {res['seconds'] / 60:.1f} мин", flush=True)
    return 0


def run() -> int:
    from scripts.fly_lifecycle_alive import sets
    S = sets(PROJECT_ROOT / "runs/fly_signatures", PROJECT_ROOT / "runs/fly_signatures_val")
    vids = [v for v, sr, rt, sp in S["dev"]]
    env = {**os.environ, "FOVEA_GPU_LOCK": "flyvis_fovea_gpu", "OMP_NUM_THREADS": "3", "MKL_NUM_THREADS": "3", "OPENBLAS_NUM_THREADS": "3", "FOVEA_THREADS": "3", "PYTHONIOENCODING": "utf-8"}
    OUT.mkdir(parents=True, exist_ok=True); st = {"done": 0, "total": len(vids), "failed": [], "t0": time.time()}
    deadline = float(os.environ.get("STAGE1_DEADLINE", "0"))   # время окна (unix): новые задания после — не начинать

    def job(v):
        if (OUT / f"{v}.json").exists():
            st["done"] += 1; return
        for attempt in range(2):
            if deadline and time.time() > deadline - 900:
                st["failed"].append(f"{v}: не начато — окно"); return
            with open(OUT / f"{v}.log", "w", encoding="utf-8") as lf:
                r = subprocess.run([sys.executable, "-u", str(Path(__file__)), "video", v], cwd=PROJECT_ROOT, env=env, stdout=lf, stderr=subprocess.STDOUT, timeout=3600)
            if r.returncode == 0 and (OUT / f"{v}.json").exists():
                st["done"] += 1
                el = time.time() - st["t0"]
                print(f"[STAGE1] {st['done']}/{st['total']} {v} | осталось ~{el / st['done'] * (st['total'] - st['done']) / 60:.0f} мин", flush=True)
                (OUT / "status.json").write_text(json.dumps({k: x for k, x in st.items() if k != "t0"}, ensure_ascii=False), encoding="utf-8")
                return
            print(f"[RETRY] {v} попытка {attempt + 1} код {r.returncode}", flush=True)
        st["failed"].append(v)
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(job, vids))
    (OUT / "status.json").write_text(json.dumps({k: x for k, x in st.items() if k != "t0"}, ensure_ascii=False), encoding="utf-8")
    return report()


def report() -> int:
    from scripts.fly_lifecycle_alive import sets
    S = sets(PROJECT_ROOT / "runs/fly_signatures", PROJECT_ROOT / "runs/fly_signatures_val")
    vids = [v for v, sr, rt, sp in S["dev"]]
    R = {v: json.loads((OUT / f"{v}.json").read_text(encoding="utf-8")) for v in vids if (OUT / f"{v}.json").exists()}
    use = [v for v in R if R[v]["eval_frames"] > 0]
    rep = {"videos_total": len(vids), "videos_done": len(R), "videos_with_eval": len(use), "complete": len(R) == len(vids),
           "episodes": int(sum(R[v]["episodes"] for v in R)), "eval_frames": int(sum(R[v]["eval_frames"] for v in R)),
           "mean_err": {k: float(np.mean([R[v]["mean_err"][k] for v in use])) for k in VARS} if use else {}, "diff": {}}
    rng = np.random.default_rng(1)
    for a_, b_ in (("new", "old"), ("new", "none"), ("new", "ctrl"), ("old", "none"), ("ctrl", "none")):
        d = np.array([R[v]["mean_err"][a_] - R[v]["mean_err"][b_] for v in use]); bs = d[rng.integers(0, len(d), (5000, len(d)))].mean(1)
        rep["diff"][f"{a_}-{b_}"] = {"mean_per_video": float(d.mean()), "ci95": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))], "better": int((d < 0).sum()), "n": len(d)}
    D = rep["diff"]; ok = lambda k: D[k]["mean_per_video"] < 0 and D[k]["ci95"][1] < 0
    rep["criterion"] = bool(rep["complete"] and ok("new-old") and ok("new-none") and ok("new-ctrl"))
    (PROJECT_ROOT / "runs/fly_efference_eye/stage1_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[REPORT] видео {rep['videos_done']}/{rep['videos_total']} (с кадрами сравнения {rep['videos_with_eval']}), эпизодов {rep['episodes']}, кадров {rep['eval_frames']}", flush=True)
    print(f"[REPORT] средняя ошибка, px: {rep['mean_err']}", flush=True)
    for k, x in D.items():
        print(f"   {k}: {x['mean_per_video']:+.2f} [{x['ci95'][0]:+.2f}; {x['ci95'][1]:+.2f}] лучше {x['better']}/{x['n']}", flush=True)
    print(f"[CRITERION] {rep['criterion']} (полный: {rep['complete']})\n[OK]", flush=True)
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1]
    raise SystemExit(run() if cmd == "run" else video(sys.argv[2]) if cmd == "video" else report())
