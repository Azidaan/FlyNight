"""Г3: глаз мухи перед детектором — замороженный YOLO26 на кадрах после фоторецептора / накопления (карточка fly_eye_frontend_v1).

Зачем. Четыре гейта 03.10 показали: пересчёт после детектора не создаёт нового свидетельства
(правило 6.14). Новый сигнал может войти только до детектора, на уровне пикселей и нескольких
кадров: накопление света во времени ночью, локальная адаптация яркости (пересвет, тень).

Варианты входа (только прошлые кадры — работает онлайн):
  raw       — сырой кадр (база; сверка с кэшем raw_video.npz);
  tavg3/5   — среднее 3/5 последних кадров, совмещённых гомографией фона (DIS, RANSAC) — НЕбиологический контроль накопления;
  pr        — фоторецептор (tracking/retina.py, режим photoreceptor: закон Вебера с локальной адаптацией τ 0,3 с, σ 24 px), применён к яркости, цвет сохранён отношением;
  pr_int    — то же с временной интеграцией фоторецептора (tau_filter 0,1 с) — ночное накопление «по-мушиному», без компенсации движения;
  pr_tavg3  — фоторецептор поверх совмещённого среднего 3 кадров.
Детектор: weights/baseline_seed42/best.pt, imgsz 960, conf 0,01, iou 0,7 (как в кэшах).
Наборы: val7 (детектор не видел) и dev+test 70 видео train_cache (только размеченные кадры вне full_step5_train).

Запуск:
    python scripts/gate_fly_eye_frontend.py --workers 2          # детекции по вариантам
    python scripts/gate_fly_eye_frontend.py --eval               # оценка
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

OUT = PROJECT_ROOT / "runs" / "gate_g3"
STATUS = OUT / "status.json"
VAL7 = ["b1c66a42-6f7d68ca", "b1c81faa-3df17267", "b1cac6a7-04e33135", "b1cebfb7-284f5117", "b232c7c9-d251d9ee", "b1d22449-15fb948f", "b251064f-e7a165fd"]
VARIANTS = ["raw", "tavg3", "tavg5", "pr", "pr_int", "pr_tavg3"]
WEIGHTS = PROJECT_ROOT / "weights" / "baseline_seed42" / "best.pt"


def tasks_all(limit=0):
    sel = json.loads((PROJECT_ROOT / "runs/train_cache/selection.json").read_text(encoding="utf-8"))["videos"]
    dt = sel["dev"] + sel["test"]
    v7 = VAL7
    if limit:
        dt, v7 = dt[:limit], v7[:limit]
    t = [(v, "val", str(PROJECT_ROOT / "runs/video_rate" / v), str(PROJECT_ROOT / "data/bdd100k/videos/val" / f"{v}.mov")) for v in v7]
    t += [(v, "train", str(PROJECT_ROOT / "runs/train_cache" / v), str(PROJECT_ROOT / "data/bdd100k/videos/train" / f"{v}.mov")) for v in dt]
    return t


def write_status(**kw):
    OUT.mkdir(parents=True, exist_ok=True)
    st = json.loads(STATUS.read_text(encoding="utf-8")) if STATUS.exists() else {}
    st.update(kw, updated=time.time())
    tmp = STATUS.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(st, ensure_ascii=False), encoding="utf-8")
    for _ in range(20):
        try:
            tmp.replace(STATUS); return
        except PermissionError:
            time.sleep(0.05)


def apply_luma(bgr_f, y_new):
    """Заменить яркость, сохранив цвет: каналы умножаются на Y_new / Y_old (с ограничением)."""
    y_old = 0.114 * bgr_f[..., 0] + 0.587 * bgr_f[..., 1] + 0.299 * bgr_f[..., 2]
    k = np.clip(y_new / np.maximum(y_old, 1.0), 0.0, 8.0)[..., None]
    return np.clip(bgr_f * k, 0, 255).astype(np.uint8)


def run_video(task):
    vid, split, cache_dir, video_path = task
    out = OUT / "det" / f"{vid}.npz"
    if out.exists():
        return vid, 0.0
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    import cv2
    from ultralytics import YOLO
    from scripts.gate_motion_rescore_30hz import homographies
    from tracking.retina import make_retina

    cv2.setNumThreads(2)
    t0 = time.time()
    seen = set()
    if split == "train":
        seen = {Path(l.strip()).name for l in (PROJECT_ROOT / "training/configs/generated/full_step5_train.txt").read_text(encoding="utf-8").splitlines() if l.strip() and vid in l}
    al = json.loads((Path(cache_dir) / "alignment.json").read_text(encoding="utf-8"))
    want = {int(m["video_frame_index"]): m["reference_name"] for m in al["matches"] if m["reference_name"] not in seen}
    fps = float(al["video"]["fps"])
    with np.load(Path(cache_dir) / "motion/dis_quarter.npz") as z:
        Hs = homographies(z["points"].astype(np.float32), z["vectors"].astype(np.float32))
    model = YOLO(str(WEIGHTS))
    ret = {"pr": make_retina("photoreceptor", fps, blur_downscale=4), "pr_int": make_retina("photoreceptor", fps, blur_downscale=4, tau_filter=0.1),
           "pr_tavg3": make_retina("photoreceptor", fps, blur_downscale=4)}
    cap = cv2.VideoCapture(video_path)
    buf, gbuf = [], []  # последние цветные кадры (uint8) и серые (float32)
    res = {v: {"xyxy": [], "conf": [], "cls": [], "frame": []} for v in VARIANTS}
    names = []
    r = -1
    last_frame = max(want) if want else -1

    def tavg(stack, n):
        """Среднее n последних кадров, совмещённых с текущим гомографией фона; вне кадра — не учитывается."""
        cur = stack[-1].astype(np.float32)
        H_, W_ = cur.shape[:2]
        acc = cur.copy(); wsum = np.ones((H_, W_), np.float32)
        M = np.eye(3)
        for j in range(1, min(n, len(stack))):
            Hj = Hs[r - j + 1] if r - j + 1 < len(Hs) else None
            if Hj is None:
                break
            M = M @ Hj
            warped = cv2.warpPerspective(stack[-1 - j].astype(np.float32), M, (W_, H_), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
            ok = np.isfinite(warped if warped.ndim == 2 else warped[..., 0])
            acc[ok] += warped[ok]; wsum += ok
        return acc / (wsum if acc.ndim == 2 else wsum[..., None])

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        r += 1
        if r > last_frame:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        buf.append(frame); buf = buf[-5:]
        gbuf.append(gray); gbuf = gbuf[-3:]
        # состояние фоторецепторов обновляется на каждом кадре (по серому каналу)
        y_pr = ret["pr"].process(gray)
        y_int = ret["pr_int"].process(gray)
        y3 = tavg(gbuf, 3)
        y_pt = ret["pr_tavg3"].process(np.clip(y3, 0, 255).astype(np.uint8))
        if r not in want:
            continue
        f = frame.astype(np.float32)
        t3 = tavg(buf, 3)
        imgs = {"raw": frame, "tavg3": np.clip(t3, 0, 255).astype(np.uint8), "tavg5": np.clip(tavg(buf, 5), 0, 255).astype(np.uint8),
                "pr": apply_luma(f, y_pr), "pr_int": apply_luma(f, y_int), "pr_tavg3": apply_luma(t3, y_pt)}
        preds = model.predict([imgs[v] for v in VARIANTS], imgsz=960, conf=0.01, iou=0.7, verbose=False, device=0)
        for v, p in zip(VARIANTS, preds):
            b = p.boxes
            res[v]["xyxy"].append(b.xyxy.cpu().numpy().astype(np.float32)); res[v]["conf"].append(b.conf.cpu().numpy().astype(np.float32))
            res[v]["cls"].append(b.cls.cpu().numpy().astype(np.int16)); res[v]["frame"].append(np.full(len(b), len(names), np.int32))
        names.append(want[r])
        if len(names) % 20 == 0:
            write_status(**{f"w{os.getpid()}": {"video": vid, "done_frames": len(names), "frames": len(want)}})
    cap.release()
    arrays = {"names": np.array(names)}
    for v in VARIANTS:
        for k in ("xyxy", "conf", "cls", "frame"):
            arrays[f"{v}__{k}"] = np.concatenate(res[v][k]) if res[v][k] else np.zeros((0, 4) if k == "xyxy" else 0)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **arrays)
    return vid, time.time() - t0


def produce(args):
    tasks = tasks_all(args.limit)
    done = [t[0] for t in tasks if (OUT / "det" / f"{t[0]}.npz").exists()]
    write_status(stage="detect", total=len(tasks), done=len(done), started=time.time())
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for vid, sec in pool.map(run_video, tasks):
            done.append(vid)
            write_status(done=len(done), last=vid, last_sec=sec)
            print(f"[DET] {len(done)}/{len(tasks)} {vid} {sec:.0f} с", flush=True)
    write_status(stage="detect_done")


def evaluate(args):
    from datasets.subset_filter import filter_videos_by_subset, load_video_attributes
    from scripts.gate_motion_rescore_30hz import iou_mat, load_gt

    tasks = tasks_all(args.limit)
    cond = {}
    for split in ("val", "train"):
        attrs = load_video_attributes(str(PROJECT_ROOT / f"data/bdd100k/labels/bdd100k_labels_images_{split}.json"))
        ids = [t[0] for t in tasks if t[1] == split]
        for c in ("normal", "night"):
            for v in filter_videos_by_subset(attrs, ids, c):
                cond.setdefault(v, c)
    data = {}
    for vid, split, cache_dir, _ in tasks:
        z = np.load(OUT / "det" / f"{vid}.npz")
        gt = load_gt(split, vid)
        per = {}
        for v in VARIANTS:
            xy, cf, cl, fr = z[f"{v}__xyxy"], z[f"{v}__conf"], z[f"{v}__cls"].astype(np.int64), z[f"{v}__frame"]
            gs_all, ag_all, fp_all, blind = [], [], [], 0
            for i, name in enumerate(z["names"]):
                gb, gc, ages = gt[str(name)]
                if len(gb) == 0:
                    continue
                sel = np.flatnonzero(fr == i)
                m = np.where(gc[:, None] == cl[sel][None, :], iou_mat(gb, xy[sel]), 0.0) if len(sel) else np.zeros((len(gb), 0))
                blind += int((m.max(1) < 0.5).sum()) if m.shape[1] else len(gb)
                gs = np.full(len(gb), -np.inf); used = np.zeros(len(gb), bool)
                for j in np.argsort(-cf[sel]):
                    col = np.where(used, 0.0, m[:, j]); k = int(col.argmax())
                    if col[k] >= 0.5:
                        used[k] = True; gs[k] = cf[sel][j]
                    else:
                        fp_all.append(cf[sel][j])
                gs_all.append(gs); ag_all.append(ages)
            per[v] = (np.concatenate(gs_all), np.concatenate(ag_all), np.array(fp_all, np.float32), blind)
        data[vid] = per
    sets = {"val7": [t[0] for t in tasks if t[1] == "val"], "devtest": [t[0] for t in tasks if t[1] == "train"]}
    res = {"sets": {}}
    rng = np.random.default_rng(0)
    for sname, vids in sets.items():
        budget = sum(int((data[v]["raw"][2] >= 0.25).sum()) for v in vids)
        thr = {}
        for var in VARIANTS:
            fs = np.sort(np.concatenate([data[v][var][2] for v in vids]))[::-1]
            thr[var] = 0.25 if var == "raw" else float(fs[min(budget, len(fs)) - 1])
        cnt = {var: {v: (int((data[v][var][0] >= thr[var]).sum()), int(((data[v][var][0] >= thr[var]) & (data[v][var][1] == 0)).sum()), data[v][var][3]) for v in vids} for var in VARIANTS}
        n = {v: (len(data[v]["raw"][0]), int((data[v]["raw"][1] == 0).sum())) for v in vids}
        groups = {"all": vids, "night": [v for v in vids if cond.get(v) == "night"], "day": [v for v in vids if cond.get(v) == "normal"]}
        def rec(var, vs, i=0):
            return sum(cnt[var][v][i] for v in vs) / max(sum(n[v][min(i, 1)] for v in vs), 1)
        def blind(var, vs):
            return sum(cnt[var][v][2] for v in vs) / max(sum(n[v][0] for v in vs), 1)
        r = {"budget": budget, "thresholds": thr, "videos": {g: len(vs) for g, vs in groups.items()}, "recall": {}, "age0": {}, "blind": {}, "diff": {}, "ci95": {}}
        for g, vs in groups.items():
            if not vs:
                continue
            r["recall"][g] = {var: rec(var, vs) for var in VARIANTS}
            r["age0"][g] = {var: rec(var, vs, 1) for var in VARIANTS}
            r["blind"][g] = {var: blind(var, vs) for var in VARIANTS}
            for var in VARIANTS[1:]:
                for metric, fn in (("recall", lambda a, vv: rec(a, vv)), ("age0", lambda a, vv: rec(a, vv, 1))):
                    key = f"{g}.{metric}.{var}-raw"
                    r["diff"][key] = fn(var, vs) - fn("raw", vs)
                    b = []
                    for _ in range(args.boot):
                        pick = [vs[i] for i in rng.integers(0, len(vs), len(vs))]
                        b.append(fn(var, pick) - fn("raw", pick))
                    r["ci95"][key] = [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
            for fly in ("pr", "pr_int", "pr_tavg3"):
                key = f"{g}.recall.{fly}-best_tavg"
                best = max(("tavg3", "tavg5"), key=lambda t: rec(t, vs))
                r["diff"][key] = rec(fly, vs) - rec(best, vs)
                b = []
                for _ in range(args.boot):
                    pick = [vs[i] for i in rng.integers(0, len(vs), len(vs))]
                    b.append(rec(fly, pick) - rec(best, pick))
                r["ci95"][key] = [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
        res["sets"][sname] = r
        print(f"== {sname}: бюджет FP {budget}; видео {r['videos']}")
        for g in r["recall"]:
            print(f"  [{g}]")
            for var in VARIANTS:
                print(f"    {var:9s} порог {thr[var]:.3f} полнота {r['recall'][g][var]:.4f} возраст0 {r['age0'][g][var]:.4f} слепота {r['blind'][g][var]:.4f}"
                      + (f"  Δ {r['diff'][f'{g}.recall.{var}-raw'] * 100:+.2f} пп [{r['ci95'][f'{g}.recall.{var}-raw'][0] * 100:+.2f}; {r['ci95'][f'{g}.recall.{var}-raw'][1] * 100:+.2f}]" if var != "raw" else ""))
    # сверка raw с кэшем детекций
    chk = []
    for vid, split, cache_dir, _ in tasks[:5]:
        z = np.load(OUT / "det" / f"{vid}.npz"); c = np.load(Path(cache_dir) / "detections/raw_video.npz")
        chk.append((len(z["raw__conf"]), vid))
    res["raw_check_counts"] = chk
    (OUT / "summary.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[OK] {OUT / 'summary.json'}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--eval", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.eval:
        evaluate(args)
    else:
        produce(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
