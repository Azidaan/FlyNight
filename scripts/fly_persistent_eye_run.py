"""Ночной прогон fly_persistent_eye_v1 по записанному протоколу (разрешение автора 11.10; переделка после просмотра dev запрещена).

1) Подбор g: 20 видео train (seed 0 из 130 train_cache), режим --fit для carry и reinit (GPU, 3 процесса с блокировкой
   FOVEA_GPU_LOCK) и dis (процессор); g = fit_gain по всем строкам варианта → runs/fly_persistent_eye/fit.json.
2) Проверка: dev 35, carry / reinit / dis с подобранными g → runs/fly_persistent_eye/dev/<вариант>/<id>.json.
3) Оценка: HOTA (TrackEval) по видео и общая; S0 — стенд без сшивки (runs/class_gate/b30); бутстрэп 5000.
   К1: carry − S0 по видео > 0, ДИ > 0; К2: carry − reinit по видео > 0, ДИ > 0; справочно dis − S0, carry − dis, ночь.
Продолжает с готового (пропускает существующие выходы); прогресс — runs/fly_persistent_eye/status.json.

Запуск: python scripts/fly_persistent_eye_run.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

OUT = PROJECT_ROOT / "runs/fly_persistent_eye"
S0 = PROJECT_ROOT / "runs/class_gate/b30"
ENV = {**os.environ, "FOVEA_GPU_LOCK": "flyvis_fovea_gpu", "OMP_NUM_THREADS": "3", "MKL_NUM_THREADS": "3", "OPENBLAS_NUM_THREADS": "3", "FOVEA_THREADS": "3", "PYTHONIOENCODING": "utf-8"}
STATE = {"stage": "", "done": 0, "total": 0, "failed": [], "t0": time.time()}


def status():
    el = time.time() - STATE["t0"]; d = max(STATE["done"], 1)
    STATE["eta_min"] = round(el / d * (STATE["total"] - STATE["done"]) / 60, 1) if STATE["done"] else None
    (OUT / "status.json").write_text(json.dumps({k: v for k, v in STATE.items() if k != "t0"}, ensure_ascii=False), encoding="utf-8")


def job(args):
    variant, vid, g, fit, out = args
    marker = out / (f"{vid}_fit.npy" if fit else f"{vid}.json")
    if marker.exists():
        STATE["done"] += 1; status(); return True
    cmd = [sys.executable, "-u", str(PROJECT_ROOT / "scripts/fly_persistent_eye.py"), "--variant", variant, "--video-id", vid, "--out", str(out), "--gpu-mem-fraction", "0.3"]
    cmd += ["--fit"] if fit else ["--g", repr(g)]
    for attempt in range(2):
        out.mkdir(parents=True, exist_ok=True)
        with open(out / f"{vid}.log", "w", encoding="utf-8") as lf:
            r = subprocess.run(cmd, cwd=PROJECT_ROOT, env=ENV, stdout=lf, stderr=subprocess.STDOUT, timeout=3600)
        if r.returncode == 0 and marker.exists():
            STATE["done"] += 1; status()
            print(f"[{STATE['stage']}] {STATE['done']}/{STATE['total']} {variant} {vid} | осталось ~{STATE.get('eta_min')} мин", flush=True)
            return True
        print(f"[RETRY] {variant} {vid} попытка {attempt + 1} код {r.returncode}", flush=True)
    STATE["failed"].append(f"{variant}:{vid}"); STATE["done"] += 1; status(); return False


def run_stage(name, tasks):
    STATE.update(stage=name, done=0, total=len(tasks), t0=time.time()); status()
    gpu = [t for t in tasks if t[0] != "dis"]; cpu = [t for t in tasks if t[0] == "dis"]
    for t in cpu:
        job(t)
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(job, gpu))


def main() -> int:
    from scripts.fly_persistent_eye import fit_gain
    from scripts.fly_lifecycle_alive import sets
    OUT.mkdir(parents=True, exist_ok=True)
    sel = json.loads((PROJECT_ROOT / "runs/train_cache/selection.json").read_text(encoding="utf-8"))["videos"]
    train20 = sorted(np.random.default_rng(0).choice(sel["train"], 20, replace=False).tolist())
    S = sets(PROJECT_ROOT / "runs/fly_signatures", PROJECT_ROOT / "runs/fly_signatures_val")
    dev = [v for v, sr, rt, sp in S["dev"]]
    V = ("carry", "reinit", "dis")
    run_stage("fit", [(v, vid, 0.0, True, OUT / "fit" / v) for v in V for vid in train20])
    fit = {"train20": train20}
    for v in V:
        rows = [np.load(OUT / "fit" / v / f"{vid}_fit.npy") for vid in train20 if (OUT / "fit" / v / f"{vid}_fit.npy").exists()]
        fit[v] = fit_gain(rows)   # список массивов по видео: пары строятся внутри видео (исправление 12.10)
        print(f"[FIT] {v}: {fit[v]}", flush=True)
    (OUT / "fit.json").write_text(json.dumps(fit, ensure_ascii=False, indent=1), encoding="utf-8")
    run_stage("dev", [(v, vid, fit[v]["g"], False, OUT / "dev" / v) for v in V for vid in dev])
    # оценка
    from evaluation.hota_eval import evaluate_hota
    from datasets.subset_filter import filter_videos_by_subset, load_video_attributes
    KEYS = ("HOTA", "DetA", "AssA", "IDF1", "IDSw")
    dirs = {"S0": S0, **{v: OUT / "dev" / v for v in V}}
    vids = [v for v in dev if all((d / f"{v}.json").exists() for d in dirs.values())]
    per = {n: {v: evaluate_hota(d, [v], split="train") for v in vids} for n, d in dirs.items()}
    night = set(filter_videos_by_subset(load_video_attributes(str(PROJECT_ROOT / "data/bdd100k/labels/bdd100k_labels_images_train.json")), vids, "night"))
    rep = {"videos": len(vids), "missing": [v for v in dev if v not in vids], "failed": STATE["failed"], "g": {v: fit[v]["g"] for v in V}, "pooled": {}, "diff": {}}
    rng = np.random.default_rng(1)
    for grp, gv in (("all", vids), ("night", [v for v in vids if v in night])):
        for n, d in dirs.items():
            m = evaluate_hota(d, gv, split="train"); rep["pooled"][f"{n}|{grp}"] = {k: m.get(k) for k in KEYS}
        for a_, b_ in (("carry", "S0"), ("carry", "reinit"), ("reinit", "S0"), ("dis", "S0"), ("carry", "dis")):
            dd = np.array([per[a_][v]["HOTA"] - per[b_][v]["HOTA"] for v in gv]); bs = dd[rng.integers(0, len(dd), (5000, len(dd)))].mean(1)
            rep["diff"][f"{a_}-{b_}|{grp}"] = {"mean_per_video": float(dd.mean()), "ci95": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))], "better": int((dd > 0).sum()), "worse": int((dd < 0).sum()), "n": len(dd),
                                               "pooled_gain": rep["pooled"][f"{a_}|{grp}"]["HOTA"] - rep["pooled"][f"{b_}|{grp}"]["HOTA"]}
    D = rep["diff"]; ok = lambda k: D[k]["mean_per_video"] > 0 and D[k]["ci95"][0] > 0
    rep["K1"] = bool(ok("carry-S0|all")); rep["K2"] = bool(ok("carry-reinit|all"))
    (OUT / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    for grp in ("all", "night"):
        print(f"[HOTA] {grp}: " + "; ".join(f"{n} {rep['pooled'][f'{n}|{grp}']['HOTA']:.3f} (AssA {rep['pooled'][f'{n}|{grp}']['AssA']:.2f}, IDSw {rep['pooled'][f'{n}|{grp}']['IDSw']:.0f})" for n in dirs), flush=True)
        for k, x in D.items():
            if k.endswith(f"|{grp}"):
                print(f"   {k}: общий {x['pooled_gain']:+.3f}; по видео {x['mean_per_video']:+.3f} [{x['ci95'][0]:+.3f}; {x['ci95'][1]:+.3f}] лучше {x['better']}, хуже {x['worse']} из {x['n']}", flush=True)
    print(f"[K1] {rep['K1']}  [K2] {rep['K2']}  [FAILED] {rep['failed']}\n[OK]", flush=True)
    STATE.update(stage="done"); status()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
