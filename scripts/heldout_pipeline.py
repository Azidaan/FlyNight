"""Конвейер для отложенных размеченных val-видео (подтверждение зафиксированного варианта; подготовка 11.10).

Обобщает конвейер life_filter_newval_v1 (newval_build / newval_jobs / newval_eval) на любой набор с именем NAME:
  cache  — для видео списка с alignment_ok в runs/newval_videos/manifest.json: выравнивание, частота, детекции стенда
           (YOLO26s, 960, conf 0,01, iou 0,7; GPU) → runs/NAME_cache/<id>, selection.json;
  jobs   — задания сигнатур мухи с историей (--history) → runs/jobs_signatures_NAME.json, выход runs/fly_signatures_NAME;
  (DINO и MASA — scripts/build_dino_embeddings.py / build_masa_embeddings.py с --ids-json runs/NAME_cache/selection.json)
  stitch — голова пары «база + MASA» на train 130, жадная сшивка 0,6 → runs/NAME_eval/stitched; жизни и признаки v2
           (life_filter_head.video_task) с метками → runs/NAME_eval/lives.pkl;
  check  — сверка выходов NAME с эталонным набором (по умолчанию новые val: runs/newval_cache, fly_signatures_newval,
           newval_eval) на общих видео — для проверки самого конвейера, не для выводов.
Видео без выравнивания не используются (правило списка подтверждения: runs/life_fly_history/confirm_videos.json).
Пути входов шага stitch можно переопределить (--sig-root, --cache-root, --dino, --masa) — так конвейер проверяется на
готовых кэшах новых val без GPU.

Запуск: python scripts/heldout_pipeline.py cache --name heldout --ids runs/life_fly_history/confirm_videos.json
        python scripts/heldout_pipeline.py jobs --name heldout
        python scripts/heldout_pipeline.py stitch --name heldout
        python scripts/heldout_pipeline.py check --name heldout_test
"""
from __future__ import annotations

import argparse
import json
import pickle
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

WEIGHTS = PROJECT_ROOT / "weights" / "baseline_seed42" / "best.pt"
VIDEOS = PROJECT_ROOT / "data/bdd100k/videos/val"
MANIFEST = PROJECT_ROOT / "runs/newval_videos/manifest.json"


def ids_from(path: Path) -> list[str]:
    d = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(d, list):
        return d
    if "videos" in d:
        return list(d["videos"])
    return [v for k in ("night", "normal", "other") for v in d.get(k, [])]


def cmd_cache(a) -> int:
    from scripts.run_bdd_video_rate_experiment import build_detection_cache, save_json
    man = json.loads(MANIFEST.read_text(encoding="utf-8"))["videos"]
    want = ids_from(a.ids)
    ok = [v for v in want if man.get(v, {}).get("alignment_ok")]
    skipped = {v: man.get(v, {}).get("error", "нет в манифесте (не скачано / не выровнено)") for v in want if v not in ok}
    out_root = PROJECT_ROOT / f"runs/{a.name}_cache"; out_root.mkdir(parents=True, exist_ok=True)
    print(f"[PLAN] {a.name}: с выравниванием {len(ok)} из {len(want)}; без — {len(skipped)}", flush=True)
    t0 = time.perf_counter()
    for k, v in enumerate(ok, 1):
        out = out_root / v; out.mkdir(parents=True, exist_ok=True)
        if not (out / "alignment.json").exists():
            shutil.copyfile(MANIFEST.parent / v / "alignment.json", out / "alignment.json")
        meta = out / "motion" / "flyvis_tiled.json"
        if not meta.exists():
            meta.parent.mkdir(parents=True, exist_ok=True)
            save_json(meta, {"source": "fps_only", "fps": float(man[v]["container"]["fps"]), "note": "только частота для build_fly_signatures"})
        build_detection_cache(VIDEOS / f"{v}.mov", WEIGHTS, out / "detections" / "raw_video.npz", imgsz=960, conf=0.01, iou=0.7, device="0", force=False)
        el = time.perf_counter() - t0
        print(f"[CACHE] {k}/{len(ok)} {v}: {el / 60:.1f} мин, осталось ~{el / k * (len(ok) - k) / 60:.1f} мин", flush=True)
    save_json(out_root / "selection.json", {"videos": ok, "skipped_no_alignment": skipped, "source": str(a.ids)})
    print("[OK] cache", flush=True)
    return 0


def cmd_jobs(a) -> int:
    ids = json.loads((PROJECT_ROOT / f"runs/{a.name}_cache/selection.json").read_text(encoding="utf-8"))["videos"]
    jobs = [dict(builder="scripts/build_fly_signatures.py", marker="history.npz", retina="lamina", retina_downscale=4, output=f"runs/fly_signatures_{a.name}",
                 video_id=v, videos="data/bdd100k/videos/val", detections_root=f"runs/{a.name}_cache", flow_root=f"runs/{a.name}_cache", extra_args=["--history"]) for v in ids]
    p = PROJECT_ROOT / f"runs/jobs_signatures_{a.name}.json"; p.write_text(json.dumps(jobs, indent=1), encoding="utf-8")
    print(f"[JOBS] {len(jobs)} → {p}", flush=True)
    return 0


def _register(name, path):
    import scripts.life_filter_head as LH
    LH.SRC[name] = Path(path)


def cmd_stitch(a) -> int:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from scripts import stitch_eval_v2 as V
    from scripts.eval_fly_reid import stitch_video
    from scripts.fly_lifecycle_alive import sets
    from scripts.life_filter_head import video_task
    cache = Path(a.cache_root or PROJECT_ROOT / f"runs/{a.name}_cache"); sig = Path(a.sig_root or PROJECT_ROOT / f"runs/fly_signatures_{a.name}")
    dino = Path(a.dino or PROJECT_ROOT / f"runs/dino_embeddings_{a.name}"); masa = Path(a.masa or PROJECT_ROOT / f"runs/masa_embeddings_{a.name}")
    out = PROJECT_ROOT / f"runs/{a.name}_eval"; out.mkdir(parents=True, exist_ok=True)
    sel = ids_from(Path(a.ids)) if a.ids else json.loads((cache / "selection.json").read_text(encoding="utf-8"))["videos"]
    vids = [v for v in sel if (sig / v / "signatures.npz").exists() and (masa / f"{v}.npz").exists() and (dino / f"{v}.npz").exists()]
    print(f"[PLAN] {a.name}: видео с полным набором {len(vids)} из {len(sel)}", flush=True)
    S = sets(PROJECT_ROOT / "runs/fly_signatures", PROJECT_ROOT / "runs/fly_signatures_val")
    with ProcessPoolExecutor(max_workers=a.workers, initializer=V._set, initargs=(150, 5.0, True)) as pool:
        tr = list(pool.map(V.cached_pairs, [(v, str(sr), str(rt), sp) for v, sr, rt, sp in S["train"]]))
    idx = [V.NAMES.index(c) for c in V.HEADS["base_masa"]]
    X = np.concatenate([d["X"][d["y"] >= 0] for d in tr]); y = np.concatenate([d["y"][d["y"] >= 0] for d in tr]); del tr
    pm = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)).fit(X[:, idx], y); del X, y
    with ProcessPoolExecutor(max_workers=a.workers, initializer=V._set_dirs, initargs=(150, 5.0, str(dino), str(masa))) as pool:
        pairs = list(pool.map(V.video_pairs, [(v, str(sig), str(cache), "val") for v in vids]))
    tasks = [(d["video_id"], str(sig), str(cache), "val", d["pairs"], pm.predict_proba(d["X"][:, idx])[:, 1] if len(d["y"]) else np.zeros(0), 0.6, str(out / "stitched")) for d in pairs]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        list(pool.map(stitch_video, tasks))
    print(f"[STITCH] сшито {len(tasks)}", flush=True)
    with ProcessPoolExecutor(max_workers=a.workers, initializer=_register, initargs=(a.name, str(out / "stitched"))) as pool:
        L = list(pool.map(video_task, [(v, str(sig), str(cache), "val", a.name, True) for v in vids]))
    pickle.dump(L, open(out / "lives.pkl", "wb"))
    seg = {k: int(sum(c["segstats"][k] for c in L)) for k in L[0]["segstats"]} if L else {}
    print(f"[LIVES] жизней {sum(len(c['label']) for c in L)}; сопоставление рамок {seg}\n[OK] stitch", flush=True)
    return 0


def cmd_check(a) -> int:
    """Сверка с эталоном на общих видео: детекции, жизни, сшитый выход, метки и признаки жизней."""
    ref_cache, ref_sig, ref_eval = PROJECT_ROOT / "runs/newval_cache", PROJECT_ROOT / "runs/fly_signatures_newval", PROJECT_ROOT / "runs/newval_eval"
    cache, sig, ev = PROJECT_ROOT / f"runs/{a.name}_cache", PROJECT_ROOT / f"runs/fly_signatures_{a.name}", PROJECT_ROOT / f"runs/{a.name}_eval"
    rep = {}
    refL = {c["video_id"]: c for c in pickle.load(open(ref_eval / "lives.pkl", "rb"))}
    L = {c["video_id"]: c for c in pickle.load(open(ev / "lives.pkl", "rb"))} if (ev / "lives.pkl").exists() else {}
    vids = sorted(set(L) & set(refL)) if L else []
    for v in vids:
        r = {}
        if (cache / v / "detections/raw_video.npz").exists():
            x, y = dict(np.load(cache / v / "detections/raw_video.npz")), dict(np.load(ref_cache / v / "detections/raw_video.npz"))
            # поля времени работы (*_seconds) различаются от прогона к прогону — не входят в сверку
            r["detections_equal"] = all(k in y and x[k].shape == y[k].shape and np.allclose(x[k], y[k], atol=1e-3) for k in x if not k.endswith("_seconds"))
        if (sig / v / "signatures.npz").exists():
            x, y = np.load(sig / v / "signatures.npz"), np.load(ref_sig / v / "signatures.npz")
            r["lives_equal"] = all(np.array_equal(x[k], y[k]) for k in ("track_id", "start", "end")) and np.allclose(x["outputs"], y["outputs"])
            r["fly_max_diff"] = float(np.abs(x["first_fly"].astype(np.float32) - y["first_fly"].astype(np.float32)).max())
        s1 = json.loads((ev / "stitched" / f"{v}.json").read_text(encoding="utf-8")); s2 = json.loads((ref_eval / "stitched" / f"{v}.json").read_text(encoding="utf-8"))
        r["stitched_equal"] = s1 == s2
        c, d = L[v], refL[v]
        r["labels_equal"] = bool(np.array_equal(c["label"], d["label"])); r["nofly_max_diff"] = float(np.abs(c["nofly"] - d["nofly"]).max()); r["fly_feat_max_diff"] = float(np.abs(c["fly"] - d["fly"]).max())
        rep[v] = r; print(v, r, flush=True)
    (PROJECT_ROOT / f"runs/{a.name}_eval").mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / f"runs/{a.name}_eval/check.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[OK] check: {len(rep)} видео", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=("cache", "jobs", "stitch", "check"))
    ap.add_argument("--name", required=True)
    ap.add_argument("--ids", type=Path, default=None)
    ap.add_argument("--sig-root"); ap.add_argument("--cache-root"); ap.add_argument("--dino"); ap.add_argument("--masa")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    return {"cache": cmd_cache, "jobs": cmd_jobs, "stitch": cmd_stitch, "check": cmd_check}[a.stage](a)


if __name__ == "__main__":
    raise SystemExit(main())
