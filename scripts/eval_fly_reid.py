"""Н11: повторное опознание по сигнатуре глаза мухи и офлайн-сшивка треков (карточка fly_signature_reid_v1).

Пары «жизнь A закончилась → жизнь B другого трека началась через 1…60 кадров, центр B не дальше
3 размеров A». Метка — один GT-объект на последнем опорном кадре A и первом опорном кадре B
(IoU ≥ 0,5). Признаки: геометрия; внешность (картинка 16×16, HSV); вход глаза (721 фасетка);
муха (34 типа × 19 макроколонок: общее косинусное сходство и по группам типов); контроль —
сигнатура мухи B, взятая у случайной другой жизни того же ролика. Лестница логистических голов:
обучение train, порог сшивки — dev (максимум IDF1 сшитого трекера), отчёт test и val7.
Сшивка (идея AFLink, StrongSORT — заимствовано): жадно по убыванию вероятности, один к одному,
p > порога; ID жизни B и всех её продолжений заменяется на ID A.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_bdd_video_rate_experiment import save_json  # noqa: E402

VAL7 = ["b1c66a42-6f7d68ca", "b1c81faa-3df17267", "b1cac6a7-04e33135", "b1cebfb7-284f5117", "b232c7c9-d251d9ee", "b1d22449-15fb948f", "b251064f-e7a165fd"]
MAX_GAP, MAX_DIST = 60, 3.0
GROUPS_OF_TYPES = {"T123": range(0, 4), "T4": range(4, 8), "T5": range(8, 12), "Tm": range(12, 25), "TmY": range(25, 34)}
GEOMETRY = ("gap", "dist", "log_size_ratio", "same_class", "a_len", "b_len")
APPEAR = ("thumb_cos", "hist_l1")
HEXIN = ("hexin_cos",)
FLY = ("fly_cos",) + tuple(f"fly_cos_{g}" for g in GROUPS_OF_TYPES)
SHUF = tuple(f"shuf_{n}" for n in FLY)
NAMES = GEOMETRY + APPEAR + HEXIN + FLY + SHUF
LADDER = {"geometry": GEOMETRY, "geometry_appearance": GEOMETRY + APPEAR, "geometry_appearance_hexin": GEOMETRY + APPEAR + HEXIN,
          "geometry_appearance_hexin_fly": GEOMETRY + APPEAR + HEXIN + FLY, "geometry_appearance_hexin_shuffled": GEOMETRY + APPEAR + HEXIN + SHUF,
          "geometry_fly": GEOMETRY + FLY}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sig-root", type=Path, default=PROJECT_ROOT / "runs" / "fly_signatures")
    parser.add_argument("--sig-val", type=Path, default=PROJECT_ROOT / "runs" / "fly_signatures_val")
    parser.add_argument("--train-cache", type=Path, default=PROJECT_ROOT / "runs" / "train_cache")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "runs" / "fly_reid")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--val-only", action="store_true", help="разведка: только val7 (LOVO), без сшивки")
    return parser.parse_args()


def cos(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean(); b = b - b.mean()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def fly_feats(a: np.ndarray, b: np.ndarray) -> list[float]:
    a = a.astype(np.float64); b = b.astype(np.float64)
    out = [cos(a.ravel(), b.ravel())]
    for rng in GROUPS_OF_TYPES.values():
        out.append(float(np.mean([cos(a[t], b[t]) for t in rng])))
    return out


def video_pairs(task: tuple) -> dict:
    video_id, sig_dir, root, label_split = task
    from scripts.evaluate_flyvis_qmatch_gate import truth_arrays
    from tracking.flyvis_tracker import box_iou

    z = np.load(Path(sig_dir) / video_id / "signatures.npz")
    S = {k: z[k] for k in z.files}
    outputs = S["outputs"]  # frame, x1, y1, x2, y2, tid, score, cls
    matches = json.loads((Path(root) / video_id / "alignment.json").read_text(encoding="utf-8"))["matches"]
    ref_of_raw = {int(m["video_frame_index"]): i for i, m in enumerate(matches)}
    truth = json.loads((PROJECT_ROOT / "data/bdd100k/labels/box_track_20" / label_split / f"{video_id}.json").read_text(encoding="utf-8"))
    rows_by_frame: dict[int, list[np.ndarray]] = {}
    for r in outputs:
        rows_by_frame.setdefault(int(r[0]), []).append(r)
    ref_frames = sorted(f for f in ref_of_raw if f in rows_by_frame)

    def gt_of(tid: int, frames: list[int]) -> str | None:
        for f in frames:
            row = next((r for r in rows_by_frame.get(f, []) if int(r[5]) == tid), None)
            if row is None:
                continue
            tb, tc, tids, _ = truth_arrays(truth[ref_of_raw[f]])
            if not len(tb):
                return None
            o = box_iou(row[1:5][None, :].astype(np.float32), tb)[0]
            return tids[int(o.argmax())] if o.max() >= 0.5 else None
        return None

    n = len(S["track_id"])
    end_gt, start_gt = [], []
    for i in range(n):
        tid, s, e = int(S["track_id"][i]), int(S["start"][i]), int(S["end"][i])
        within = [f for f in ref_frames if s <= f <= e]
        end_gt.append(gt_of(tid, within[::-1])); start_gt.append(gt_of(tid, within))
    rng = np.random.default_rng(0)
    feats, labels, meta = [], [], []
    for a in range(n):
        for b in range(n):
            if S["track_id"][a] == S["track_id"][b]:
                continue
            gap = int(S["start"][b]) - int(S["end"][a])
            if not (0 < gap <= MAX_GAP):
                continue
            A, B = S["last_box"][a].astype(np.float64), S["first_box"][b].astype(np.float64)
            sa = np.sqrt(max((A[2] - A[0]) * (A[3] - A[1]), 1.0)); sb = np.sqrt(max((B[2] - B[0]) * (B[3] - B[1]), 1.0))
            dist = np.linalg.norm((A[:2] + A[2:]) / 2 - (B[:2] + B[2:]) / 2) / sa
            if dist > MAX_DIST:
                continue
            if end_gt[a] is None or start_gt[b] is None:
                continue
            other = int(rng.integers(0, n))
            geo = [gap / 30.0, dist, float(np.log(sb / sa)), float(S["cls"][a] == S["cls"][b]), float(S["n_last"][a]) / 5, float(S["n_first"][b]) / 5]
            ap = [cos(S["last_thumb"][a].astype(np.float64), S["first_thumb"][b].astype(np.float64)), float(np.abs(S["last_hist"][a].astype(np.float64) - S["first_hist"][b].astype(np.float64)).sum())]
            hx = [cos(S["last_hexin"][a].astype(np.float64), S["first_hexin"][b].astype(np.float64))]
            fl = fly_feats(S["last_fly"][a], S["first_fly"][b]); sh = fly_feats(S["last_fly"][a], S["first_fly"][other])
            feats.append(geo + ap + hx + fl + sh); labels.append(int(end_gt[a] == start_gt[b])); meta.append((a, b, other))
    meta = np.asarray(meta, np.int32).reshape(-1, 3)
    diffs = {}
    if len(meta):
        A, Bi, O = meta[:, 0], meta[:, 1], meta[:, 2]
        f32 = lambda k: S[k].astype(np.float32).reshape(len(S[k]), -1)
        diffs = {"fly": np.abs(f32("last_fly")[A] - f32("first_fly")[Bi]), "shuf": np.abs(f32("last_fly")[A] - f32("first_fly")[O]),
                 "hexin": np.abs(f32("last_hexin")[A] - f32("first_hexin")[Bi]),
                 "app": np.concatenate([np.abs(f32("last_thumb")[A] - f32("first_thumb")[Bi]), np.abs(f32("last_hist")[A] - f32("first_hist")[Bi])], axis=1)}
    return {"video_id": video_id, "X": np.asarray(feats, np.float64).reshape(-1, len(NAMES)), "y": np.asarray(labels, np.int8), "pairs": meta, "D": diffs}


def stitch_video(task: tuple) -> dict:
    """Сшить жизни по вероятностям p (пары A→B), записать scalabel на опорных кадрах, HOTA по видео."""
    video_id, sig_dir, root, label_split, pairs, probs, thr, out_dir = task
    from evaluation.hota_eval import evaluate_hota
    from scripts.run_bdd_video_rate_experiment import scalabel_frame

    z = np.load(Path(sig_dir) / video_id / "signatures.npz")
    tids, outputs = z["track_id"], z["outputs"]
    frames_of: dict[int, set] = {}
    for r in outputs:
        frames_of.setdefault(int(r[5]), set()).add(int(r[0]))
    order = np.argsort(-probs) if len(probs) else []
    used_a, used_b, parent = set(), set(), {}
    def root_of(t):
        seen = set()
        while t in parent and t not in seen:
            seen.add(t); t = parent[t]
        return t
    merged = {t: set(f) for t, f in frames_of.items()}  # кадры объединённой цепочки по корню
    for k in order:
        if probs[k] <= thr:
            break
        a, b = int(pairs[k, 0]), int(pairs[k, 1])
        ra, rb = root_of(int(tids[a])), root_of(int(tids[b]))
        if a in used_a or b in used_b or ra == rb:
            continue
        if merged.get(ra, set()) & merged.get(rb, set()):
            continue  # у цепочек есть общие кадры (трекер позже сам восстановил трек) — одна рамка на ID в кадре
        used_a.add(a); used_b.add(b); parent[rb] = ra
        merged[ra] = merged.get(ra, set()) | merged.pop(rb, set())
    matches = json.loads((Path(root) / video_id / "alignment.json").read_text(encoding="utf-8"))["matches"]
    truth = json.loads((PROJECT_ROOT / "data/bdd100k/labels/box_track_20" / label_split / f"{video_id}.json").read_text(encoding="utf-8"))
    by_frame = {}
    for r in outputs:
        by_frame.setdefault(int(r[0]), []).append(r[1:8].copy())
    rows = []
    for i, m in enumerate(matches):
        out = np.asarray(by_frame.get(int(m["video_frame_index"]), []), np.float64).reshape(-1, 7)
        if len(out):
            out[:, 4] = [root_of(int(t)) for t in out[:, 4]]
        rows.append(scalabel_frame(video_id, truth[i], out))
    result_dir = Path(out_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    save_json(result_dir / f"{video_id}.json", rows)
    metrics = evaluate_hota(result_dir, [video_id], split=label_split)
    return {"video_id": video_id, "links": len(parent), "metrics": {k: metrics.get(k) for k in ("HOTA", "IDF1", "AssA", "IDSw")}}


def fit(X: np.ndarray, y: np.ndarray, cols: list[int]):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000, random_state=42)).fit(X[:, cols], y)


def main() -> int:
    from sklearn.metrics import average_precision_score

    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    selection = json.loads((args.train_cache / "selection.json").read_text(encoding="utf-8"))["videos"]
    ready = lambda d, vids: [v for v in vids if (d / v / "signatures.npz").exists()]
    splits = {"val7": [(v, str(args.sig_val), str(PROJECT_ROOT / "runs/video_rate"), "val") for v in ready(args.sig_val, VAL7)]}
    if not args.val_only:
        for s in ("train", "dev", "test"):
            splits[s] = [(v, str(args.sig_root), str(args.train_cache), "train") for v in ready(args.sig_root, selection[s])]
    data = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for s, tasks in splits.items():
            data[s] = list(pool.map(video_pairs, tasks))
            print(f"[DATA] {s}: {len(tasks)} роликов, пар {sum(len(d['y']) for d in data[s])}, положительных {sum(int(d['y'].sum()) for d in data[s])}", flush=True)
    report = {"ladder": {}, "pairs": {s: {"videos": len(d), "pairs": int(sum(len(x['y']) for x in d)), "positives": int(sum(int(x['y'].sum()) for x in d))} for s, d in data.items()}}
    if args.val_only:
        vids = [d for d in data["val7"] if len(d["y"]) and 0 < d["y"].sum() < len(d["y"])]
        for name, cols in LADDER.items():
            idx = [NAMES.index(c) for c in cols]; aps = {}
            for k, held in enumerate(vids):
                tr = [d for j, d in enumerate(vids) if j != k]
                m = fit(np.concatenate([d["X"] for d in tr]), np.concatenate([d["y"] for d in tr]), idx)
                aps[held["video_id"]] = float(average_precision_score(held["y"], m.predict_proba(held["X"][:, idx])[:, 1]))
            report["ladder"][name] = {"val7_lovo_by_video": aps, "val7_lovo_mean": float(np.mean(list(aps.values())))}
            print(f"[LOVO] {name:36s} AP {report['ladder'][name]['val7_lovo_mean']:.4f}", flush=True)
        save_json(args.output / "report_val7_lovo.json", report)
        return 0
    Xtr = np.concatenate([d["X"] for d in data["train"]]); ytr = np.concatenate([d["y"] for d in data["train"]])
    models = {}
    for name, cols in LADDER.items():
        idx = [NAMES.index(c) for c in cols]
        m = fit(Xtr, ytr, idx); models[name] = (m, idx)
        rec = {}
        for s in ("dev", "test", "val7"):
            X = np.concatenate([d["X"] for d in data[s]]); y = np.concatenate([d["y"] for d in data[s]])
            rec[s] = float(average_precision_score(y, m.predict_proba(X[:, idx])[:, 1]))
            rec[f"{s}_by_video"] = {d["video_id"]: float(average_precision_score(d["y"], m.predict_proba(d["X"][:, idx])[:, 1])) for d in data[s] if len(d["y"]) > 5 and 0 < d["y"].sum() < len(d["y"])}
        report["ladder"][name] = rec
        print(f"[AP] {name:36s} dev {rec['dev']:.4f} test {rec['test']:.4f} val7 {rec['val7']:.4f}", flush=True)
    for s in ("dev", "test", "val7"):
        for p in (("geometry_appearance_hexin", "geometry_appearance_hexin_fly"), ("geometry_appearance_hexin_shuffled", "geometry_appearance_hexin_fly")):
            a, b = report["ladder"][p[0]][f"{s}_by_video"], report["ladder"][p[1]][f"{s}_by_video"]
            d = np.asarray([b[v] - a[v] for v in a if v in b])
            rng = np.random.default_rng(42); boot = d[rng.integers(0, len(d), (5000, len(d)))].mean(1)
            report.setdefault("bootstrap", {})[f"{p[1]}-{p[0]}|{s}"] = {"mean": float(d.mean()), "ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))], "better": int((d > 0).sum()), "worse": int((d < 0).sum())}
            print(f"[BOOT] {p[1]} − {p[0]} {s}: {d.mean():+.4f} [{np.quantile(boot, 0.025):+.4f}; {np.quantile(boot, 0.975):+.4f}] {int((d > 0).sum())}/{int((d < 0).sum())}", flush=True)
    # Обучаемая метрика: поэлементные разности полных векторов (+ все скалярные признаки лестницы без мухи/с мухой).
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    base_cols = [NAMES.index(c) for c in GEOMETRY + APPEAR + HEXIN]
    fly_cols = [NAMES.index(c) for c in FLY]; shuf_cols = [NAMES.index(c) for c in SHUF]
    def mat(split, parts, scalar_cols):
        rows = [np.concatenate([d["X"][:, scalar_cols]] + [d["D"][p] for p in parts], axis=1) for d in data[split] if len(d["y"])]
        return np.concatenate(rows).astype(np.float32), np.concatenate([d["y"] for d in data[split] if len(d["y"])])
    learned = {"L_app": (["app"], base_cols), "L_app_hexin": (["app", "hexin"], base_cols), "L_app_hexin_fly": (["app", "hexin", "fly"], base_cols + fly_cols),
               "L_app_hexin_shuffled": (["app", "hexin", "shuf"], base_cols + shuf_cols), "L_geometry_fly": (["fly"], [NAMES.index(c) for c in GEOMETRY] + fly_cols)}
    for name, (parts, cols) in learned.items():
        Xtr_l, ytr_l = mat("train", parts, cols)
        best = None
        for C in (0.01, 0.1, 1.0):
            m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000, random_state=42)).fit(Xtr_l, ytr_l)
            Xd, yd = mat("dev", parts, cols); ap_dev = float(average_precision_score(yd, m.predict_proba(Xd)[:, 1]))
            if best is None or ap_dev > best[0]:
                best = (ap_dev, C, m)
        rec = {"C": best[1], "dev": best[0]}
        for s_ in ("test", "val7"):
            Xs, ys = mat(s_, parts, cols); rec[s_] = float(average_precision_score(ys, best[2].predict_proba(Xs)[:, 1]))
            byv = {}
            for d in data[s_]:
                if len(d["y"]) > 5 and 0 < d["y"].sum() < len(d["y"]):
                    Xv = np.concatenate([d["X"][:, cols]] + [d["D"][p] for p in parts], axis=1).astype(np.float32)
                    byv[d["video_id"]] = float(average_precision_score(d["y"], best[2].predict_proba(Xv)[:, 1]))
            rec[f"{s_}_by_video"] = byv
        report["ladder"][name] = rec
        print(f"[AP] {name:36s} C={best[1]} dev {rec['dev']:.4f} test {rec['test']:.4f} val7 {rec['val7']:.4f}", flush=True)
    for s_ in ("test", "val7"):
        for p in (("L_app_hexin", "L_app_hexin_fly"), ("L_app_hexin_shuffled", "L_app_hexin_fly")):
            a, b = report["ladder"][p[0]][f"{s_}_by_video"], report["ladder"][p[1]][f"{s_}_by_video"]
            d = np.asarray([b[v] - a[v] for v in a if v in b])
            rng = np.random.default_rng(42); boot = d[rng.integers(0, len(d), (5000, len(d)))].mean(1)
            report["bootstrap"][f"{p[1]}-{p[0]}|{s_}"] = {"mean": float(d.mean()), "ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))], "better": int((d > 0).sum()), "worse": int((d < 0).sum())}
            print(f"[BOOT] {p[1]} − {p[0]} {s_}: {d.mean():+.4f} [{np.quantile(boot, 0.025):+.4f}; {np.quantile(boot, 0.975):+.4f}] {int((d > 0).sum())}/{int((d < 0).sum())}", flush=True)
    # Сшивка: без сшивки (стенд) / голова без мухи / голова с мухой; порог — максимум IDF1 на dev.
    from evaluation.hota_eval import evaluate_hota

    heads = {"stitch_base": models["geometry_appearance_hexin"], "stitch_fly": models["geometry_appearance_hexin_fly"]}
    split_meta = {s_: (tasks[0][1], tasks[0][2], tasks[0][3]) for s_, tasks in splits.items() if tasks}
    def run_stitch(name, head, split, thr):
        m, idx = head
        out_dir = args.output / "stitch" / f"{name}_{thr:.2f}" / split
        tasks = [(d["video_id"], *split_meta[split], d["pairs"][:, :2] if len(d["y"]) else np.zeros((0, 2), np.int32),
                  m.predict_proba(d["X"][:, idx])[:, 1] if len(d["y"]) else np.zeros(0), thr, str(out_dir)) for d in data[split]]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            per = {r["video_id"]: r for r in pool.map(stitch_video, tasks)}
        agg = evaluate_hota(out_dir, [d["video_id"] for d in data[split]], split=split_meta[split][2])
        return {"per_video": per, "summary": {k: agg.get(k) for k in ("HOTA", "IDF1", "AssA", "IDSw")}, "links": int(sum(r["links"] for r in per.values()))}
    stitch = {"none": {s_: run_stitch("none", heads["stitch_base"], s_, 2.0) for s_ in ("dev", "test", "val7")}}
    for name, head in heads.items():
        dev_runs = {thr: run_stitch(name, head, "dev", thr) for thr in (0.2, 0.3, 0.4, 0.5, 0.7)}
        thr = max(dev_runs, key=lambda t: dev_runs[t]["summary"]["IDF1"])
        stitch[name] = {"threshold": thr, "dev": dev_runs[thr], "dev_grid": {str(t): r["summary"] for t, r in dev_runs.items()}, "test": run_stitch(name, head, "test", thr), "val7": run_stitch(name, head, "val7", thr)}
    for name, r in stitch.items():
        print(f"[STITCH] {name:12s} " + " | ".join(f"{s_}: HOTA {r[s_]['summary']['HOTA']:.3f} IDF1 {r[s_]['summary']['IDF1']:.3f} IDSw {r[s_]['summary']['IDSw']:.0f} связей {r[s_]['links']}" for s_ in ("dev", "test", "val7")) + (f" (порог {r['threshold']})" if 'threshold' in r else ""), flush=True)
    for pair in (("none", "stitch_base"), ("none", "stitch_fly"), ("stitch_base", "stitch_fly")):
        for s_ in ("test", "val7"):
            a, b = stitch[pair[0]][s_]["per_video"], stitch[pair[1]][s_]["per_video"]
            for met in ("HOTA", "IDF1", "IDSw"):
                d = np.asarray([b[v]["metrics"][met] - a[v]["metrics"][met] for v in a if v in b], np.float64)
                rng = np.random.default_rng(42); boot = d[rng.integers(0, len(d), (5000, len(d)))].mean(1)
                report.setdefault("stitch_bootstrap", {})[f"{pair[1]}-{pair[0]}|{s_}|{met}"] = {"mean": float(d.mean()), "ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))], "better": int(((d < 0) if met == "IDSw" else (d > 0)).sum()), "worse": int(((d > 0) if met == "IDSw" else (d < 0)).sum())}
            k = f"{pair[1]}-{pair[0]}|{s_}"
            print(f"[SBOOT] {k}: HOTA {report['stitch_bootstrap'][k + '|HOTA']['mean']:+.3f} {report['stitch_bootstrap'][k + '|HOTA']['ci95']}, IDF1 {report['stitch_bootstrap'][k + '|IDF1']['mean']:+.3f} {report['stitch_bootstrap'][k + '|IDF1']['ci95']}, IDSw {report['stitch_bootstrap'][k + '|IDSw']['mean']:+.2f} {report['stitch_bootstrap'][k + '|IDSw']['ci95']}", flush=True)
    report["stitch"] = {n: {k: (v if k in ("threshold", "dev_grid") else {"summary": v["summary"], "links": v["links"], "per_video": {vid: x["metrics"] for vid, x in v["per_video"].items()}}) for k, v in r.items()} for n, r in stitch.items()}
    import pickle
    with open(args.output / "models.pkl", "wb") as fh:
        pickle.dump({k: v for k, v in models.items()}, fh)
    save_json(args.output / "report.json", report)
    print(f"[OK] {args.output / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
