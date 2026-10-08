"""Сшивка обрывков v2 (карточка stitch_appearance_models_v1): кандидаты без разметки, внешность DINOv3-S и MASA-R50, муха.

Исправление по замечанию автора 08.10: в scripts/eval_fly_reid.py:video_pairs пара-кандидат допускалась, только если оба конца
сопоставлены с эталоном — сшиватель видел только треки настоящих объектов (утечка разметки в список кандидатов). Здесь
кандидаты — все пары жизней «конец A → начало B» в окне (разрыв ≤ G кадров, расстояние ≤ D размеров рамки); метка — 1/0, если
оба конца сопоставлены с эталоном, иначе −1 (пара не участвует в обучении и AP, но участвует в сшивке).
Признаки пары: геометрия (как в eval_fly_reid), простая внешность (миниатюра, гистограмма), вход глаза, муха (косинусы по
группам типов), DINOv3-S (runs/dino_embeddings), MASA-R50 (runs/masa_embeddings) — косинус эмбеддингов конца A и начала B.
Головы — логистическая регрессия на train 130; порог сшивки — максимум HOTA на dev 35; проверка — test 35 и val7;
сшиватель — scripts/eval_fly_reid.py:stitch_video (жадный, без общих кадров у цепочек).

Запуск: python scripts/stitch_eval_v2.py [--max-gap 150 --max-dist 5]
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

OUT = PROJECT_ROOT / "runs" / "stitch_v2"
GEO = ["gap", "dist", "log_size_ratio", "same_class", "a_len", "b_len"]
APP = ["thumb_cos", "hist_l1"]
HEX = ["hexin_cos"]
FLY = ["fly_cos", "fly_T123", "fly_T4", "fly_T5", "fly_Tm", "fly_TmY"]
DINO = ["dino_cos"]
MASA = ["masa_cos"]
NAMES = GEO + APP + HEX + FLY + DINO + MASA
HEADS = {"geometry": GEO, "base_app_hexin": GEO + APP + HEX, "dino": GEO + DINO, "masa": GEO + MASA, "dino_masa": GEO + DINO + MASA,
         "masa_fly": GEO + MASA + FLY, "base_fly": GEO + APP + HEX + FLY, "all": GEO + APP + HEX + DINO + MASA, "all_fly": GEO + APP + HEX + DINO + MASA + FLY}
STITCH_HEADS = ["geometry", "base_app_hexin", "dino", "masa", "masa_fly", "all", "all_fly"]
WIN = {"gap": 150, "dist": 5.0}


def _set(g, d):
    WIN["gap"], WIN["dist"] = g, d


def video_pairs(task):
    vid, sig_dir, root, split = task
    from scripts.eval_fly_reid import cos, fly_feats
    from scripts.evaluate_flyvis_qmatch_gate import truth_arrays
    from tracking.flyvis_tracker import box_iou
    z = np.load(Path(sig_dir) / vid / "signatures.npz"); S = {k: z[k] for k in z.files}
    dz = np.load(PROJECT_ROOT / "runs/dino_embeddings" / f"{vid}.npz"); mz = np.load(PROJECT_ROOT / "runs/masa_embeddings" / f"{vid}.npz")
    outputs = S["outputs"]
    matches = json.loads((Path(root) / vid / "alignment.json").read_text(encoding="utf-8"))["matches"]
    ref_of_raw = {int(m["video_frame_index"]): i for i, m in enumerate(matches)}
    truth = json.loads((PROJECT_ROOT / "data/bdd100k/labels/box_track_20" / split / f"{vid}.json").read_text(encoding="utf-8"))
    rows_by_frame = {}
    for r in outputs:
        rows_by_frame.setdefault(int(r[0]), []).append(r)
    ref_frames = sorted(f for f in ref_of_raw if f in rows_by_frame)

    def gt_of(tid, frames):
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

    n = len(S["track_id"]); end_gt, start_gt = [], []
    for i in range(n):
        tid, s, e = int(S["track_id"][i]), int(S["start"][i]), int(S["end"][i])
        within = [f for f in ref_frames if s <= f <= e]
        end_gt.append(gt_of(tid, within[::-1])); start_gt.append(gt_of(tid, within))
    dl, df = dz["last"].astype(np.float32), dz["first"].astype(np.float32)
    ml, mf = mz["last"].astype(np.float32), mz["first"].astype(np.float32)
    feats, labels, meta = [], [], []
    for a in range(n):
        for b in range(n):
            if S["track_id"][a] == S["track_id"][b]:
                continue
            gap = int(S["start"][b]) - int(S["end"][a])
            if not (0 < gap <= WIN["gap"]):
                continue
            A, B = S["last_box"][a].astype(np.float64), S["first_box"][b].astype(np.float64)
            sa = np.sqrt(max((A[2] - A[0]) * (A[3] - A[1]), 1.0)); sb = np.sqrt(max((B[2] - B[0]) * (B[3] - B[1]), 1.0))
            dist = np.linalg.norm((A[:2] + A[2:]) / 2 - (B[:2] + B[2:]) / 2) / sa
            if dist > WIN["dist"]:
                continue
            geo = [gap / 30.0, dist, float(np.log(sb / sa)), float(S["cls"][a] == S["cls"][b]), float(S["n_last"][a]) / 5, float(S["n_first"][b]) / 5]
            app = [cos(S["last_thumb"][a].astype(np.float64), S["first_thumb"][b].astype(np.float64)), float(np.abs(S["last_hist"][a].astype(np.float64) - S["first_hist"][b].astype(np.float64)).sum())]
            hx = [cos(S["last_hexin"][a].astype(np.float64), S["first_hexin"][b].astype(np.float64))]
            fl = fly_feats(S["last_fly"][a], S["first_fly"][b])
            feats.append(geo + app + hx + fl + [float(dl[a] @ df[b])] + [float(ml[a] @ mf[b])])
            labels.append(int(end_gt[a] == start_gt[b]) if end_gt[a] is not None and start_gt[b] is not None else -1)
            meta.append((a, b))
    return {"video_id": vid, "X": np.asarray(feats, np.float64).reshape(-1, len(NAMES)), "y": np.asarray(labels, np.int8), "pairs": np.asarray(meta, np.int32).reshape(-1, 2)}


def main() -> int:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from evaluation.hota_eval import evaluate_hota
    from scripts.eval_fly_reid import stitch_video
    from scripts.fly_lifecycle_alive import sets
    ap_ = argparse.ArgumentParser(); ap_.add_argument("--max-gap", type=int, default=150); ap_.add_argument("--max-dist", type=float, default=5.0)
    ap_.add_argument("--thresholds", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]); a = ap_.parse_args()
    out = OUT / f"gap{a.max_gap}_dist{a.max_dist:g}"; out.mkdir(parents=True, exist_ok=True)
    S = sets(PROJECT_ROOT / "runs/fly_signatures", PROJECT_ROOT / "runs/fly_signatures_val")
    data = {}
    with ProcessPoolExecutor(max_workers=8, initializer=_set, initargs=(a.max_gap, a.max_dist)) as pool:
        for s in ("train", "dev", "test", "val7"):
            data[s] = list(pool.map(video_pairs, [(v, str(sr), str(rt), sp) for v, sr, rt, sp in S[s]]))
            n = sum(len(d["y"]) for d in data[s]); kn = sum(int((d["y"] >= 0).sum()) for d in data[s])
            print(f"[DATA] {s}: кандидатов {n}, из них с меткой {kn}, положительных {sum(int((d['y'] == 1).sum()) for d in data[s])}", flush=True)
    meta = {s: (S[s][0][1], S[s][0][2], S[s][0][3]) for s in data}
    Xtr = np.concatenate([d["X"][d["y"] >= 0] for d in data["train"]]); ytr = np.concatenate([d["y"][d["y"] >= 0] for d in data["train"]])
    models, rep = {}, {"window": vars(a), "AP": {}, "stitch": {}, "diff": {}}
    for h, cols in HEADS.items():
        idx = [NAMES.index(c) for c in cols]
        models[h] = (make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)).fit(Xtr[:, idx], ytr), idx)
        rep["AP"][h] = {}
        for s in ("dev", "test", "val7"):
            X = np.concatenate([d["X"][d["y"] >= 0] for d in data[s]]); y = np.concatenate([d["y"][d["y"] >= 0] for d in data[s]])
            rep["AP"][h][s] = float(average_precision_score(y, models[h][0].predict_proba(X[:, idx])[:, 1]))
        print(f"[AP] {h:16s} dev {rep['AP'][h]['dev']:.4f} test {rep['AP'][h]['test']:.4f} val7 {rep['AP'][h]['val7']:.4f}", flush=True)

    def run(h, s, thr):
        m, idx = models[h]; d_out = out / "stitch" / f"{h}_{thr:g}" / s
        tasks = [(d["video_id"], *meta[s], d["pairs"], m.predict_proba(d["X"][:, idx])[:, 1] if len(d["y"]) else np.zeros(0), thr, str(d_out)) for d in data[s]]
        with ProcessPoolExecutor(max_workers=8) as pool:
            per = {r["video_id"]: r["metrics"] for r in pool.map(stitch_video, tasks)}
        return per
    base = {s: run("geometry", s, 2.0) for s in ("dev", "test", "val7")}   # порог 2 — без сшивки
    rep["stitch"]["none"] = {s: float(np.mean([x["HOTA"] for x in base[s].values()])) for s in base}
    per_head = {}
    for h in STITCH_HEADS:
        dev = {t: run(h, "dev", t) for t in a.thresholds}
        thr = max(dev, key=lambda t: np.mean([x["HOTA"] for x in dev[t].values()]))
        per_head[h] = {"dev": dev[thr], "test": run(h, "test", thr), "val7": run(h, "val7", thr)}
        rep["stitch"][h] = {"threshold": thr, **{s: float(np.mean([x["HOTA"] for x in per_head[h][s].values()])) for s in ("dev", "test", "val7")},
                            "dev_grid": {str(t): float(np.mean([x["HOTA"] for x in r.values()])) for t, r in dev.items()}}
        print(f"[STITCH] {h:16s} порог {thr}: HOTA dev {rep['stitch'][h]['dev']:.3f} test {rep['stitch'][h]['test']:.3f} val7 {rep['stitch'][h]['val7']:.3f} (без сшивки test {rep['stitch']['none']['test']:.3f})", flush=True)
        (out / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    rng = np.random.default_rng(1)
    pairs = [(h, "none") for h in STITCH_HEADS] + [("masa", "dino"), ("masa", "geometry"), ("dino", "geometry"), ("masa", "base_app_hexin"), ("dino", "base_app_hexin"), ("masa_fly", "masa"), ("all_fly", "all"), ("all", "masa")]
    for x, y in pairs:
        for s in ("test", "val7"):
            A = per_head[x][s]; B = base[s] if y == "none" else per_head[y][s]
            for met in ("HOTA", "IDF1"):
                d = np.array([A[v][met] - B[v][met] for v in A]); b = d[rng.integers(0, len(d), (5000, len(d)))].mean(1)
                rep["diff"][f"{x}-{y}|{s}|{met}"] = {"mean": float(d.mean()), "ci95": [float(np.quantile(b, .025)), float(np.quantile(b, .975))], "better": int((d > 0).sum()), "n": len(d)}
            r = rep["diff"][f"{x}-{y}|{s}|HOTA"]
            print(f"[DIFF] {x} − {y} {s}: HOTA {r['mean']:+.3f} [{r['ci95'][0]:+.3f}; {r['ci95'][1]:+.3f}] лучше {r['better']}/{r['n']}", flush=True)
    (out / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print("[OK]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
