"""Трекер на мухе, этап 1: «жив ли потерянный трек» по отпечатку мухи вместо таймера ByteTrack (карточка fly_tracker_lifecycle_v1).

Событие — конец жизни глаза мухи (трекер-стенд перестал выдавать трек; сигнатуры runs/fly_signatures*, тот же
трекер XYWHByteTrack, что и стенд). Метка (по разметке, только для оценки): объект, которому соответствовал трек на
последнем размеченном кадре жизни (тот же класс, IoU ≥ 0,5), присутствует в разметке хотя бы на одном размеченном
кадре в (end, end + 30] — «жив» (1), иначе «ушёл» (0). События без сопоставленного эталона или без размеченного
кадра в окне пропускаются.

Признаки на момент потери: геометрия (рамка, размер, положение, класс, длина жизни, скорость по последним выходам);
муха — last_fly (34 типа × 19 макроколонок); вход глаза — last_hexin (721 фасетка); внешность — last_thumb + last_hist.
Головы (логистическая регрессия, стандартизация; обучение — train, оценка — dev / test / val7, бутстреп по видео):
  geometry, geometry_fly, geometry_hexin, geometry_hexin_fly, geometry_appearance.
Трекер (--track): стенд; потерянный трек выдаётся по предсказанию Калмана ≤ --coast кадров, только если голова
говорит «жив» (p ≥ порога, выбранного на dev по F1). Сравнение HOTA с базой и со слепым продолжением.

Запуск:
    python scripts/fly_lifecycle_alive.py --heads              # головы, AP
    python scripts/fly_lifecycle_alive.py --track --coast 3    # трекер, HOTA
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.oracle_detection_ladder import CATS, VAL7, iou_mat  # noqa: E402

OUT = PROJECT_ROOT / "runs" / "fly_lifecycle"
WINDOW = 30
HEADS = {"geometry": ["geo"], "geometry_fly": ["geo", "fly"], "geometry_hexin": ["geo", "hexin"],
         "geometry_hexin_fly": ["geo", "hexin", "fly"], "geometry_appearance": ["geo", "app"]}


def sets(sig_root: Path, sig_val: Path):
    sel = json.loads((PROJECT_ROOT / "runs/train_cache/selection.json").read_text(encoding="utf-8"))["videos"]
    tc = PROJECT_ROOT / "runs/train_cache"
    out = {s: [(v, sig_root, tc, "train") for v in sel[s]] for s in ("train", "dev", "test")}
    out["val7"] = [(v, sig_val, PROJECT_ROOT / "runs/video_rate", "val") for v in VAL7]
    # сигнатуры собраны не для всех видео выборки (140 из 200 train_cache) — берутся только имеющиеся
    out = {s: [t for t in ts if (Path(t[1]) / t[0] / "signatures.npz").exists()] for s, ts in out.items()}
    print("видео с сигнатурами:", {s: len(ts) for s, ts in out.items()}, flush=True)
    return out


def video_events(task):
    vid, sig_root, root, split = task
    z = np.load(Path(sig_root) / vid / "signatures.npz")
    outs = z["outputs"]
    matches = json.loads((Path(root) / vid / "alignment.json").read_text(encoding="utf-8"))["matches"]
    truth = json.loads((PROJECT_ROOT / "data/bdd100k/labels/box_track_20" / split / f"{vid}.json").read_text(encoding="utf-8"))
    lab_r = np.array([int(m["video_frame_index"]) for m in matches])
    gt = []
    for fr in truth:
        b, c, ids = [], [], []
        for l in fr.get("labels") or []:
            if l.get("box2d") and l.get("category") in CATS:
                bb = l["box2d"]; b.append([bb["x1"], bb["y1"], bb["x2"], bb["y2"]]); c.append(CATS[l["category"]]); ids.append(l["id"])
        gt.append((np.array(b, np.float32).reshape(-1, 4), np.array(c), ids))
    rows = []
    for k in range(len(z["track_id"])):
        tid, st, en = int(z["track_id"][k]), int(z["start"][k]), int(z["end"][k])
        if ((outs[:, 0] == en) & (outs[:, 5] == tid)).any():
            continue  # жизнь глаза оборвалась, а трек выдаётся дальше — это не потеря трека
        inlife = np.flatnonzero((lab_r >= st) & (lab_r <= en))
        after = np.flatnonzero((lab_r > en) & (lab_r <= en + WINDOW))
        if not len(inlife) or not len(after):
            continue
        i0 = inlife[-1]; r0 = lab_r[i0]
        o = outs[(outs[:, 0] == r0) & (outs[:, 5] == tid)]
        if not len(o):
            continue
        gb, gc, gids = gt[i0]
        if not len(gb):
            continue
        m = np.where(gc == int(z["cls"][k]), iou_mat(o[:1, 1:5], gb)[0], 0.0)
        if m.max() < 0.5:
            continue
        g = gids[int(m.argmax())]
        alive = int(any(g in gt[i][2] for i in after))
        tr = outs[outs[:, 5] == tid]; tr = tr[(tr[:, 0] <= en) & (tr[:, 0] >= max(st, en - 5))]
        tr = tr[np.argsort(tr[:, 0])]
        lb = z["last_box"][k].astype(np.float32); w, h = max(lb[2] - lb[0], 1.0), max(lb[3] - lb[1], 1.0)  # вырожденные рамки
        if len(tr) >= 2:
            dt = max(tr[-1, 0] - tr[0, 0], 1); v = ((tr[-1, 1:5] - tr[0, 1:5]) / dt)
        else:
            v = np.zeros(4, np.float32)
        geo = np.r_[lb / np.array([1280, 720, 1280, 720]), np.log1p(w * h), w / max(h, 1), np.log1p(en - st), (en - st) / 30.0,
                    v / max(np.hypot(w, h), 1), np.eye(8)[int(z["cls"][k]) % 8], float(z["n_last"][k])]
        rows.append(dict(vid=vid, tid=tid, end=en, alive=alive, geo=geo.astype(np.float32),
                         fly=z["last_fly"][k].astype(np.float32).ravel(), hexin=z["last_hexin"][k].astype(np.float32),
                         app=np.r_[z["last_thumb"][k], z["last_hist"][k]].astype(np.float32)))
    return vid, rows


def build_X(rows, parts):
    return np.nan_to_num(np.stack([np.concatenate([r[p] for p in parts]) for r in rows]), nan=0.0, posinf=0.0, neginf=0.0)


def ap(y, s):
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(y, s)) if 0 < y.sum() < len(y) else float("nan")


def fit_heads(ev, out_dir: Path):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    tr = [r for v in ev["train"] for r in ev["train"][v]]
    y = np.array([r["alive"] for r in tr])
    models, res = {}, {"n_train": len(tr), "alive_share_train": float(y.mean()), "heads": {}, "bootstrap": {}}
    for name, parts in HEADS.items():
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=2000))
        clf.fit(build_X(tr, parts), y); models[name] = clf
    rng = np.random.default_rng(0)
    probs = {}
    for s in ("dev", "test", "val7"):
        vids = [v for v in ev[s] if ev[s][v]]
        per = {}
        for name, parts in HEADS.items():
            pv = {}
            for v in vids:
                rv = ev[s][v]; p = models[name].predict_proba(build_X(rv, parts))[:, 1]
                pv[v] = (np.array([r["alive"] for r in rv]), p)
                for r, q in zip(rv, p):
                    probs.setdefault(name, {})[(v, r["tid"], r["end"])] = float(q)
            per[name] = pv
            yy = np.concatenate([pv[v][0] for v in vids]); pp = np.concatenate([pv[v][1] for v in vids])
            res["heads"][f"{name}|{s}"] = {"AP": ap(yy, pp), "n": int(len(yy)), "alive_share": float(yy.mean())}
        for a, b in (("geometry_fly", "geometry"), ("geometry_hexin", "geometry"), ("geometry_hexin_fly", "geometry_hexin"), ("geometry_appearance", "geometry"), ("geometry_fly", "geometry_appearance")):
            bs = []
            for _ in range(2000):
                pick = [vids[i] for i in rng.integers(0, len(vids), len(vids))]
                ya = np.concatenate([per[a][v][0] for v in pick]); pa_ = np.concatenate([per[a][v][1] for v in pick]); pb_ = np.concatenate([per[b][v][1] for v in pick])
                bs.append(ap(ya, pa_) - ap(ya, pb_))
            bs = np.array([x for x in bs if np.isfinite(x)])
            d = res["heads"][f"{a}|{s}"]["AP"] - res["heads"][f"{b}|{s}"]["AP"]
            res["bootstrap"][f"{a}-{b}|{s}"] = {"mean": float(d), "ci95": [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]}
    # порог «жив» — максимум F1 на dev для каждой головы
    from sklearn.metrics import precision_recall_curve
    thr = {}
    for name in HEADS:
        yy = np.array([r["alive"] for v in ev["dev"] for r in ev["dev"][v]])
        pp = np.array([probs[name][(v, r["tid"], r["end"])] for v in ev["dev"] for r in ev["dev"][v]])
        pr, rc, th = precision_recall_curve(yy, pp)
        f1 = 2 * pr * rc / np.maximum(pr + rc, 1e-9)
        thr[name] = float(th[int(np.nanargmax(f1[:-1]))]) if len(th) else 0.5
    res["threshold_f1_dev"] = thr
    thr_del = {}
    for name in HEADS:
        yy = np.array([r["alive"] for v in ev["dev"] for r in ev["dev"][v]])
        pp = np.array([probs[name][(v, r["tid"], r["end"])] for v in ev["dev"] for r in ev["dev"][v]])
        best = 0.0
        for tau in np.unique(np.quantile(pp, np.linspace(0.001, 0.5, 300))):
            sel = pp < tau
            if sel.sum() >= 20 and (yy[sel] == 0).mean() >= 0.8:
                best = max(best, float(tau))
        thr_del[name] = best
    res["threshold_delete_dev"] = thr_del
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "probs.pkl", "wb") as fh:
        pickle.dump({"probs": probs, "thr": thr, "thr_del": thr_del}, fh)
    (out_dir / "heads.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    return res


def track_video(task):
    vid, root, split, out_dir, probs_path, head, coast, mode = task
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from ultralytics.trackers.basetrack import BaseTrack
    from evaluation.hota_eval import evaluate_hota
    from scripts.freeze_bench import XYWH_CONFIG
    from scripts.run_bdd_video_rate_experiment import detections_for_frame, load_detection_cache, save_json, scalabel_frame
    from tracking.offline_association import _DetSlice, make_tracker_args
    from tracking.xywh_bytetrack import XYWHByteTrack

    tag = f"{head}_c{coast}" + ("" if mode == "pred" else f"_{mode}")
    per_path = Path(out_dir) / tag / "per_video" / f"{vid}.json"
    if per_path.exists():
        return tag, vid, json.loads(per_path.read_text(encoding="utf-8"))
    P = pickle.load(open(probs_path, "rb")) if head not in ("base", "blind") else None
    det = load_detection_cache(Path(root) / vid / "detections" / "raw_video.npz")
    BaseTrack.reset_id()
    tracker = XYWHByteTrack(make_tracker_args(**XYWH_CONFIG))
    by = {}
    for t in range(int(det["frame_count"])):
        xyxy, conf, cls = detections_for_frame(det, t)
        out = np.asarray(tracker.update(_DetSlice(xyxy, conf, cls)))
        rows = [out[:, :7]] if len(out) else []
        if mode == "lowscore_del" and head not in ("base", "blind"):
            for tr in list(tracker.lost_stracks):
                if tracker.frame_id - tr.end_frame == 1:
                    q = P["probs"][head].get((vid, tr.track_id, tr.end_frame))
                    if q is not None and q < P["thr_del"][head]:
                        tr.mark_removed(); tracker.lost_stracks.remove(tr); tracker.removed_stracks.append(tr)
        if head != "base" and coast:
            extra = []
            for tr in tracker.lost_stracks:
                gap = tracker.frame_id - tr.end_frame
                if not (tr.is_activated and 0 < gap <= coast):
                    continue
                end_raw = tr.end_frame  # совпадает с end жизни глаза в сигнатуре (первый кадр потери; проверено на 0001542f)
                if head != "blind":
                    q = P["probs"][head].get((vid, tr.track_id, end_raw))
                    if q is None or q < P["thr"][head]:
                        continue
                if mode == "pred":
                    extra.append([*tr.xyxy.tolist(), tr.track_id, tr.score, tr.cls])
                    continue
                # mode lowscore: рамка слабого кандидата детектора (0,01–0,25, тот же класс, IoU ≥ 0,3 к предсказанию),
                # не занятого выходами трекера и другими продолжениями
                weak = np.flatnonzero((conf < 0.25) & (cls.astype(int) == int(tr.cls)))
                if not len(weak):
                    continue
                parts = ([out[:, :4]] if len(out) else []) + ([np.asarray([x[:4] for x in extra])] if extra else [])
                busy = np.vstack(parts) if parts else np.zeros((0, 4))
                cand = weak[(iou_mat(tr.xyxy[None].astype(np.float32), xyxy[weak])[0] >= 0.3)]
                if len(busy) and len(cand):
                    cand = cand[iou_mat(xyxy[cand], busy.astype(np.float32)).max(1) < 0.5]
                if not len(cand):
                    continue
                j = cand[int(np.argmax(iou_mat(tr.xyxy[None].astype(np.float32), xyxy[cand])[0]))]
                extra.append([*xyxy[j].tolist(), tr.track_id, float(conf[j]), tr.cls])
            if extra:
                rows.append(np.asarray(extra, np.float64))
        if rows:
            by[t] = np.vstack(rows).astype(np.float64)
    matches = json.loads((Path(root) / vid / "alignment.json").read_text(encoding="utf-8"))["matches"]
    truth = json.loads((PROJECT_ROOT / "data/bdd100k/labels/box_track_20" / split / f"{vid}.json").read_text(encoding="utf-8"))
    res_dir = Path(out_dir) / tag / split; res_dir.mkdir(parents=True, exist_ok=True)
    save_json(res_dir / f"{vid}.json", [scalabel_frame(vid, truth[i], by.get(int(m["video_frame_index"]), np.zeros((0, 7)))) for i, m in enumerate(matches)])
    met = evaluate_hota(res_dir, [vid], split=split)
    r = {k: met.get(k) for k in ("HOTA", "DetA", "AssA", "IDF1", "IDSw")}
    per_path.parent.mkdir(parents=True, exist_ok=True); per_path.write_text(json.dumps(r), encoding="utf-8")
    return tag, vid, r


def main() -> int:
    ap_ = argparse.ArgumentParser(description=__doc__)
    ap_.add_argument("--sig-root", type=Path, default=PROJECT_ROOT / "runs/fly_signatures")
    ap_.add_argument("--sig-val", type=Path, default=PROJECT_ROOT / "runs/fly_signatures_val")
    ap_.add_argument("--output", type=Path, default=OUT / "m000")
    ap_.add_argument("--heads", action="store_true")
    ap_.add_argument("--track", action="store_true")
    ap_.add_argument("--coast", type=int, default=3)
    ap_.add_argument("--mode", default="pred", choices=["pred", "lowscore", "lowscore_del"], help="чем продолжать потерянный трек: предсказанием Калмана или рамкой слабой детекции")
    ap_.add_argument("--workers", type=int, default=8)
    args = ap_.parse_args()
    S = sets(args.sig_root, args.sig_val)
    if args.heads:
        ev = {}
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for s, tasks in S.items():
                ev[s] = dict(pool.map(video_events, [(v, str(sr), str(rt), sp) for v, sr, rt, sp in tasks]))
        res = fit_heads(ev, args.output)
        print(f"событий train {res['n_train']}, доля «жив» {res['alive_share_train']:.3f}")
        for k, v in res["heads"].items():
            print(f"  {k:32s} AP {v['AP']:.4f} (n {v['n']}, жив {v['alive_share']:.2f})")
        for k, v in res["bootstrap"].items():
            print(f"  Δ {k:40s} {v['mean']:+.4f} [{v['ci95'][0]:+.4f}; {v['ci95'][1]:+.4f}]")
    if args.track:
        from evaluation.hota_eval import evaluate_hota
        heads = ["base", "blind", "geometry", "geometry_fly", "geometry_hexin", "geometry_hexin_fly"]
        tasks = [(v, str(rt), sp, str(args.output), str(args.output / "probs.pkl"), h, args.coast, args.mode) for s in ("test", "val7") for v, sr, rt, sp in S[s] for h in heads]
        per = {}
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for tag, vid, r in pool.map(track_video, tasks):
                per.setdefault(tag, {})[vid] = r
        rep, rng = {}, np.random.default_rng(0)
        for s in ("test", "val7"):
            vids = [v for v, *_ in S[s]]; split = S[s][0][3]
            sfx = "" if args.mode == "pred" else f"_{args.mode}"
            base = f"base_c{args.coast}{sfx}"
            for h in heads:
                tag = f"{h}_c{args.coast}{sfx}"
                agg = evaluate_hota(args.output / tag / split, vids, split=split)
                d = np.array([per[tag][v]["HOTA"] - per[base][v]["HOTA"] for v in vids])
                b = d[rng.integers(0, len(d), (5000, len(d)))].mean(1)
                rep[f"{tag}|{s}"] = {**{k: agg.get(k) for k in ("HOTA", "DetA", "AssA", "IDF1", "IDSw")}, "dHOTA": float(d.mean()), "ci95": [float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))], "better": int((d > 0).sum()), "worse": int((d < 0).sum())}
                r = rep[f"{tag}|{s}"]
                print(f"{s:5s} {tag:26s} HOTA {r['HOTA']:.3f} DetA {r['DetA']:.3f} AssA {r['AssA']:.3f} IDSw {r['IDSw']:.0f}  Δ {r['dHOTA']:+.3f} [{r['ci95'][0]:+.3f}; {r['ci95'][1]:+.3f}] {r['better']}/{r['worse']}")
        (args.output / f"track_c{args.coast}{'' if args.mode == 'pred' else '_' + args.mode}.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
