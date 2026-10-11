"""Журнал событий повторного сопоставления потерянных треков ByteTrack (карточка association_log_v1).

Трекер стенда (XYWHByteTrack) на сохранённых детекциях dev 35, буфер 30 и 150, без мухи, на процессоре. Потерянный трек
возвращается только на первом сопоставлении (уверенные детекции, стоимость 1 − IoU(прогноз, детекция) · уверенность,
порог 0,8). Для каждого возврата записывается: время в состоянии «потерян» (кадр возврата − кадр последнего обновления),
последняя наблюдённая рамка (детекция последнего обновления), предсказанная рамка (Калман перед сопоставлением), все
уверенные детекции кадра с IoU к прогнозу и к последнему наблюдению, выбранный кандидат.
Принадлежность по разметке: рамки объекта разметки линейно интерполируются между соседними размеченными кадрами
(alignment.json: размеченный кадр → кадр видео); детекция → объект по венгерскому сопоставлению внутри класса, IoU ≥ 0,5
(crowd пропускается). Объект трека — объект его последнего наблюдения.
Разбор неверных возвратов (выбран другой объект, объект трека известен) — А / Б / В по правилу карточки.
Сверка: выход трекера совпадает с outputs сигнатур (runs/fly_signatures — буфер 30, runs/fly_signatures_buf150 — 150).

Запуск: python scripts/association_log.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

OUT = PROJECT_ROOT / "runs/association_log"
SIG = {30: PROJECT_ROOT / "runs/fly_signatures", 150: PROJECT_ROOT / "runs/fly_signatures_buf150"}


def iou(a, B):
    from scripts.output_filter_oracle import _iou
    return _iou([a], B)[0] if len(B) else np.zeros(0)


def gt_index(gt, vf, valid):
    """Кадр видео → [(объект, класс, рамка)] с линейной интерполяцией между соседними размеченными кадрами объекта."""
    track = {}
    for k, fr in enumerate(gt):
        for l in fr.get("labels") or []:
            if l["category"] in valid and not (l.get("attributes") or {}).get("crowd", False):
                b = l["box2d"]; track.setdefault(str(l["id"]), []).append((vf[k], l["category"], np.array([b["x1"], b["y1"], b["x2"], b["y2"]])))
    by = {}
    for g, s in track.items():
        s.sort(key=lambda x: x[0])
        for (f0, c, b0), (f1, _, b1) in zip(s, s[1:] + [s[-1]]):
            if f1 - f0 > 12 or f1 == f0:          # соседние размеченные кадры (5 к/с → ~6 кадров видео); дальше не интерполируем
                by.setdefault(f0, []).append((g, c, b0)); continue
            for f in range(f0, f1):
                w = (f - f0) / (f1 - f0); by.setdefault(f, []).append((g, c, (1 - w) * b0 + w * b1))
    return by


def label(boxes, cls_names, objs):
    """Детекции кадра → объект разметки (или None): венгерское внутри класса, IoU ≥ 0,5."""
    from scipy.optimize import linear_sum_assignment
    from scripts.output_filter_oracle import _iou
    out = [None] * len(boxes)
    for c in set(cls_names):
        di = [i for i, n in enumerate(cls_names) if n == c]; gi = [j for j, o in enumerate(objs) if o[1] == c]
        if not di or not gi:
            continue
        S = _iou([boxes[i] for i in di], [objs[j][2] for j in gi]); S[S < 0.5] = 0
        r, cc = linear_sum_assignment(-S)
        for a, b in zip(r, cc):
            if S[a, b] > 0:
                out[di[a]] = objs[gi[b]][0]
    return out


def run(task):
    from ultralytics.trackers.basetrack import BaseTrack, TrackState
    from ultralytics.trackers.bot_sort import BOTrack
    from ultralytics.trackers.utils import matching
    from datasets.bdd100k_loader import CLASS_NAMES
    from scripts.freeze_bench import XYWH_CONFIG
    from scripts.output_filter_oracle import VALID
    from scripts.run_bdd_video_rate_experiment import detections_for_frame, load_detection_cache
    from tracking.offline_association import _DetSlice, make_tracker_args
    from tracking.xywh_bytetrack import XYWHByteTrack
    vid, root, split, buf = task
    events = []

    class LogTrack(BOTrack):
        def activate(self, kf, frame_id):
            self.last_obs = self.xyxy.copy(); super().activate(kf, frame_id)   # до activate mean пуст → рамка детекции

        def update(self, new, frame_id):
            self.last_obs = new.xyxy.copy(); super().update(new, frame_id)

        def re_activate(self, new, frame_id, new_id=False):
            self.last_obs = new.xyxy.copy(); super().re_activate(new, frame_id, new_id)

    class LogBT(XYWHByteTrack):
        def init_track(self, results, img=None):
            from ultralytics.trackers.bot_sort import parse_bboxes
            if len(results) == 0:
                return []
            return [LogTrack(x, s, c) for x, s, c in zip(parse_bboxes(results), results.conf, results.cls)]

        def _first_association(self, pool, dets, activated, refind):
            self._pool, self._dets = pool, dets
            return super()._first_association(pool, dets, activated, refind)

        def _apply_match(self, track, det, activated, refind):
            if track.state != TrackState.Tracked and getattr(self, "_dets", None) is not None and any(det is d for d in self._dets):
                D = np.array([d.xyxy for d in self._dets]) if self._dets else np.zeros((0, 4))
                events.append({"frame": self.frame_id - 1, "tid": int(track.track_id), "lost_frames": int(self.frame_id - track.end_frame),
                               "pred": track.xyxy.copy(), "last_obs": track.last_obs.copy(), "last_frame": int(track.end_frame) - 1,
                               "dets": D, "det_cls": [CLASS_NAMES[int(d.cls)] for d in self._dets], "det_score": [float(d.score) for d in self._dets],
                               "chosen": next(i for i, d in enumerate(self._dets) if d is det), "track_cls": CLASS_NAMES[int(track.cls)]})
            super()._apply_match(track, det, activated, refind)

    det = load_detection_cache(PROJECT_ROOT / "runs/train_cache" / vid / "detections" / "raw_video.npz")
    BaseTrack.reset_id()
    tr = LogBT(make_tracker_args(**dict(XYWH_CONFIG, track_buffer=buf)))
    n = int(np.load(SIG[buf] / vid / "signatures.npz")["frame_count"])
    outs, obs = [], {}            # obs: tid → [(кадр, рамка детекции последнего обновления)] — для объекта трека
    for f in range(n):
        xyxy, conf, cls = detections_for_frame(det, f)
        tr._dets = None
        o = tr.update(_DetSlice(xyxy, conf, cls))
        if len(o):
            outs.append(np.column_stack([np.full(len(o), f), np.asarray(o)[:, :7]]).astype(np.float32))
        for t in tr.tracked_stracks:
            if t.frame_id == tr.frame_id:
                obs.setdefault(int(t.track_id), {})[f] = (t.last_obs.copy(), CLASS_NAMES[int(t.cls)])
    ours = np.concatenate(outs) if outs else np.zeros((0, 8), np.float32)
    ref = np.load(SIG[buf] / vid / "signatures.npz")["outputs"]
    same = ours.shape == ref.shape and bool(np.allclose(ours, ref, atol=1e-2))
    # принадлежность по разметке
    gt = json.loads((PROJECT_ROOT / "data/bdd100k/labels/box_track_20" / split / f"{vid}.json").read_text(encoding="utf-8"))
    vf = [int(m["video_frame_index"]) for m in json.loads((Path(root) / vid / "alignment.json").read_text(encoding="utf-8"))["matches"]]
    G = gt_index(gt, vf, VALID)
    rows = []
    for e in events:
        f, lf = e["frame"], e["last_frame"]
        objs = G.get(f, []); lab = label(list(e["dets"]), e["det_cls"], objs) if objs else [None] * len(e["dets"])
        prev = G.get(lf, []); lo = obs.get(e["tid"], {}).get(lf)
        own = label([lo[0]], [lo[1]], prev)[0] if (lo is not None and prev) else None
        ch = e["chosen"]; ip = iou(e["pred"], e["dets"]); il = iou(e["last_obs"], e["dets"])
        sz = float(np.sqrt(max((e["last_obs"][2] - e["last_obs"][0]) * (e["last_obs"][3] - e["last_obs"][1]), 1.0)))
        drift = float(np.linalg.norm((e["pred"][:2] + e["pred"][2:]) / 2 - (e["last_obs"][:2] + e["last_obs"][2:]) / 2) / sz)
        r = {"video": vid, "frame": f, "tid": e["tid"], "lost_frames": e["lost_frames"], "own": own, "chosen_obj": lab[ch],
             "iou_pred_chosen": float(ip[ch]), "iou_obs_chosen": float(il[ch]), "drift_sizes": drift, "size": sz,
             "score_chosen": e["det_score"][ch], "same_cls": e["det_cls"][ch] == e["track_cls"], "n_cand_iou_pred_gt0": int((ip > 0).sum())}
        if own is not None and lab[ch] is not None:
            r["verdict"] = "correct" if lab[ch] == own else "wrong"
        else:
            r["verdict"] = "unverifiable"; r["why"] = "own_unknown" if own is None else "chosen_unmatched"
        if r["verdict"] == "wrong":
            k = [i for i, x in enumerate(lab) if x == own]
            if not k:
                r["case"] = "A"
            else:
                i = k[0]; r["iou_pred_correct"] = float(ip[i]); r["iou_obs_correct"] = float(il[i]); r["score_correct"] = e["det_score"][i]
                r["case"] = "B" if il[i] > il[ch] else "C"
        rows.append(r)
    return vid, buf, same, len(ours), rows


def bucket(x):
    return "le30" if x <= 30 else ("31-60" if x <= 60 else ("61-150" if x <= 150 else "gt150"))


def main() -> int:
    from scripts.fly_lifecycle_alive import sets
    OUT.mkdir(parents=True, exist_ok=True)
    S = sets(PROJECT_ROOT / "runs/fly_signatures", PROJECT_ROOT / "runs/fly_signatures_val")
    dev = [(v, str(rt), sp) for v, sr, rt, sp in S["dev"]]
    vids = sys.argv[1:] or [v for v, _, _ in dev]
    tasks = [(v, rt, sp, b) for v, rt, sp in dev if v in vids for b in (30, 150)]
    with ProcessPoolExecutor(max_workers=4) as pool:
        got = list(pool.map(run, tasks))
    rep = {"videos": len(vids), "reproduced": {f"{v}|{b}": s for v, b, s, _, _ in got}, "by_buffer": {}}
    rows = {b: [r for v, bb, s, n, rs in got if bb == b for r in rs] for b in (30, 150)}
    for b, rs in rows.items():
        d = {}
        for k in ("le30", "31-60", "61-150", "gt150"):
            g = [r for r in rs if bucket(r["lost_frames"]) == k]
            c = Counter(r["verdict"] for r in g); cs = Counter(r.get("case") for r in g if r["verdict"] == "wrong")
            ver = c["correct"] + c["wrong"]
            d[k] = {"events": len(g), **dict(c), "unverifiable_why": dict(Counter(r.get("why") for r in g if r["verdict"] == "unverifiable")), "correct_share_verified": round(c["correct"] / ver, 3) if ver else None, "wrong_cases": {x: cs.get(x, 0) for x in "ABC"}}
        rep["by_buffer"][b] = d
    w = [r for r in rows[150] if r["verdict"] == "wrong" and r["lost_frames"] > 30]
    cs = Counter(r["case"] for r in w)
    rep["rule"] = {"wrong_gt30_b150": len(w), **{x: cs.get(x, 0) for x in "ABC"},
                   "B_share": round(cs.get("B", 0) / len(w), 3) if w else None, "AC_share": round((cs.get("A", 0) + cs.get("C", 0)) / len(w), 3) if w else None}
    for name, sel in (("correct_gt30_b150", [r for r in rows[150] if r["verdict"] == "correct" and r["lost_frames"] > 30]), ("wrong_gt30_b150", w)):
        if sel:
            rep[name] = {k: round(float(np.median([r[k] for r in sel])), 3) for k in ("lost_frames", "drift_sizes", "iou_pred_chosen", "iou_obs_chosen", "score_chosen", "n_cand_iou_pred_gt0")}
            rep[name]["same_cls_share"] = round(float(np.mean([r["same_cls"] for r in sel])), 3)
    (OUT / "events.json").write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    (OUT / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in rep.items() if k != "reproduced"}, ensure_ascii=False, indent=1))
    print(f"[REPRO] совпало {sum(rep['reproduced'].values())}/{len(rep['reproduced'])}")
    print(f"[OK] {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
