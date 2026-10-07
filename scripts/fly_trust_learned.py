"""Обучаемое доверие к движению мухи с контролем без признаков мухи (карточка fly_motion_trust_learned_v1).

Запускается, только если дешёвый вариант (fly_motion_trust_cheap_v1) не прошёл проверку на отложенных видео.
Схема 5 Гц (motion_scheme_search_v1). На видео подбора (trust_split.json → tune) трекер идёт с чистым Калманом и для
каждого трека на каждом размеченном кадре записывает признаки и метку «поправка мухи ближе к эталону, чем Калман»
(как в fly_motion_trust_v1, по разметке). Два модуля — логистическая регрессия:
  with_fly — уверенность поля мухи, величина поправки, расхождение поправки с Калманом, размер рамки, скорость Калмана,
             потерян ли трек, длина трека;
  no_fly   — то же без признаков мухи (размер рамки, скорость Калмана, потерян ли трек, длина трека): контроль.
Применение: поправка мухи берётся, если вероятность > 0,5, иначе Калман. Проверка — 17 других видео dev и val7;
критерий — парная разность HOTA с «без движения» на 24 отложенных видео > 0, ДИ > 0, и with_fly − no_fly > 0.

Запуск: python scripts/fly_trust_learned.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import scripts.motion_scheme_search as M  # noqa: E402

OUT = PROJECT_ROOT / "runs/motion_scheme_search"
FLY_FEATS = [0, 1, 2, 3, 4, 5, 6]
NOFLY_FEATS = [3, 4, 5, 6]


class Logged(M.MotionXYWH):
    """Применяет политику gate(признаки) → брать ли поправку; при сборе данных — пишет признаки и метку оракула."""

    def __init__(self, args, apply_to="all"):
        super().__init__(args, apply_to)
        self.gate = None; self.collect = None; self.label = None

    def multi_predict(self, tracks):
        from ultralytics.trackers.basetrack import TrackState
        from ultralytics.trackers.bot_sort import BOTrack
        if not tracks:
            return
        before_c = [t.mean[:2].copy() for t in tracks]; before_b = [t.xyxy.copy() for t in tracks]
        BOTrack.multi_predict(tracks)
        if self.motion is None:
            return
        for t, c0, b0 in zip(tracks, before_c, before_b):
            if not (np.isfinite(b0).all() and b0[2] - b0[0] > 1 and b0[3] - b0[1] > 1):
                continue
            vec, conf = self.motion(b0)
            kal = t.mean[:2] - c0
            x = np.array([conf, np.linalg.norm(vec), np.linalg.norm(vec - kal), np.sqrt((b0[2] - b0[0]) * (b0[3] - b0[1])),
                          np.linalg.norm(kal), float(t.state == TrackState.Lost), float(t.tracklet_len)], np.float64)
            if self.collect is not None and self.label is not None:
                y = self.label(b0, c0 + kal, c0 + vec)
                if y is not None:
                    self.collect.append((x, y))
            if self.gate is not None and self.gate(x):
                t.mean[:2] = c0 + vec


def make_label(prev_gt, cur_gt):
    def lab(b0, c_kal, c_fix):
        best, bid = 0.3, None
        for gid, gb in prev_gt.items():
            v = M._iou(b0, gb)
            if v >= best:
                best, bid = v, gid
        if bid is None or bid not in cur_gt:
            return None
        g = (cur_gt[bid][:2] + cur_gt[bid][2:]) / 2
        return int(np.linalg.norm(c_fix - g) < np.linalg.norm(c_kal - g))
    return lab


def run_video(vid, root, split, gate=None, collect=None):
    det = M.load_detection_cache(root / vid / "detections" / "raw_video.npz")
    matches = json.loads((root / vid / "alignment.json").read_text(encoding="utf-8"))["matches"]
    truth = json.loads((PROJECT_ROOT / "data/bdd100k/labels/box_track_20" / split / f"{vid}.json").read_text(encoding="utf-8"))
    field = M.load_motion_cache(root / vid / "motion/flyvis_tiled_lcn24.npz", 3.0)
    gt = [{l["id"]: np.array([l["box2d"]["x1"], l["box2d"]["y1"], l["box2d"]["x2"], l["box2d"]["y2"]], np.float32) for l in fr.get("labels") or [] if l.get("box2d")} for fr in truth]
    tr = Logged(M.make_tracker_args(track_high_thresh=0.25, track_low_thresh=0.1, track_buffer=5, match_thresh=0.8))
    tr.gate, tr.collect = gate, collect
    rows, prev = [], None
    for i, m in enumerate(matches):
        f = int(m["video_frame_index"])
        if prev is not None and f < len(field[1]):
            tr.motion = (lambda b, r0=prev, r1=f: M.propagate(b, field[0], field[1], r0, r1)); tr.label = make_label(gt[i - 1], gt[i])
        else:
            tr.motion = None; tr.label = None
        out = tr.update(M._DetSlice(*M.detections_for_frame(det, f)))
        rows.append(M.scalabel_frame(vid, truth[i], out)); prev = f
    return rows


def hota(rows, vid, split, name):
    d = OUT / "results" / f"r5_all_learned_{name}"
    M.save_json(d / f"{vid}.json", rows)
    return M.evaluate_hota(d, [vid], split=split)["HOTA"]


def main() -> int:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    split = json.loads((OUT / "trust_split.json").read_text(encoding="utf-8"))
    tc, vr = PROJECT_ROOT / "runs/train_cache", PROJECT_ROOT / "runs/video_rate"
    data = []
    for i, v in enumerate(split["tune"]):
        run_video(v, tc, "train", gate=None, collect=data)
        print(f"[COLLECT] {i + 1}/{len(split['tune'])} {v}: примеров {len(data)}", flush=True)
    X = np.array([x for x, _ in data]); y = np.array([t for _, t in data])
    models = {}
    for name, cols in (("with_fly", FLY_FEATS), ("no_fly", NOFLY_FEATS)):
        mdl = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight=None)).fit(X[:, cols], y)
        models[name] = (mdl, cols)
        print(f"[FIT] {name}: доля «поправка лучше» {y.mean():.3f}, точность на подборе {mdl.score(X[:, cols], y):.3f}, доля принятых {mdl.predict(X[:, cols]).mean():.3f}", flush=True)
    res = {"n_train": int(len(y)), "pos_rate": float(y.mean()), "per_video": {}}
    tests = [(v, tc, "train") for v in split["test"]] + [(v, vr, "val") for v in M.VAL7]
    base = {**json.loads((OUT / "dev_trust_test.json").read_text(encoding="utf-8"))["r5_all_none"], **json.loads((OUT / "val7_trust_test.json").read_text(encoding="utf-8"))["r5_all_none"]}
    for i, (v, root, sp) in enumerate(tests):
        r = {"none": base[v]["HOTA"]}
        for name, (mdl, cols) in models.items():
            gate = (lambda x, mdl=mdl, cols=cols: bool(mdl.predict(x[cols].reshape(1, -1))[0]))
            r[name] = hota(run_video(v, root, sp, gate=gate), v, sp, name)
        res["per_video"][v] = r
        print(f"[TEST] {i + 1}/{len(tests)} {v}: без движения {r['none']:.2f} | с мухой {r['with_fly']:.2f} | без признаков мухи {r['no_fly']:.2f}", flush=True)
    rng = np.random.default_rng(1)
    def boot(d):
        bs = d[rng.integers(0, len(d), (5000, len(d)))].mean(1)
        return {"mean": float(d.mean()), "ci95": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))], "wins": int((d > 0).sum()), "n": int(len(d))}
    P = res["per_video"]; vs = list(P)
    for a_, b_ in (("with_fly", "none"), ("no_fly", "none"), ("with_fly", "no_fly")):
        res[f"{a_}-{b_}"] = boot(np.array([P[v][a_] - P[v][b_] for v in vs]))
        r = res[f"{a_}-{b_}"]
        print(f"== {a_} − {b_}: {r['mean']:+.3f} [{r['ci95'][0]:+.3f}; {r['ci95'][1]:+.3f}], лучше {r['wins']}/{r['n']}", flush=True)
    (OUT / "trust_learned.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("[OK]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
