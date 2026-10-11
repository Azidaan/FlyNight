"""Поиск схемы трекера, в которой внешнее движение даёт прирост HOTA (карточка motion_scheme_search_v1).

fly_input_norm_hota_v1: в схеме подмешивания старого ByteTrack (XYAH, 30 Гц) не помогает даже DIS. Здесь —
стендовый XYWHByteTrack (h0,25 / l0,1 / m0,8, fuse_score) и поправка прогноза Калмана внешним движением рамки:
сдвиг центра заменяется смесью «Калман / движение поля» с весом = уверенность оценки поля (как в
tracking/motion_augmented_bytetrack.py). Схемы:
  r30_all  — 30 Гц, все кадры видео, поправка всех треков;     r30_lost — то же, только потерянных;
  r5_all   — 5 Гц: трекер видит только размеченные кадры (буфер 5 = 1 с), движение рамки накапливается
             полем по сырым кадрам между ними (propagate_box), поправка всех треков;  r5_lost — только потерянных.
Источники: none (штатный Калман), dis (dis_quarter, усиление 1), fly (flyvis_tiled_lcn24, усиление 3 — только
там, где есть кэш). Данные: dev 35 видео train_cache (детекции, DIS) — для поиска схемы; val7 — проверка.
Детекции dev — от детектора, видевшего эти видео (оптимистичны), но одинаковы для всех вариантов.

Запуск: python scripts/motion_scheme_search.py --set dev [--schemes r5_all r5_lost r30_all r30_lost] [--sources none dis fly fly_oracle dis_oracle]
  *_oracle — диагностика (fly_motion_trust_v1): поправка берётся только там, где по разметке она ближе к правде, чем Калман (потолок, не для применения)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.hota_eval import evaluate_hota  # noqa: E402
from scripts.run_bdd_video_rate_experiment import detections_for_frame, load_detection_cache, load_motion_cache, save_json, scalabel_frame  # noqa: E402
from tracking.flyvis_tracker import MotionField  # noqa: E402
from tracking.offline_association import _DetSlice, make_tracker_args  # noqa: E402
from tracking.xywh_bytetrack import XYWHByteTrack  # noqa: E402

OUT = PROJECT_ROOT / "runs" / "motion_scheme_search"
VAL7 = ["b1c66a42-6f7d68ca", "b1c81faa-3df17267", "b1cac6a7-04e33135", "b1cebfb7-284f5117", "b1d22449-15fb948f", "b232c7c9-d251d9ee", "b251064f-e7a165fd"]
SOURCES = {"dis": ("dis_quarter", 1.0), "fly": ("flyvis_tiled_lcn24", 3.0)}


class MotionXYWH(XYWHByteTrack):
    """XYWHByteTrack, у которого сдвиг центра в прогнозе смешивается с движением рамки из внешнего поля."""

    def __init__(self, args, apply_to="all"):
        super().__init__(args)
        self.apply_to = apply_to
        self.motion = None          # функция box_xyxy → (вектор, уверенность) или None
        self.oracle = None          # диагностика (fly_motion_trust_v1): функция (рамка до, центр Калмана, центр с поправкой) → брать ли поправку
        self.w_scale, self.min_inside, self.points = 1.0, 0, None   # дешёвое доверие (fly_motion_trust_cheap_v1)
        self.audit = []

    def multi_predict(self, tracks):
        if not tracks:
            return
        before_c = [t.mean[:2].copy() for t in tracks]; before_b = [t.xyxy.copy() for t in tracks]
        super().multi_predict(tracks)
        if self.motion is None:
            return
        from ultralytics.trackers.basetrack import TrackState
        for t, c0, b0 in zip(tracks, before_c, before_b):
            if self.apply_to == "lost" and t.state != TrackState.Lost:
                continue
            if not (np.isfinite(b0).all() and b0[2] - b0[0] > 1 and b0[3] - b0[1] > 1):
                continue   # вырожденная рамка прогноза — поле не оценивает, остаётся Калман
            if self.min_inside and self.points is not None and int(np.all((self.points >= b0[:2]) & (self.points <= b0[2:]), axis=1).sum()) < self.min_inside:
                continue   # мало колонок поля внутри рамки — поправки нет, остаётся Калман
            vec, conf = self.motion(b0)
            w = float(np.clip(conf * self.w_scale, 0, 1))
            kal = t.mean[:2] - c0
            if self.oracle is not None:   # идеальный выбор по разметке: поправка целиком или чистый Калман
                w = 1.0 if self.oracle(b0, c0 + kal, c0 + vec) else 0.0
            t.mean[:2] = c0 + (1 - w) * kal + w * vec
            self.audit.append(w)


def propagate(box, points, vectors, r0, r1):
    from scripts.evaluate_flyvis_qmatch_gate import propagate_box
    pred, conf = propagate_box(box, points, vectors, r0, r1)
    c0 = (np.asarray(box)[:2] + np.asarray(box)[2:]) / 2; c1 = (pred[:2] + pred[2:]) / 2
    return c1 - c0, conf


def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1]); x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def make_oracle(prev_gt, cur_gt):
    """Трек сопоставляется с эталоном прошлого размеченного кадра (IoU ≥ 0,3); если объект есть и на текущем —
    выбирается прогноз, чей центр ближе к его центру; иначе — Калман. Использует разметку: только потолок."""
    def oracle(b0, c_kal, c_fix):
        best, bid = 0.3, None
        for gid, gb in prev_gt.items():
            v = _iou(b0, gb)
            if v >= best:
                best, bid = v, gid
        if bid is None or bid not in cur_gt:
            return False
        g = (cur_gt[bid][:2] + cur_gt[bid][2:]) / 2
        return float(np.linalg.norm(c_fix - g)) < float(np.linalg.norm(c_kal - g))
    return oracle


def ncc(a, b):
    a = a - a.mean(); b = b - b.mean(); d = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / d) if d > 1e-6 else -1.0


def crop(img, box):
    h, w = img.shape
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    if x1 < 0 or y1 < 0 or x2 > w or y2 > h or x2 - x1 < 4 or y2 - y1 < 4:
        return None
    return img[y1:y2, x1:x2]


def make_photo_verifier(f0, f1, scale=0.5):
    """Фотометрическая проверка (fly_motion_photometric_v1): шаблон объекта из кадра r0 сравнивается (нормированная
    корреляция) с кадром r1 в рамке, сдвинутой прогнозом Калмана и предложенной поправкой; принимается поправка, если её
    совпадение лучше. Разметка не используется — только изображение (постоянство яркости, как проверка шаблона в KLT)."""
    def verify(b0, c_kal, c_fix):
        b = np.asarray(b0, np.float64) * scale
        tpl = crop(f0, b)
        if tpl is None:
            return False
        half = (b[2:] - b[:2]) / 2
        sc = []
        for c in (np.asarray(c_kal) * scale, np.asarray(c_fix) * scale):
            pt = crop(f1, np.r_[c - half, c + half])
            if pt is None or pt.shape != tpl.shape:
                return False
            sc.append(ncc(tpl, pt))
        return sc[1] > sc[0]
    return verify


def labeled_frames(path, idx, scale=0.5):
    import cv2
    want = set(idx); out = {}
    cap = cv2.VideoCapture(str(path)); f = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if f in want:
            g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            out[f] = cv2.resize(g, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA).astype(np.float32)
        f += 1
        if f > max(want):
            break
    cap.release()
    return out


def run_one(vid, root, split, scheme, source):
    det = load_detection_cache(root / vid / "detections" / "raw_video.npz")
    matches = json.loads((root / vid / "alignment.json").read_text(encoding="utf-8"))["matches"]
    truth = json.loads((PROJECT_ROOT / "data/bdd100k/labels/box_track_20" / split / f"{vid}.json").read_text(encoding="utf-8"))
    anchor = {int(m["video_frame_index"]): i for i, m in enumerate(matches)}
    field = None
    photo_mode = source.endswith("#photo")
    source_core = source.removesuffix("#photo")
    oracle_mode = source_core.endswith("_oracle")
    base_source, _, knobs = source_core.removesuffix("_oracle").partition("@")   # «fly@w0.3m3» — вес × 0,3, запрет при < 3 колонок
    w_scale = float(knobs.split("w")[1].split("m")[0]) if knobs else 1.0
    min_inside = int(knobs.split("m")[1]) if knobs else 0
    if source != "none" and base_source != "zero":
        name, gain = SOURCES[base_source]
        path = root / vid / "motion" / f"{name}.npz"
        if not path.exists():
            return None
        field = load_motion_cache(path, gain)
    rate, apply_to = scheme.split("_")
    rows = []
    if rate == "r30":
        tr = MotionXYWH(make_tracker_args(track_high_thresh=0.25, track_low_thresh=0.1, track_buffer=30, match_thresh=0.8), apply_to)
        for f in range(int(det["frame_count"])):
            if field is not None and f < len(field[1]):
                mf = MotionField(field[0], field[1][f])
                def motion(b, mf=mf):
                    e = mf.estimate_box(b, context_scale=2.0, min_samples=3); return e.vector_xy, e.confidence
                tr.motion = motion
            out = tr.update(_DetSlice(*detections_for_frame(det, f)))
            if f in anchor:
                rows.append(scalabel_frame(vid, truth[anchor[f]], out))
    else:
        tr = MotionXYWH(make_tracker_args(track_high_thresh=0.25, track_low_thresh=0.1, track_buffer=5, match_thresh=0.8), apply_to)
        tr.w_scale, tr.min_inside, tr.points = w_scale, min_inside, (field[0] if field is not None and base_source != "zero" else None)
        prev = None
        gt = [{l["id"]: np.array([l["box2d"]["x1"], l["box2d"]["y1"], l["box2d"]["x2"], l["box2d"]["y2"]], np.float32)
               for l in fr.get("labels") or [] if l.get("box2d")} for fr in truth] if oracle_mode else None
        frames = None
        if photo_mode:
            vdir = PROJECT_ROOT / "data/bdd100k/videos" / ("val" if split == "val" else "train")
            frames = labeled_frames(vdir / f"{vid}.mov", [int(m["video_frame_index"]) for m in matches])
        if base_source == "zero":   # контроль: предложение «нулевое движение» вместо поля
            field = (np.zeros((1, 2), np.float32), None)
        for i, m in enumerate(matches):
            f = int(m["video_frame_index"])
            if base_source == "zero" and prev is not None:
                tr.motion = (lambda b: (np.zeros(2, np.float32), 1.0))
                tr.oracle = make_photo_verifier(frames[prev], frames[f]) if photo_mode and prev in frames and f in frames else None
            elif field is not None and prev is not None and f < len(field[1]):
                tr.motion = (lambda b, r0=prev, r1=f: propagate(b, field[0], field[1], r0, r1))
                if photo_mode:
                    tr.oracle = make_photo_verifier(frames[prev], frames[f]) if prev in frames and f in frames else None
                else:
                    tr.oracle = make_oracle(gt[i - 1], gt[i]) if oracle_mode else None
            else:
                tr.motion = None; tr.oracle = None
            out = tr.update(_DetSlice(*detections_for_frame(det, f)))
            rows.append(scalabel_frame(vid, truth[i], out))
            prev = f
    d = OUT / "results" / f"{scheme}_{source.replace('@', '_').replace('#', '_')}"
    save_json(d / f"{vid}.json", rows)
    met = evaluate_hota(d, [vid], split=split)
    return {"HOTA": met["HOTA"], "AssA": met["AssA"], "DetA": met["DetA"], "IDF1": met["IDF1"], "mean_weight": float(np.mean(tr.audit)) if tr.audit else 0.0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=("dev", "test", "val7"), default="dev")
    ap.add_argument("--schemes", nargs="+", default=["r5_all", "r5_lost", "r30_all", "r30_lost"])
    ap.add_argument("--sources", nargs="+", default=["none", "dis"])
    ap.add_argument("--videos", nargs="*", default=None, help="подмножество видео набора (подбор / проверка)")
    ap.add_argument("--tag", default="", help="суффикс файла результатов")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.set in ("dev", "test"):   # test 35 — проверка действия на видео, не участвовавших в выборе схемы (dis_r30_test_v1)
        vids = json.loads((PROJECT_ROOT / "runs/train_cache/selection.json").read_text(encoding="utf-8"))["videos"][a.set]
        root, split = PROJECT_ROOT / "runs/train_cache", "train"
    else:
        vids, root, split = VAL7, PROJECT_ROOT / "runs/video_rate", "val"
    if a.videos:
        vids = [v for v in vids if v in set(a.videos)]
    out_path = OUT / f"{a.set}{a.tag}.json"
    res = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}
    t0 = time.time(); todo = [(s, src, v) for s in a.schemes for src in a.sources for v in vids]
    for i, (s, src, v) in enumerate(todo):
        key = f"{s}_{src}"
        if v in res.get(key, {}):
            continue
        r = run_one(v, root, split, s, src)
        if r is None:
            continue
        res.setdefault(key, {})[v] = r
        out_path.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        el = time.time() - t0
        print(f"[SCHEME] {i + 1}/{len(todo)} {key} {v}: HOTA {r['HOTA']:.2f} вес {r['mean_weight']:.2f} | осталось ~{el / (i + 1) * (len(todo) - i - 1) / 60:.0f} мин", flush=True)
    rng = np.random.default_rng(1); summ = {}
    for s in a.schemes:
        base = res.get(f"{s}_none", {})
        for src in a.sources:
            k = f"{s}_{src}"
            if k not in res:
                continue
            vs = [v for v in res[k] if v in base]
            entry = {"videos": len(vs), "HOTA": float(np.mean([res[k][v]["HOTA"] for v in vs]))}
            if src != "none" and vs:
                d = np.array([res[k][v]["HOTA"] - base[v]["HOTA"] for v in vs]); bs = d[rng.integers(0, len(d), (5000, len(d)))].mean(1)
                entry.update({"minus_none": float(d.mean()), "ci95": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))], "wins": int((d > 0).sum())})
            summ[k] = entry
            print(f"== {k}: HOTA {entry['HOTA']:.2f}" + (f" | − без движения {entry['minus_none']:+.3f} [{entry['ci95'][0]:+.3f}; {entry['ci95'][1]:+.3f}], лучше {entry['wins']}/{len(vs)}" if "minus_none" in entry else ""), flush=True)
    res["_summary"] = summ
    out_path.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[OK] {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
