"""З13 (карточка fly_object_readout_z13_v1): извлечение обучающих эпизодов потери трека (процессор, без GPU).

Трекер стенда (XYWHByteTrack, h0,25 / l0,1 / буфер 30 / m0,8) без вмешательства на сохранённых детекциях; вход глаза — как у
стенда (сетчатка lamina, уменьшение 4; tracking/foveated_eyes.EyeBank._render — вырезка crop_box, 721 колонка).
Эпизод — одна потеря трека: 4 последних ведомых кадра (вырезки вокруг рамок трека; если ведомых кадров меньше —
повтор первого) + кадры потери (вырезки вокруг прогноза Калмана перед первым сопоставлением) до последнего размеченного
кадра потери. У половины эпизодов (сид по видео, треку и кадру потери) центр области поиска сдвинут на постоянный вектор
длиной до 0,25 стороны области плюс дрожание до 0,05 стороны на кадр.
Метки — только на размеченных кадрах и только по исходной разметке (без интерполяции):
  объект образца — объект детекции трека на последнем размеченном кадре, где трек был ведомым (венгерское внутри класса,
  IoU ≥ 0,5); id_dist — кадров от этого кадра до последнего ведомого;
  kind 1 — объект образца установлен (id_dist ≤ 12) и его центр в разметке кадра внутри поля колонок глаза (шестиугольник
  внутри квадратной вырезки: ближайшая колонка не дальше 0,6 шага; цель — эта колонка); центр в углу вырезки вне поля — kind 0;
  kind 0 — установлен, но в разметке кадра его нет или центр вне области; kind −1 — не установлен, либо его нет в разметке
  кадра, но он размечен на обоих соседних размеченных кадрах (похоже на дыру в разметке) — в обучение отказу не входит.
Эпизоды только с kind −1 сохраняются с вероятностью 0,25 (для справки о вмешательствах на неизвестных).
Выход — runs/z13_data/<id>.npz и runs/z13_data/summary.json.

Запуск: python scripts/z13_data.py [--set train|dev] [--workers 4] [--video-id ...]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import zlib
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

OUT = PROJECT_ROOT / "runs/z13_data"
FRAME = (1280, 720)
ID_MAX_DIST = 12
PERTURB_MAX, JITTER = 0.25, 0.05
KEEP_UNKNOWN = 0.25
FIELD_TOL = 0.6   # центр внутри поля колонок: ближайшая колонка не дальше 0,6 шага (внутри шестиугольника ≤ шаг/√3 ≈ 0,577)


def raw_index(truth: list, vf: list[int], valid) -> dict[int, list]:
    """Кадр видео → [(объект, класс, рамка)] только на размеченных кадрах, без интерполяции."""
    out = {}
    for k, fr in enumerate(truth):
        objs = []
        for l in fr.get("labels") or []:
            if l["category"] in valid and not (l.get("attributes") or {}).get("crowd", False):
                b = l["box2d"]; objs.append((str(l["id"]), l["category"], np.array([b["x1"], b["y1"], b["x2"], b["y2"]], np.float64)))
        out[int(vf[k])] = objs
    return out


def episode_rng(vid: str, tid: int, start: int) -> np.random.Generator:
    return np.random.default_rng(zlib.crc32(f"{vid}|{tid}|{start}".encode()))


def label_frame(identity, id_dist, objs, crop, geom, eye_size, around=False):
    """→ (kind, колонка, центр, рамка) по исходной разметке кадра; around — объект размечен на обоих соседних размеченных кадрах."""
    from tracking.z13_readout import to_eye
    if identity is None or id_dist < 0 or id_dist > ID_MAX_DIST:
        return -1, -1, None, None
    hit = [b for g, _, b in objs if g == identity]
    if not hit:   # нет в разметке кадра; если есть на обоих соседних размеченных — похоже на дыру в разметке, не отказ
        return (-1 if around else 0), -1, None, None
    b = hit[0]; c = (b[:2] + b[2:]) / 2
    x1, y1, x2, y2 = crop
    if not (x1 <= c[0] <= x2 and y1 <= c[1] <= y2):
        return 0, -1, c, b
    ce = np.asarray(to_eye(c, crop, eye_size)); col = geom.nearest(ce)
    if np.linalg.norm(geom.points[col] - ce) > FIELD_TOL * geom.spacing:   # угол квадрата вне шестиугольника колонок — глаз объект не видит
        return 0, -1, c, b
    return 1, col, c, b


def extract(task):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""   # извлечение — только процессор (несколько CUDA-процессов зависают)
    import torch
    torch.cuda.is_available = lambda: False   # на Windows при скрытом GPU возвращает True (device_count 0) — FlyVis выбрал бы CUDA
    import cv2
    from ultralytics.trackers.basetrack import BaseTrack, TrackState
    import scripts.fovea_target_30hz as FT
    from scripts.freeze_bench import XYWH_CONFIG
    from scripts.output_filter_oracle import VALID, _iou
    from scripts.run_bdd_video_rate_experiment import detections_for_frame, load_detection_cache, make_motion_estimator
    from tracking.foveated_eyes import EyeBank, box_to_crop_frame, crop_box
    from tracking.offline_association import _DetSlice, make_tracker_args
    from tracking.retina import make_retina
    from tracking.z13_readout import N_EXEMPLAR, HexGeometry
    vid, root, split, out_dir, videos_dir = task
    out_path = Path(out_dir) / f"{vid}.npz"
    if out_path.exists():
        return vid, json.loads(out_path.with_suffix(".json").read_text(encoding="utf-8"))
    det = load_detection_cache(PROJECT_ROOT / "runs/train_cache" / vid / "detections" / "raw_video.npz")
    vf = [int(m["video_frame_index"]) for m in json.loads((Path(root) / vid / "alignment.json").read_text(encoding="utf-8"))["matches"]]
    truth = json.loads((PROJECT_ROOT / "data/bdd100k/labels/box_track_20" / split / f"{vid}.json").read_text(encoding="utf-8"))
    R = raw_index(truth, vf, VALID)
    fps = float(json.loads((PROJECT_ROOT / "runs/train_cache" / vid / "motion" / "flyvis_tiled.json").read_text(encoding="utf-8"))["fps"])
    est = make_motion_estimator("flyvis_tiled", fps, 0.02)
    bank = EyeBank(est); eye_size = bank.eye_size
    geom = HexGeometry(bank.points_local, eye_size)
    retina = make_retina("lamina", fps, blur_downscale=4)
    ctx = {}
    Base = FT.oracle_tracker(None, ctx, FT.new_stats())
    hist: dict[int, deque] = {}          # трек → последние ведомые кадры: (кадр, рамка xyxy, hex)
    ident: dict[int, tuple] = {}         # трек → (объект или None, кадр) на последнем размеченном ведомом кадре
    open_ep: dict[int, dict] = {}
    done: list[dict] = []
    cur = {"f": -1, "eye": None}

    def render(crops):
        with bank.torch.inference_mode():
            return bank._render(cur["eye"], crops)[:, 0].cpu().numpy().astype(np.float16)

    class Hook(Base):
        def _pre_first_associate(self, pool, unconfirmed, img, results_high):
            super()._pre_first_associate(pool, unconfirmed, img, results_high)
            f = cur["f"]; lost = [t for t in pool if t.state != TrackState.Tracked]
            seen = set()
            crops, owners = [], []
            for t in lost:
                tid = int(t.track_id); seen.add(tid)
                if tid not in open_ep:
                    h = list(hist.get(tid, []))
                    if not h:
                        continue
                    h = [h[0]] * (N_EXEMPLAR - len(h)) + h[-N_EXEMPLAR:]
                    idt = ident.get(tid, (None, -1))
                    rng = episode_rng(vid, tid, f)
                    pert = bool(rng.random() < 0.5); ang = rng.uniform(0, 2 * np.pi); rad = rng.uniform(0, PERTURB_MAX)
                    open_ep[tid] = {"tid": tid, "start": f, "ex": h, "identity": idt[0], "id_dist": (h[-1][0] - idt[1]) if idt[1] >= 0 else -1,
                                    "pert": pert, "off": (np.cos(ang) * rad, np.sin(ang) * rad), "rng": rng, "search": []}
                ep = open_ep[tid]
                cx, cy, w, hh = (float(v) for v in t.mean[:4])
                side = max(2.0 * max(w, hh), 64.0)
                dx = dy = 0.0
                if ep["pert"]:
                    j = ep["rng"].uniform(-JITTER, JITTER, 2)
                    dx, dy = (ep["off"][0] + j[0]) * side, (ep["off"][1] + j[1]) * side
                crop = crop_box((cx + dx, cy + dy, w, hh), FRAME)
                crops.append(crop); owners.append((tid, (cx, cy)))
            if crops:
                hx = render(crops)
                for (tid, pc), crop, hv in zip(owners, crops, hx):
                    open_ep[tid]["search"].append((f, crop, hv, pc))
            for tid in [k for k in open_ep if k not in seen]:
                done.append(open_ep.pop(tid))

    BaseTrack.reset_id()
    tr = Hook(make_tracker_args(**XYWH_CONFIG))
    cap = cv2.VideoCapture(str(Path(videos_dir) / f"{vid}.mov"))
    n = int(np.load(PROJECT_ROOT / "runs/fly_signatures" / vid / "signatures.npz")["frame_count"])
    for f in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        cur["f"] = f; cur["eye"] = retina.process(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        xyxy, conf, cls = detections_for_frame(det, f)
        lab = FT.oracle_pre(tr, ctx, R, f, xyxy, conf, cls)
        tr.update(_DetSlice(xyxy, conf, cls))
        boxes = np.asarray(xyxy)
        upd = [t for t in tr.tracked_stracks if t.frame_id == tr.frame_id]
        if upd:
            crops = [crop_box(tuple(float(v) for v in t.mean[:4]), FRAME) for t in upd]
            hx = render(crops)
            for t, crop, hv in zip(upd, crops, hx):
                tid = int(t.track_id)
                hist.setdefault(tid, deque(maxlen=N_EXEMPLAR)).append((f, np.asarray(t.xyxy, np.float64), crop, hv))
                if f in R and len(boxes):
                    ident[tid] = (lab[int(np.argmax(_iou([t.xyxy], boxes)[0]))], f)
    cap.release()
    done += list(open_ep.values())
    # сборка массивов
    E = {k: [] for k in ("tid", "start", "identity", "id_dist", "pert", "offset")}
    F = {k: [] for k in ("hex", "frame", "crop")}
    L = {k: [] for k in ("ep", "pos", "kind", "col", "center", "gtbox", "pred", "crop")}
    masks = []
    counts = {"episodes_total": len(done), "episodes_kept": 0, "pos": 0, "neg": 0, "unknown": 0, "frames": 0}
    ann = sorted(R); nb_of = {f: (ann[i - 1] if i else None, ann[i + 1] if i + 1 < len(ann) else None) for i, f in enumerate(ann)}

    def around(g, f):
        a, b = nb_of[f]
        return a is not None and b is not None and any(o[0] == g for o in R[a]) and any(o[0] == g for o in R[b])

    for ep in done:
        rows = []
        for i, (f, crop, hv, pc) in enumerate(ep["search"]):
            if f in R:
                kind, col, c, b = label_frame(ep["identity"], ep["id_dist"], R[f], crop, geom, eye_size, around(ep["identity"], f))
                rows.append((i, kind, col, c, b, pc, crop))
        if not rows:
            continue
        if all(r[1] == -1 for r in rows) and ep["rng"].random() >= KEEP_UNKNOWN:
            continue
        last = max(r[0] for r in rows)
        e = len(E["tid"]); E["tid"].append(ep["tid"]); E["start"].append(ep["start"]); E["identity"].append(ep["identity"] or "")
        E["id_dist"].append(ep["id_dist"]); E["pert"].append(ep["pert"]); E["offset"].append(len(F["frame"]))
        fx, bx, cx_, hv = ep["ex"][-1]
        masks.append(geom.mask_in_box(box_to_crop_frame(bx, cx_, eye_size)))
        for f, box, crop, h in ep["ex"]:
            F["hex"].append(h); F["frame"].append(f); F["crop"].append(crop)
        for f, crop, h, pc in ep["search"][:last + 1]:
            F["hex"].append(h); F["frame"].append(f); F["crop"].append(crop)
        for i, kind, col, c, b, pc, crop in rows:
            L["ep"].append(e); L["pos"].append(i); L["kind"].append(kind); L["col"].append(col)
            L["center"].append(c if c is not None else (np.nan, np.nan)); L["gtbox"].append(b if b is not None else (np.nan,) * 4); L["pred"].append(pc); L["crop"].append(crop)
            counts[{1: "pos", 0: "neg", -1: "unknown"}[kind]] += 1
        counts["episodes_kept"] += 1
    E["offset"].append(len(F["frame"])); counts["frames"] = len(F["frame"])
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, ep_tid=np.asarray(E["tid"], np.int32), ep_start=np.asarray(E["start"], np.int32), ep_identity=np.asarray(E["identity"]),
                        ep_id_dist=np.asarray(E["id_dist"], np.int32), ep_pert=np.asarray(E["pert"], bool), ep_offset=np.asarray(E["offset"], np.int64),
                        ex_mask=np.asarray(masks, bool).reshape(-1, len(geom.points)), fr_hex=np.asarray(F["hex"], np.float16).reshape(-1, len(geom.points)),
                        fr_frame=np.asarray(F["frame"], np.int32), fr_crop=np.asarray(F["crop"], np.int32).reshape(-1, 4),
                        lab_ep=np.asarray(L["ep"], np.int32), lab_pos=np.asarray(L["pos"], np.int32), lab_kind=np.asarray(L["kind"], np.int8), lab_col=np.asarray(L["col"], np.int32),
                        lab_center=np.asarray(L["center"], np.float32).reshape(-1, 2), lab_gtbox=np.asarray(L["gtbox"], np.float32).reshape(-1, 4),
                        lab_pred=np.asarray(L["pred"], np.float32).reshape(-1, 2), lab_crop=np.asarray(L["crop"], np.int32).reshape(-1, 4))
    out_path.with_suffix(".json").write_text(json.dumps(counts), encoding="utf-8")
    return vid, counts


def relabel_field(out_dir: Path) -> dict:
    """Перемаркировка сохранённых эпизодов правилом поля колонок (Т1 13.10: центр в углу вырезки вне шестиугольника); kind 1 → 0."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    torch.cuda.is_available = lambda: False
    from scripts.run_bdd_video_rate_experiment import make_motion_estimator
    from tracking.foveated_eyes import EyeBank
    from tracking.z13_readout import HexGeometry, to_eye
    bank = EyeBank(make_motion_estimator("flyvis_tiled", 30.0, 0.02)); geom = HexGeometry(bank.points_local, bank.eye_size)
    tot = {"files": 0, "relabelled": 0}
    for fp in sorted(Path(out_dir).glob("*.npz")):
        d = dict(np.load(fp)); k = d["lab_kind"].copy(); col = d["lab_col"].copy(); n = 0
        for i in np.flatnonzero(k == 1):
            ce = np.asarray(to_eye(d["lab_center"][i], tuple(int(v) for v in d["lab_crop"][i]), bank.eye_size))
            if np.linalg.norm(geom.points[col[i]] - ce) > FIELD_TOL * geom.spacing:
                k[i] = 0; col[i] = -1; n += 1
        if n:
            d["lab_kind"], d["lab_col"] = k, col
            np.savez_compressed(fp, **d)
            c = json.loads(fp.with_suffix(".json").read_text(encoding="utf-8")); c["pos"] -= n; c["neg"] += n; c["relabelled_field"] = n
            fp.with_suffix(".json").write_text(json.dumps(c), encoding="utf-8")
        tot["files"] += 1; tot["relabelled"] += n
    return tot


def main() -> int:
    from scripts.fly_lifecycle_alive import sets
    ap = argparse.ArgumentParser(); ap.add_argument("--set", default="train"); ap.add_argument("--workers", type=int, default=4); ap.add_argument("--video-id", action="append", default=[])
    ap.add_argument("--out", type=Path, default=OUT); ap.add_argument("--relabel-field", action="store_true")
    a = ap.parse_args()
    if a.relabel_field:
        print(f"[RELABEL] {relabel_field(a.out)}", flush=True)
        print("[OK]", flush=True)
        return 0
    S = sets(PROJECT_ROOT / "runs/fly_signatures", PROJECT_ROOT / "runs/fly_signatures_val")
    items = [(v, str(rt), sp) for v, sr, rt, sp in S[a.set] if not a.video_id or v in a.video_id]
    tasks = [(v, rt, sp, str(a.out), str(PROJECT_ROOT / "data/bdd100k/videos/train")) for v, rt, sp in items]
    tot = {}
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for k, (vid, c) in enumerate(pool.map(extract, tasks)):
            for key, v in c.items():
                tot[key] = tot.get(key, 0) + v
            print(f"[{k + 1}/{len(tasks)}] {vid}: {c}", flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    summ = a.out / "summary.json"
    prev = json.loads(summ.read_text(encoding="utf-8")) if summ.exists() else {}
    prev[a.set] = dict(tot, videos=len(tasks)); summ.write_text(json.dumps(prev, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[TOTAL] {a.set}: {tot}\n[OK]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
