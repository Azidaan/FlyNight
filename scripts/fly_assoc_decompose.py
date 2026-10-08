"""Разложение потолка сопоставления (+6,7 HOTA, fly_assoc_target_v1) на разрывы треков и обмены номерами.

Выход стенда (signatures.npz → outputs) на размеченных кадрах; сопоставление рамки с эталоном — тот же класс, IoU ≥ 0,5.
  merge — склейка: каждому треку — номер эталонного объекта, с которым он совпадает чаще всего за жизнь (чинит разрывы:
          обрывки одного объекта получают один номер; обмены внутри трека остаются);
  split — разрезание: номер = (трек, эталонный объект под ним) — трек делится там, где под ним сменился объект (чинит обмены;
          разрывы остаются).
HOTA на test 35 и val7 против «как есть» и полного оракула (runs/fly_assoc_target).

Запуск: python scripts/fly_assoc_decompose.py
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

OUT = PROJECT_ROOT / "runs" / "fly_assoc_target"
CLASS_NAMES = ["pedestrian", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"]


def gid_int(g):
    return 2 * 10 ** 8 + (int(g) if str(g).isdigit() else abs(hash(g)) % 10 ** 7)


def video(task):
    vid, sig_root, root, split = task
    from scripts.oracle_detection_ladder import CATS, iou_mat
    z = np.load(Path(sig_root) / vid / "signatures.npz"); outs = z["outputs"]
    matches = json.loads((Path(root) / vid / "alignment.json").read_text(encoding="utf-8"))["matches"]
    truth = json.loads((PROJECT_ROOT / "data/bdd100k/labels/box_track_20" / split / f"{vid}.json").read_text(encoding="utf-8"))
    frames = []
    for m, fr in zip(matches, truth):
        f = int(m["video_frame_index"]); o = outs[outs[:, 0] == f]
        gb, gc, gid = [], [], []
        for l in fr.get("labels") or []:
            if l.get("box2d") and l.get("category") in CATS:
                bb = l["box2d"]; gb.append([bb["x1"], bb["y1"], bb["x2"], bb["y2"]]); gc.append(CATS[l["category"]]); gid.append(l["id"])
        gb = np.array(gb, np.float32).reshape(-1, 4); gc = np.array(gc)
        iou = iou_mat(o[:, 1:5], gb) if len(o) and len(gb) else np.zeros((len(o), 0))
        assign, used = [], set()
        for i, r in enumerate(o):
            g = None
            if iou.shape[1]:
                cand = np.where(gc == int(r[7]), iou[i], 0.0); j = int(cand.argmax())
                if cand[j] >= 0.5 and j not in used:
                    g = gid[j]; used.add(j)
            assign.append(g)
        frames.append((fr, o, assign))
    major = {}
    cnt = {}
    for fr, o, assign in frames:
        for r, g in zip(o, assign):
            if g is not None:
                cnt.setdefault(int(r[5]), Counter())[g] += 1
    for tid, c in cnt.items():
        major[tid] = c.most_common(1)[0][0]
    pair_id = {}
    for mode in ("merge", "split"):
        rows = []
        for fr, o, assign in frames:
            labels, seen = [], set()
            for r, g in zip(o, assign):
                tid = int(r[5])
                if mode == "merge":
                    nid = gid_int(major[tid]) if tid in major else tid
                    if nid in seen:   # два трека с одним основным объектом в одном кадре — второй остаётся со своим номером
                        nid = tid
                else:
                    nid = tid if g is None else pair_id.setdefault((tid, g), 10 ** 9 + len(pair_id))
                seen.add(nid)
                labels.append({"id": str(nid), "category": CLASS_NAMES[int(r[7])], "score": float(r[6]),
                               "box2d": {"x1": float(r[1]), "y1": float(r[2]), "x2": float(r[3]), "y2": float(r[4])}})
            rows.append({"videoName": vid, "name": fr["name"], "frameIndex": fr["frameIndex"], "labels": labels})
        d = OUT / "relabel" / mode / split; d.mkdir(parents=True, exist_ok=True)
        (d / f"{vid}.json").write_text(json.dumps(rows), encoding="utf-8")
    return vid


def main() -> int:
    from evaluation.hota_eval import evaluate_hota
    from scripts.fly_lifecycle_alive import sets
    S = sets(PROJECT_ROOT / "runs/fly_signatures", PROJECT_ROOT / "runs/fly_signatures_val")
    res = {}
    for s in ("test", "val7"):
        tasks = [(v, str(sr), str(rt), sp) for v, sr, rt, sp in S[s]]
        with ProcessPoolExecutor(max_workers=8) as pool:
            list(pool.map(video, tasks))
        split = S[s][0][3]; vids = [t[0] for t in tasks]
        prev = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))["ceiling"][s]   # «как есть» и полный оракул — из fly_assoc_target_v1
        res[s] = {"asis": prev["asis"], **{m: {k: evaluate_hota(OUT / "relabel" / m / split, vids, split=split).get(k) for k in ("HOTA", "AssA", "IDF1", "IDSw")} for m in ("merge", "split")}, "oracle": prev["oracle"]}
        print(f"[DECOMP] {s}: " + " | ".join(f"{m} HOTA {r['HOTA']:.3f} AssA {r['AssA']:.3f} IDSw {r['IDSw']:.0f}" for m, r in res[s].items()), flush=True)
    (OUT / "decompose.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("[OK]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
