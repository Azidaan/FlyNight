"""Сшивка с вероятностью «трек настоящий» (карточка stitch_realness_fly_v1) — замена узла «сдвиг помеченные пары → все кандидаты».

Голова пары (база + MASA, как в stitch_eval_v2) обучена только на парах с эталоном у обоих концов и не знает, что конец A или
начало B может быть ложным треком. Здесь — отдельная голова «конец жизни сопоставлен с эталоном» (для A) и «начало жизни
сопоставлено с эталоном» (для B), обучение на жизнях train 130 с метками по эталону, интерполированному между опорными кадрами (interp_labels;
stitch_unlabeled_diag_v1 показал, что метки «есть опорный кадр» штрафуют полезные обрывки), и вес пары
p' = p · r_end(A) · r_start(B). Две версии головы, отличие ровно одно — признаки мухи:
  nofly — длина жизни, число строк, уверенность детектора (средняя, максимальная, у конца/начала — 5 строк), площадь,
          пропорции и высота центра рамки, n_first/n_last, класс, вход глаза (hexin 721);
  fly   — то же + сигнатура мухи у конца/начала (34 типа × 19).
Сшивка жадная (eval_fly_reid.stitch_video), порог — по HOTA на dev 35 из сетки, проверка test 35 и val7, парный бутстрэп.

Запуск: python scripts/stitch_realness.py
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts import stitch_eval_v2 as V  # noqa: E402

OUT = PROJECT_ROOT / "runs/stitch_realness"
PAIR_HEADS = ["base", "base_masa"]
GRID = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]


def life_features(sig_path: str) -> dict:
    z = np.load(Path(sig_path)); o = z["outputs"]; n = len(z["track_id"])
    rows = {}
    for r in o:
        rows.setdefault(int(r[5]), []).append(r)
    common = np.zeros((n, 8), np.float32); ends = {"end": np.zeros((n, 7), np.float32), "start": np.zeros((n, 7), np.float32)}
    cls = np.zeros((n, 8), np.float32)
    for i, t in enumerate(z["track_id"]):
        R = np.asarray(rows.get(int(t), []), np.float32).reshape(-1, 8); sc = R[:, 6] if len(R) else np.zeros(1, np.float32)
        common[i] = [np.log1p(int(z["end"][i]) - int(z["start"][i])), np.log1p(len(R)), sc.mean(), sc.max(), sc.min(), sc.std(), 0, 0]
        cls[i, min(int(z["cls"][i]), 7)] = 1
        for name, box, nn, part in (("end", z["last_box"][i], z["n_last"][i], sc[-5:]), ("start", z["first_box"][i], z["n_first"][i], sc[:5])):
            b = box.astype(np.float64); w, h = max(b[2] - b[0], 1.0), max(b[3] - b[1], 1.0)
            ends[name][i] = [np.log(w * h), np.log(w / h), (b[1] + b[3]) / 2 / 720.0, (b[0] + b[2]) / 2 / 1280.0, float(nn) / 5, part.mean(), part.max()]
    return {"common": common, "cls": cls, "end": ends["end"], "start": ends["start"],
            "hex_end": z["last_hexin"].astype(np.float32), "hex_start": z["first_hexin"].astype(np.float32),
            "fly_end": z["last_fly"].reshape(n, -1).astype(np.float32), "fly_start": z["first_fly"].reshape(n, -1).astype(np.float32)}


def interp_labels(task) -> np.ndarray:
    """Метки «настоящий» для конца и начала каждой жизни по эталону, интерполированному между опорными кадрами
    (stitch_unlabeled_diag_v1: 53 % жизней — обрывки без опорного кадра). Кадр f: если опорный — эталон как есть; иначе
    объекты с тем же id в ближайших опорных кадрах слева и справа (разрыв ≤ MAX_SPAN), рамки линейно. Совпадение — IoU ≥ 0,5.
    Возвращает (n, 2) int8: 1 — совпало, 0 — эталон есть, но не совпало, −1 — эталона нет (вне опорных кадров)."""
    from scripts.evaluate_flyvis_qmatch_gate import truth_arrays
    vid, sig_dir, root, split = task

    def iou_max(b, G):   # своя IoU: tracking.flyvis_tracker.box_iou отвергает вырожденные рамки (бывают у концов жизней)
        x1 = np.maximum(b[0], G[:, 0]); y1 = np.maximum(b[1], G[:, 1]); x2 = np.minimum(b[2], G[:, 2]); y2 = np.minimum(b[3], G[:, 3])
        inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
        ua = max((b[2] - b[0]) * (b[3] - b[1]), 0) + np.clip(G[:, 2] - G[:, 0], 0, None) * np.clip(G[:, 3] - G[:, 1], 0, None) - inter
        return float((inter / np.maximum(ua, 1e-9)).max())
    z = np.load(Path(sig_dir) / vid / "signatures.npz")
    matches = json.loads((Path(root) / vid / "alignment.json").read_text(encoding="utf-8"))["matches"]
    truth = json.loads((PROJECT_ROOT / "data/bdd100k/labels/box_track_20" / split / f"{vid}.json").read_text(encoding="utf-8"))
    rf = np.array([int(m["video_frame_index"]) for m in matches]); order = np.argsort(rf); rf = rf[order]
    T = [truth_arrays(truth[int(i)]) for i in order]
    cache = {}

    def gt_at(f):
        if f in cache:
            return cache[f]
        k = int(np.searchsorted(rf, f))
        if k < len(rf) and rf[k] == f:
            out = (T[k][0], T[k][2])
        elif 0 < k < len(rf) and rf[k] - rf[k - 1] <= MAX_SPAN:
            (b0, _, i0, _), (b1, _, i1, _) = T[k - 1], T[k]; w = (f - rf[k - 1]) / (rf[k] - rf[k - 1])
            pos1 = {t: j for j, t in enumerate(i1)}
            common = [(j, pos1[t], t) for j, t in enumerate(i0) if t in pos1]
            out = (np.asarray([b0[a] * (1 - w) + b1[b] * w for a, b, _ in common], np.float32).reshape(-1, 4), [t for *_, t in common])
        else:
            out = None
        cache[f] = out
        return out
    n = len(z["track_id"]); lab = np.full((n, 2), -1, np.int8)
    for i in range(n):
        for col, f, box in ((0, int(z["end"][i]), z["last_box"][i]), (1, int(z["start"][i]), z["first_box"][i])):
            g = gt_at(f)
            if g is None:
                continue
            lab[i, col] = int(len(g[1]) > 0 and np.isfinite(box).all() and iou_max(box.astype(np.float64), g[0].astype(np.float64)) >= 0.5)
    return lab


MAX_SPAN = 12   # опорные кадры — каждый 6-й; допускается один пропущенный


def X_of(f: dict, side: str, fly: bool) -> np.ndarray:
    parts = [f["common"], f["cls"], f[side], f[f"hex_{side}"]] + ([f[f"fly_{side}"]] if fly else [])
    return np.concatenate(parts, 1)


def main() -> int:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from evaluation.hota_eval import evaluate_hota
    from scripts.eval_fly_reid import stitch_video
    from scripts.fly_lifecycle_alive import sets
    OUT.mkdir(parents=True, exist_ok=True)
    S = sets(PROJECT_ROOT / "runs/fly_signatures", PROJECT_ROOT / "runs/fly_signatures_val")
    data, feats = {}, {}
    with ProcessPoolExecutor(max_workers=6, initializer=V._set, initargs=(150, 5.0, True)) as pool:
        for s in ("train", "dev", "test", "val7"):
            data[s] = list(pool.map(V.cached_pairs, [(v, str(sr), str(rt), sp) for v, sr, rt, sp in S[s]]))
            feats[s] = list(pool.map(life_features, [str(Path(sr) / v / "signatures.npz") for v, sr, rt, sp in S[s]]))
            for f, lab in zip(feats[s], pool.map(interp_labels, [(v, str(sr), str(rt), sp) for v, sr, rt, sp in S[s]])):
                f["real"] = lab
            L = np.concatenate([f["real"] for f in feats[s]])
            print(f"[LABEL] {s}: концов с меткой {(L[:, 0] >= 0).mean():.3f} (настоящих {(L[:, 0] == 1).mean():.3f}), начал с меткой {(L[:, 1] >= 0).mean():.3f}; прежняя разметка (опорный кадр) — {np.concatenate([d['life_known'] for d in data[s]])[:, 0].mean():.3f}", flush=True)
            print(f"[DATA] {s}: видео {len(data[s])}, кандидатов {sum(len(d['y']) for d in data[s])}, жизней {sum(len(d['life_known']) for d in data[s])}", flush=True)
    meta = {s: (S[s][0][1], S[s][0][2], S[s][0][3]) for s in data}
    rep = {"realness_AP": {}, "stitch": {}, "diff": {}}
    rng = np.random.default_rng(1)
    # головы «настоящий»: конец (life_known[:, 0]) и начало (life_known[:, 1])
    real = {}
    for side, col in (("end", 0), ("start", 1)):
        ytr_all = np.concatenate([f["real"][:, col] for f in feats["train"]]); ok = ytr_all >= 0; ytr = ytr_all[ok].astype(int)
        for fly in (False, True):
            name = f"{side}_{'fly' if fly else 'nofly'}"
            Xtr = np.concatenate([X_of(f, side, fly) for f in feats["train"]])[ok]
            m = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000)).fit(Xtr, ytr)
            real[name] = {s: [m.predict_proba(X_of(f, side, fly))[:, 1] for f in feats[s]] for s in ("dev", "test", "val7")}
            rep["realness_AP"][name] = {}
            for s in ("dev", "test", "val7"):
                yy = np.concatenate([f["real"][:, col] for f in feats[s]]); pp = np.concatenate(real[name][s])
                rep["realness_AP"][name][s] = float(average_precision_score(yy[yy >= 0], pp[yy >= 0]))
            print(f"[REAL] {name:12s} доля настоящих train {ytr.mean():.3f} (жизней с меткой {ok.sum()}) | AP dev {rep['realness_AP'][name]['dev']:.4f} test {rep['realness_AP'][name]['test']:.4f} val7 {rep['realness_AP'][name]['val7']:.4f}", flush=True)
        y_v = [f["real"][:, col] for f in feats["test"]]
        a_v, b_v = real[f"{side}_fly"]["test"], real[f"{side}_nofly"]["test"]
        bs = []
        for _ in range(1000):
            k = rng.integers(0, len(y_v), len(y_v)); yy = np.concatenate([y_v[i] for i in k]); mk = yy >= 0
            bs.append(average_precision_score(yy[mk], np.concatenate([a_v[i] for i in k])[mk]) - average_precision_score(yy[mk], np.concatenate([b_v[i] for i in k])[mk]))
        rep["realness_AP"][f"{side}_fly-nofly|test"] = {"mean": rep["realness_AP"][f"{side}_fly"]["test"] - rep["realness_AP"][f"{side}_nofly"]["test"], "ci95": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]}
        r = rep["realness_AP"][f"{side}_fly-nofly|test"]
        print(f"[REAL] {side}: муха − без мухи test AP {r['mean']:+.4f} [{r['ci95'][0]:+.4f}; {r['ci95'][1]:+.4f}]", flush=True)
    Xtr = np.concatenate([d["X"][d["y"] >= 0] for d in data["train"]]); ytr = np.concatenate([d["y"][d["y"] >= 0] for d in data["train"]])
    pair_p = {}
    for h in PAIR_HEADS:
        idx = [V.NAMES.index(c) for c in V.HEADS[h]]
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)).fit(Xtr[:, idx], ytr)
        pair_p[h] = {s: [m.predict_proba(d["X"][:, idx])[:, 1] for d in data[s]] for s in ("dev", "test", "val7")}

    def weights(h, var, s):
        out = []
        for i, d in enumerate(data[s]):
            p = pair_p[h][s][i]
            if var != "plain":
                p = p * real[f"end_{var}"][s][i][d["pairs"][:, 0]] * real[f"start_{var}"][s][i][d["pairs"][:, 1]]
            out.append(p)
        return out

    def run(key, h, var, s, thr):
        d_out = OUT / "stitch" / f"{key}_{thr:g}" / s; w = weights(h, var, s)
        tasks = [(d["video_id"], *meta[s], d["pairs"], w[i], thr, str(d_out)) for i, d in enumerate(data[s])]
        with ProcessPoolExecutor(max_workers=6) as pool:
            per = {r["video_id"]: r["metrics"] for r in pool.map(stitch_video, tasks)}
        agg = evaluate_hota(d_out, [d["video_id"] for d in data[s]], split=meta[s][2])
        per["__pooled__"] = {k: agg.get(k) for k in ("HOTA", "IDF1", "AssA", "IDSw")}
        return per
    H = lambda per: float(per["__pooled__"]["HOTA"])
    none = {s: run("none", "base", "plain", s, 2.0) for s in ("dev", "test", "val7")}
    rep["stitch"]["none"] = {s: H(none[s]) for s in none}
    print(f"[STITCH] none: test {rep['stitch']['none']['test']:.3f} val7 {rep['stitch']['none']['val7']:.3f}", flush=True)
    per_key = {}
    for h in PAIR_HEADS:
        for var in ("plain", "nofly", "fly"):
            key = f"{h}*{var}"
            dev = {t: run(key, h, var, "dev", t) for t in GRID}
            thr = max(dev, key=lambda t: H(dev[t]))
            per_key[key] = {"test": run(key, h, var, "test", thr), "val7": run(key, h, var, "val7", thr)}
            rep["stitch"][key] = {"threshold": thr, "dev": H(dev[thr]), "test": H(per_key[key]["test"]), "val7": H(per_key[key]["val7"]), "dev_grid": {str(t): H(r) for t, r in dev.items()},
                                  "pooled": {s: per_key[key][s]["__pooled__"] for s in ("test", "val7")}}
            r = rep["stitch"][key]
            print(f"[STITCH] {key:16s} порог {thr}: dev {r['dev']:.3f} test {r['test']:.3f} val7 {r['val7']:.3f} | Δ test {r['test'] - rep['stitch']['none']['test']:+.3f} val7 {r['val7'] - rep['stitch']['none']['val7']:+.3f}", flush=True)
            (OUT / "summary.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    cmp_ = [(k, "none") for k in per_key] + [(f"{h}*nofly", f"{h}*plain") for h in PAIR_HEADS] + [(f"{h}*fly", f"{h}*nofly") for h in PAIR_HEADS] + [(f"{h}*fly", f"{h}*plain") for h in PAIR_HEADS] + [("base_masa*fly", "base*plain"), ("base_masa*nofly", "base*plain")]
    for x, y in cmp_:
        for s in ("test", "val7"):
            A = per_key[x][s]; B = none[s] if y == "none" else per_key[y][s]
            d = np.array([A[v]["HOTA"] - B[v]["HOTA"] for v in A if v != "__pooled__"]); b = d[rng.integers(0, len(d), (5000, len(d)))].mean(1)
            rep["diff"][f"{x}-{y}|{s}"] = {"mean": float(d.mean()), "ci95": [float(np.quantile(b, .025)), float(np.quantile(b, .975))], "better": int((d > 0).sum()), "n": len(d)}
            r = rep["diff"][f"{x}-{y}|{s}"]
            print(f"[DIFF] {x} − {y} {s}: HOTA {r['mean']:+.3f} [{r['ci95'][0]:+.3f}; {r['ci95'][1]:+.3f}] лучше {r['better']}/{r['n']}", flush=True)
    (OUT / "summary.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print("[OK]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
