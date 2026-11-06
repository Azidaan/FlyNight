"""Суммирование движения T4/T5 по области рамки и по широкому полю (принцип клеток HS/VS) — карточка fly_wide_field_v1.

Штатная муха (модель 000, вход lcn24, сетка 2×3, как в fly_input_norm_v1). За один проход мухи на каждый сырой кадр
сохраняются по колонкам: выход декодера Sintel в px (усиление 1, за вычетом состояния после fade-in) и средняя за шаг
активность 8 подтипов T4a–d / T5a–d (за вычетом состояния после fade-in). Кэш — runs/fly_wide_field/cache/<id>.npz.
Считывания скорости рамки на кадр:
  stock    — штатное: медиана декодера по рамке с уверенностью (propagate_box), усиление 3 (fly_input_norm_v1);
  dec_pool — линейное считывание средних выходов декодера внутри рамки и по всему полю (контроль одним отличием);
  t45_pool — линейное считывание средних активностей T4/T5 внутри рамки и по всему полю (новая муха).
Внутри рамки — колонки, попавшие в рамку (меньше 3 — 3 ближайшие, как MotionField.estimate_box); поле — все колонки.
Веса (гребневая регрессия на стандартизованных признаках) — по 6 видео train fly_input_norm_v1: цель — смещение центра
эталонной рамки за кадр между соседними размеченными кадрами, рамка — линейная интерполяция. Оценка — метрика Д4
(перенос эталонной рамки между соседними размеченными кадрами, медиана ошибки центра, px) на 17 видео dev пилота К1;
ноль, DIS и stock — функцией fly_scale_probe.errors (сверка с runs/fly_scale_probe/real.json, 2x3s1_lcn24).

Запуск: python scripts/fly_wide_field.py --smoke        (1 видео train: сверка stock и время)
        python scripts/fly_wide_field.py --stage1       (6 train + 17 dev, ≈ 2 ч GPU)
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

import scripts.fly_finetune_k1 as K  # noqa: E402
from scripts.fly_scale_probe import errors, input_transform, make_fly  # noqa: E402

OUT = PROJECT_ROOT / "runs" / "fly_wide_field"
T45 = ["T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d"]
GAIN_STOCK = 3.0
MIN_COLS = 3
MAX_GAP = 12          # соседние размеченные кадры (5 к/с ≈ 6 сырых кадров); дальше пары не берутся
RIDGE = 1.0


def split_videos():
    """Те же 6 видео train (подбор) и 17 dev (оценка), что в fly_scale_probe.real."""
    sel = json.loads((K.TC / "selection.json").read_text(encoding="utf-8"))["videos"]
    meta = json.loads((PROJECT_ROOT / "runs/fly_finetune_k1/summary.json").read_text(encoding="utf-8"))["meta"]
    cond = dict(meta["cond"]); pilot_train = set(meta["train"])
    ct = K.conditions(sel["train"], "train"); cond.update(ct)
    tr = [v for v in sel["train"] if v not in pilot_train and ct.get(v) == "night"][:3] + [v for v in sel["train"] if v not in pilot_train and ct.get(v) == "normal"][:3]
    return tr, list(meta["eval"]), cond


def video_features(fly, path, tf):
    """(точки (P, 2), декодер px (n, P, 2), T4/T5 (n, P, 8)); кадр 0 — fade-in, нули."""
    t = fly.torch; dec = fly.base.decoder
    li = fly.net.stimulus.layer_index
    idx = t.as_tensor(np.stack([np.asarray(li[x]) for x in T45]), device=fly.dev)     # (8, 721)
    raw = list(K.read_gray(path)); n = len(raw)
    dvec = None; avec = None; pts = None
    with t.no_grad():
        for i, g in enumerate(raw):
            cur = fly.render(t.from_numpy(fly.tiles(tf(g))).to(fly.dev, t.float32).unsqueeze(1) / 255.0)
            if i == 0:
                pts, s = fly.points_scale(); sc = t.tensor(s, device=fly.dev); P = len(pts)
                dvec = np.zeros((n, P, 2), np.float16); avec = np.zeros((n, P, 8), np.float16)
                state = fly.fade_in(cur); last = cur
                base_d = dec(state.nodes.activity.unsqueeze(1))[:, 0]; base_a = state.nodes.activity[:, idx]   # (6, 2, 721), (6, 8, 721)
                continue
            state, act = fly.step(state, last, cur); last = cur
            dvec[i] = K.to_px(dec(act).mean(1) - base_d, sc, t.zeros((), device=fly.dev)).cpu().numpy()
            avec[i] = (act[:, :, idx].mean(1) - base_a).permute(0, 2, 1).reshape(-1, 8).cpu().numpy()
    return pts, dvec, avec


def cached(fly, vid, tf):
    p = OUT / "cache" / f"{vid}.npz"
    if p.exists():
        z = np.load(p); return z["points"], z["dec"], z["t45"]
    pts, d, a = video_features(fly, PROJECT_ROOT / "data/bdd100k/videos/train" / f"{vid}.mov", tf)
    p.parent.mkdir(parents=True, exist_ok=True); np.savez(p, points=pts, dec=d, t45=a)
    return pts, d, a


def pool(pts, F, box):
    """Средние признаки внутри рамки (меньше MIN_COLS — ближайшие) и по всему полю → (2k,)."""
    inside = np.flatnonzero(np.all((pts >= box[:2]) & (pts <= box[2:]), axis=1))
    if len(inside) < MIN_COLS:
        c = (box[:2] + box[2:]) / 2; half = np.maximum((box[2:] - box[:2]) / 2, 1.0)
        inside = np.unique(np.concatenate([inside, np.argsort(np.linalg.norm((pts - c) / half, axis=1))[:MIN_COLS]]))
    return np.concatenate([F[inside].mean(0), F.mean(0)])


def pairs(vid, nf):
    al = json.loads((K.TC / vid / "alignment.json").read_text(encoding="utf-8"))["matches"]
    truth = json.loads((PROJECT_ROOT / "data/bdd100k/labels/box_track_20/train" / f"{vid}.json").read_text(encoding="utf-8"))
    boxes = [(int(m["video_frame_index"]), {l["id"]: np.array([l["box2d"]["x1"], l["box2d"]["y1"], l["box2d"]["x2"], l["box2d"]["y2"]], np.float32)
                                            for l in fr.get("labels") or [] if l.get("box2d") and l.get("category") in K.CATS}) for m, fr in zip(al, truth)]
    for (r0, b0), (r1, b1) in zip(boxes[:-1], boxes[1:]):
        if r1 <= r0 or r1 >= nf:
            continue
        for tid, box in b0.items():
            if tid in b1:
                yield r0, r1, box, b1[tid]


def train_rows(pts, F, vid):
    X, Y = [], []
    for r0, r1, b0, b1 in pairs(vid, len(F)):
        if r1 - r0 > MAX_GAP:
            continue
        step = ((b1[:2] + b1[2:]) - (b0[:2] + b0[2:])) / 2 / (r1 - r0)
        for r in range(r0 + 1, r1 + 1):
            w = (r - 1 - r0) / (r1 - r0); X.append(pool(pts, F[r].astype(np.float32), (1 - w) * b0 + w * b1)); Y.append(step)
    return X, Y


def fit(X, Y):
    X = np.asarray(X, np.float64); Y = np.asarray(Y, np.float64)
    mu, sd = X.mean(0), X.std(0) + 1e-9; Z = (X - mu) / sd
    A = Z.T @ Z + RIDGE * np.eye(Z.shape[1]); W = np.linalg.solve(A, Z.T @ (Y - Y.mean(0)))
    return {"mu": mu, "sd": sd, "W": W, "b": Y.mean(0)}


def predict(m, x):
    return ((x - m["mu"]) / m["sd"]) @ m["W"] + m["b"]


def pooled_errors(pts, F, m, vid):
    e = []
    for r0, r1, b0, b1 in pairs(vid, len(F)):
        pr = b0.astype(np.float64).copy()
        for r in range(r0 + 1, r1 + 1):
            v = predict(m, pool(pts, F[r].astype(np.float32), pr)); pr = pr + np.concatenate([v, v])
        e.append(float(np.linalg.norm((pr[:2] + pr[2:]) / 2 - (b1[:2] + b1[2:]) / 2)))
    return float(np.median(e)) if e else None


def status(**kw):
    OUT.mkdir(parents=True, exist_ok=True); (OUT / "status.json").write_text(json.dumps(kw, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--smoke", action="store_true"); ap.add_argument("--stage1", action="store_true")
    args = ap.parse_args()
    fps = json.loads((K.TC / "0001542f-5ce3cf52/motion/dis_quarter.json").read_text(encoding="utf-8"))["fps"]
    tf = input_transform("lcn24", fps)
    tr, ev, cond = split_videos()
    real = json.loads((PROJECT_ROOT / "runs/fly_scale_probe/real.json").read_text(encoding="utf-8"))["2x3s1_lcn24"]
    fly = make_fly(fps, "2x3", 1)
    todo = [("train", v) for v in tr] + ([("dev", v) for v in ev] if args.stage1 else [])
    if args.smoke:
        todo = todo[:1]
    res = {"train": {}, "dev": {}}; feats = {}
    t0 = time.time()
    for i, (part, v) in enumerate(todo):
        fly.bounds = None; fly.base.reset()
        ts = time.time(); pts, d, a = cached(fly, v, tf)
        dq = np.load(K.TC / v / "motion/dis_quarter.npz"); nf = min(len(d), len(dq["vectors"]))
        r = errors(pts, d[:nf].astype(np.float32), [GAIN_STOCK], (dq["points"], dq["vectors"]), v, nf)
        res[part][v] = {"zero": r["zero"], "dis": r["dis"], "stock": r[f"g{GAIN_STOCK}"], "seconds_fly": time.time() - ts}
        feats[v] = (pts, d[:nf], a[:nf])
        ref = real[part].get(v, {})
        el = time.time() - t0; eta = el / (i + 1) * (len(todo) - i - 1)
        print(f"[{i + 1}/{len(todo)}] {part} {v} ({cond.get(v)}): ноль {r['zero']:.2f} (было {ref.get('zero', float('nan')):.2f}) DIS {r['dis']:.2f} (было {ref.get('dis', float('nan')):.2f}) "
              f"stock {r[f'g{GAIN_STOCK}']:.2f} (было {ref.get(f'g{GAIN_STOCK}', float('nan')):.2f}) | {res[part][v]['seconds_fly'] / 60:.1f} мин, осталось ~{eta / 60:.0f} мин", flush=True)
        status(stage="features", done=i + 1, total=len(todo), eta_min=eta / 60)
    if args.smoke:
        (OUT / "smoke.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8"); print("[OK] smoke"); return 0
    models = {}
    for name, k in (("dec_pool", 1), ("t45_pool", 2)):
        X, Y = [], []
        for v in tr:
            x, y = train_rows(feats[v][0], feats[v][k], v); X += x; Y += y
        models[name] = fit(X, Y); print(f"[FIT] {name}: {len(X)} строк", flush=True)
    for part in ("train", "dev"):
        for v in res[part]:
            pts, d, a = feats[v]
            res[part][v]["dec_pool"] = pooled_errors(pts, d, models["dec_pool"], v)
            res[part][v]["t45_pool"] = pooled_errors(pts, a, models["t45_pool"], v)
    rep = {"train": res["train"], "dev": res["dev"], "gain_stock": GAIN_STOCK, "summary": {}}
    rng = np.random.default_rng(1)
    for grp in ("night", "all"):
        vs = [v for v in res["dev"] if grp == "all" or cond.get(v) == "night"]
        S = {k: float(np.mean([res["dev"][v][k] for v in vs])) for k in ("zero", "dis", "stock", "dec_pool", "t45_pool")}; S["videos"] = len(vs)
        for a_, b_ in (("t45_pool", "stock"), ("t45_pool", "dec_pool"), ("dec_pool", "stock"), ("t45_pool", "zero"), ("t45_pool", "dis")):
            dd = np.array([res["dev"][v][a_] - res["dev"][v][b_] for v in vs]); bs = dd[rng.integers(0, len(dd), (5000, len(dd)))].mean(1)
            S[f"{a_}-{b_}"] = {"mean": float(dd.mean()), "ci95": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))], "better": int((dd < 0).sum()), "n": len(dd)}
        S["gap_closed_stock_to_dis"] = (S["stock"] - S["t45_pool"]) / (S["stock"] - S["dis"]) if S["stock"] != S["dis"] else None
        rep["summary"][grp] = S
        print(f"== {grp} ({len(vs)}): " + " ".join(f"{k} {S[k]:.2f}" for k in ("zero", "dis", "stock", "dec_pool", "t45_pool")), flush=True)
        for k in ("t45_pool-stock", "t45_pool-dec_pool", "dec_pool-stock", "t45_pool-zero", "t45_pool-dis"):
            print(f"   {k}: {S[k]['mean']:+.3f} [{S[k]['ci95'][0]:+.3f}; {S[k]['ci95'][1]:+.3f}] лучше {S[k]['better']}/{S[k]['n']}", flush=True)
    c = rep["summary"]["night"]["t45_pool-stock"]; rep["K1_met"] = bool(c["mean"] < 0 and c["ci95"][1] < 0)
    (OUT / "stage1.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[K1] {rep['K1_met']}\n[OK]", flush=True)
    status(stage="done", K1=rep["K1_met"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
