"""Проба масштаба (карточка fly_scale_probe_v1): видит ли муха движение, если сдвиг за её кадр ближе к шагу колонки.

Пилот К1 (fly_finetune_k1_pilot_v1): при плитках стенда (2×3) и 30 Гц сдвиг объекта 1–6 px/кадр — это 0,06–0,4
шага колонки глаза за кадр; поток мухи почти шум. Два рычага:
  увеличение — меньшие плитки той же сетки стенда (2×3 ≈ ×0,8; 3×5 ≈ ×1,6; 5×9 ≈ ×3);
  прореживание — муха видит каждый k-й кадр (k = 1, 2, 4), шаг интегрирования по integration_plan(fps/k).
Муха штатная (модель 000, декодер Sintel), усиление 1, затем общий подобранный масштаб.

Шаг А (--synthetic): кадры BDD dev сдвигаются целиком с постоянной скоростью 1–6 px/кадр 30 Гц, плитки — как в
трекере (fly.tiles); на колонку — корреляция поток↔истина по осям, R² среднего по плитке, и отношение ошибки конечной
точки при лучшем общем масштабе (подобран на половине отрезков, проверен на другой) к нулевому движению.
Шаг Б (--real CFG ...): на видео train (6, подбор масштаба по сетке) и dev (оценка) — метрика Д4 (перенос
эталонной рамки между соседними размеченными кадрами), сравнение с нулевым движением и DIS, бутстрэп по видео.

Запуск: python scripts/fly_scale_probe.py --synthetic
        python scripts/fly_scale_probe.py --real 3x5s2 5x9s4 2x3s1_lcn24 2x3s1_retina
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

OUT = PROJECT_ROOT / "runs" / "fly_scale_probe"
LAYOUTS = {"2x3": (2, 3), "3x5": (3, 5), "5x9": (5, 9)}
STRIDES = (1, 2, 4)
WARM, LEN = 4, 8
GAINS = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0]


def parse(cfg):
    """«3x5s2» или «2x3s1_lcn24» → (сетка, прореживание, вход)."""
    base, _, inp = cfg.partition("_")
    lay, s = base.split("s"); return lay, int(s), inp or "raw"


def input_transform(inp, fps):
    """Вход перед глазом: raw — штатный; lcnN — нормализация локального контраста (σ = N px); retina — tracking/retina.py (lamina)."""
    if inp == "raw":
        return lambda g: g
    if inp.startswith("lcn"):
        from scripts.fly_noise_probe import lcn
        return lcn(float(inp[3:]))
    if inp == "retina":
        from tracking.retina import make_retina
        ret = make_retina("lamina", fps)
        return lambda g: ret.process(g).astype(np.float32)
    raise ValueError(inp)


def make_fly(fps, lay, stride):
    from tracking.flyvis_motion import TileLayout
    r, c = LAYOUTS[lay]
    return K.Fly(fps / stride, layout=TileLayout(r, c, 0.20), gain=1.0)


def run_frames(fly, cart_iter, chunk=15):
    """Поток мухи (px при усилении 1) по последовательности тайлов; первый кадр — fade-in, поле нулевое.
    Плитки независимы и считаются порциями по chunk (45 плиток 5×9 разом не помещаются в 12 ГБ GPU)."""
    t = fly.torch; dec = fly.base.decoder
    out = []; st = None; sc = None
    with t.no_grad():
        for tiles in cart_iter:
            cart = t.from_numpy(tiles).to(fly.dev, t.float32).unsqueeze(1) / 255.0
            parts = [slice(i, min(i + chunk, len(cart))) for i in range(0, len(cart), chunk)]
            if st is None:
                _, s = fly.points_scale(); sc = t.tensor(s, device=fly.dev)
                st = []
                for sl in parts:
                    cur = fly.render(cart[sl]); state = fly.fade_in(cur)
                    st.append([state, cur, dec(state.nodes.activity.unsqueeze(1))[:, 0]])
                out.append(None); continue
            fl = []
            for sl, S in zip(parts, st):
                cur = fly.render(cart[sl]); S[0], act = fly.step(S[0], S[1], cur); S[1] = cur
                fl.append(dec(act).mean(1) - S[2])
            out.append(K.to_px(t.cat(fl), sc, t.zeros((), device=fly.dev)).cpu().numpy())
    return out


def synthetic(fps, only=None):
    import cv2
    sel = json.loads((K.TC / "selection.json").read_text(encoding="utf-8"))["videos"]
    rng = np.random.default_rng(0)
    frames = []
    for v in sel["dev"][:10]:
        g = [x for _, x in zip(range(300), K.read_gray(PROJECT_ROOT / "data/bdd100k/videos/train" / f"{v}.mov"))]
        frames += [g[i] for i in rng.choice(len(g), 4, replace=False)]
    vels = [np.array([np.cos(a), np.sin(a)]) * s for a, s in zip(rng.uniform(0, 2 * np.pi, len(frames)), rng.uniform(1, 6, len(frames)))]
    sp = OUT / "synthetic.json"
    res = json.loads(sp.read_text(encoding="utf-8")) if sp.exists() else {}
    for lay in LAYOUTS:
        for k in STRIDES:
            cfg = f"{lay}s{k}"; t0 = time.time()
            if cfg in res:
                continue
            fly = make_fly(fps, lay, k)
            F, Y, M, seg = [], [], [], []
            for si, (g, vel) in enumerate(zip(frames, vels)):
                seq = (fly.tiles(cv2.warpAffine(g, np.float32([[1, 0, vel[0] * j * k], [0, 1, vel[1] * j * k]]), (g.shape[1], g.shape[0]), borderMode=cv2.BORDER_REFLECT))
                       for j in range(WARM + LEN))
                flows = run_frames(fly, seq)[WARM:]
                for fl in flows:
                    F.append(fl); Y.append(np.broadcast_to(vel * k, fl.shape)); seg.append(si)
                    M.append(fl.reshape(len(fly.bounds), 721, 2).mean(1))
            F, Y, M, seg = np.stack(F), np.stack(Y), np.stack(M), np.asarray(seg)
            Ym = Y[:, :len(fly.bounds) * 721:721]   # истина на плитку
            fit, ev = seg % 2 == 0, seg % 2 == 1
            g = float((F[fit] * Y[fit]).sum() / (F[fit] ** 2).sum())
            gm = float((M[fit] * Ym[fit]).sum() / (M[fit] ** 2).sum())
            epe = lambda P, T: float(np.linalg.norm(P - T, axis=-1).mean())
            r = {"corr_col_x": float(np.corrcoef(F[..., 0].ravel(), Y[..., 0].ravel())[0, 1]),
                 "corr_col_y": float(np.corrcoef(F[..., 1].ravel(), Y[..., 1].ravel())[0, 1]),
                 "R2_tile_mean": float(1 - ((Ym[ev] - gm * M[ev]) ** 2).sum() / ((Ym[ev] - Ym[ev].mean((0, 1))) ** 2).sum()),
                 "gain_col": g, "gain_tile": gm,
                 "epe_ratio_col": epe(g * F[ev], Y[ev]) / epe(0 * F[ev], Y[ev]),
                 "epe_ratio_tile": epe(gm * M[ev], Ym[ev]) / epe(0 * M[ev], Ym[ev]),
                 "tiles": len(fly.bounds), "shift_per_fly_frame_px": [1 * k, 6 * k],
                 "tile_px": [fly.bounds[0][2] - fly.bounds[0][0], fly.bounds[0][3] - fly.bounds[0][1]]}
            res[cfg] = r
            print(f"[SYN] {cfg}: плитка {r['tile_px']} px, corr колонки x/y {r['corr_col_x']:.3f}/{r['corr_col_y']:.3f}, R² плитки {r['R2_tile_mean']:.3f}, "
                  f"ошибка/ноль: колонка {r['epe_ratio_col']:.3f}, плитка {r['epe_ratio_tile']:.3f}, масштаб {g:.2f} | {time.time() - t0:.0f} с", flush=True)
            K.status(stage="synthetic", cfg=cfg)
            (OUT / "synthetic.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
            del fly
    return res


def video_field(fly, path, stride, tf=lambda g: g):
    """Поле на каждый сырой кадр: поток за интервал мухи делится поровну между k сырыми кадрами интервала."""
    raw = list(K.read_gray(path))
    flows = run_frames(fly, (fly.tiles(tf(raw[i])) for i in range(0, len(raw), stride)))
    n = len(raw); pts, _ = fly.points_scale()
    vec = np.zeros((n, len(pts), 2), np.float32)
    for i, fl in enumerate(flows):
        if fl is None:
            continue
        r = i * stride
        vec[r - stride + 1:r + 1] = fl / stride
    return pts, vec


def errors(pts, vec, gains, dis, vid, nf):
    from scripts.evaluate_flyvis_qmatch_gate import propagate_box
    al = json.loads((K.TC / vid / "alignment.json").read_text(encoding="utf-8"))["matches"]
    truth = json.loads((PROJECT_ROOT / "data/bdd100k/labels/box_track_20/train" / f"{vid}.json").read_text(encoding="utf-8"))
    boxes = [(int(m["video_frame_index"]), {l["id"]: np.array([l["box2d"]["x1"], l["box2d"]["y1"], l["box2d"]["x2"], l["box2d"]["y2"]], np.float32)
                                            for l in fr.get("labels") or [] if l.get("box2d") and l.get("category") in K.CATS}) for m, fr in zip(al, truth)]
    srcs = {"zero": (pts, None), "dis": dis}
    srcs.update({f"g{g}": (pts, g) for g in gains})
    errs = {k: [] for k in srcs}
    zero = np.zeros_like(vec[:1]).repeat(len(vec), 0)
    for (r0, b0), (r1, b1) in zip(boxes[:-1], boxes[1:]):
        if r1 <= r0 or r1 >= nf:
            continue
        for tid, box in b0.items():
            if tid in b1:
                gt = (b1[tid][:2] + b1[tid][2:]) / 2
                for k, (p, x) in srcs.items():
                    V = zero if x is None else (x if isinstance(x, np.ndarray) else vec * x)
                    pr = propagate_box(box, p, V, r0, r1)[0]
                    errs[k].append(float(np.linalg.norm((pr[:2] + pr[2:]) / 2 - gt)))
    return {k: float(np.median(e)) for k, e in errs.items() if e}


def real(fps, cfgs):
    sel = json.loads((K.TC / "selection.json").read_text(encoding="utf-8"))["videos"]
    meta = json.loads((PROJECT_ROOT / "runs/fly_finetune_k1/summary.json").read_text(encoding="utf-8"))["meta"]
    ev = meta["eval"]; cond = dict(meta["cond"])
    pilot_train = set(meta["train"])
    ct = K.conditions(sel["train"], "train"); cond.update(ct)
    tr = [v for v in sel["train"] if v not in pilot_train and ct.get(v) == "night"][:3] + [v for v in sel["train"] if v not in pilot_train and ct.get(v) == "normal"][:3]
    out_path = OUT / "real.json"
    allres = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}
    for cfg in cfgs:
        lay, k, inp = parse(cfg)
        fly = make_fly(fps, lay, k)
        R = allres.get(cfg, {"train": {}, "dev": {}})
        t0 = time.time(); todo = [("train", v) for v in tr] + [("dev", v) for v in ev]
        g_star = None
        for i, (part, v) in enumerate(todo):
            if part == "dev" and g_star is None:
                # масштаб выбирается только по train; на dev считаются ноль, DIS и выбранный масштаб (ускорение ×4)
                g_star = min(GAINS, key=lambda g: np.mean([R["train"][u][f"g{g}"] for u in R["train"]]))
            if v in R[part]:
                continue
            fly.bounds = None; fly.base.reset()
            pts, vec = video_field(fly, PROJECT_ROOT / "data/bdd100k/videos/train" / f"{v}.mov", k, input_transform(inp, fps))
            dq = np.load(K.TC / v / "motion/dis_quarter.npz")
            nf = min(len(vec), len(dq["vectors"]))
            R[part][v] = errors(pts, vec, GAINS if part == "train" else [g_star], (dq["points"], dq["vectors"]), v, nf)
            el = time.time() - t0; eta = el / (i + 1) * (len(todo) - i - 1)
            best = min((g for g in GAINS if f"g{g}" in R[part][v]), key=lambda g: R[part][v][f"g{g}"])
            print(f"[REAL {cfg}] {i + 1}/{len(todo)} {part} {v} ({cond.get(v)}): ноль {R[part][v]['zero']:.2f} DIS {R[part][v]['dis']:.2f} лучший g{best} {R[part][v][f'g{best}']:.2f} | осталось ~{eta / 60:.0f} мин", flush=True)
            K.status(stage="real", cfg=cfg, done=i + 1, total=len(todo), eta_min=eta / 60)
            allres[cfg] = R; out_path.write_text(json.dumps(allres, ensure_ascii=False, indent=1), encoding="utf-8")
        g_star = min(GAINS, key=lambda g: np.mean([R["train"][v][f"g{g}"] for v in R["train"]]))
        R["gain_from_train"] = g_star
        brng = np.random.default_rng(1)
        for grp in ("night", "all"):
            vs = [v for v in R["dev"] if grp == "all" or cond.get(v) == "night"]
            d = np.array([R["dev"][v][f"g{g_star}"] - R["dev"][v]["zero"] for v in vs])
            bs = d[brng.integers(0, len(d), (5000, len(d)))].mean(1)
            R[f"summary_{grp}"] = {"videos": len(vs), "zero": float(np.mean([R["dev"][v]["zero"] for v in vs])), "dis": float(np.mean([R["dev"][v]["dis"] for v in vs])),
                                   "fly": float(np.mean([R["dev"][v][f"g{g_star}"] for v in vs])), "fly-zero": float(d.mean()),
                                   "ci95": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))],
                                   "per_gain_train": {f"g{g}": float(np.mean([R["train"][u][f"g{g}"] for u in R["train"]])) for g in GAINS}}
            s = R[f"summary_{grp}"]
            print(f"== {cfg} {grp} ({s['videos']}): ноль {s['zero']:.2f} DIS {s['dis']:.2f} муха (g={g_star} с train) {s['fly']:.2f}; муха − ноль {s['fly-zero']:+.3f} ДИ [{s['ci95'][0]:+.3f}; {s['ci95'][1]:+.3f}]", flush=True)
        allres[cfg] = R; out_path.write_text(json.dumps(allres, ensure_ascii=False, indent=1), encoding="utf-8")
        del fly
    return allres


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--real", nargs="*", default=[])
    a = ap.parse_args()
    K.OUT = OUT; OUT.mkdir(parents=True, exist_ok=True)
    fps = json.loads((K.TC / "0001542f-5ce3cf52/motion/dis_quarter.json").read_text(encoding="utf-8"))["fps"]
    if a.synthetic:
        synthetic(fps)
    if a.real:
        real(fps, a.real)
    print(f"[OK] {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
