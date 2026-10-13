"""З13 (карточка fly_object_readout_z13_v1): обучение считывателя и ядра, выбор τ, техническая проверка Т2.

Варианты: B — рекуррентная сеть на решётке без FlyVis (tracking/z13_readout.HexGRU); C — замороженная FlyVis; D — обучаемая FlyVis
(734 свободных параметра, притяжение λ = 1, шаги lr как К1 v2). Считыватель одинаковый (tracking/z13_readout.Readout).
Данные — runs/z13_data/<id>.npz (scripts/z13_data.py): train 130 → обучение 104 видео / выбор τ 26 (деление по видео, сид 0).
Пример — помеченный кадр (kind 1 — цель-колонка, kind 0 — «отказ»; kind −1 в обучение не входит); эпизод прогоняется
от начала до этого кадра. Adam, 4000 итераций, пакет 16, перекрёстная энтропия по 722 вариантам, считыватель и сеть B — lr 3e-3.
Правило τ — на 26 видео, эпизоды без возмущения, все помеченные кадры: успех — kind 1, ошибка центра ≤ 0,25 · max(ш, в) рамки
и меньше ошибки штатного прогноза; вмешательство на kind 0 — ошибка; τ — наименьший порог с долей успехов ≥ 0,9 при ≥ 50
вмешательствах; нет такого — вариант ничего не меняет.

Запуск:  python scripts/z13_train.py --arm D            (обучение + τ → runs/z13/<arm>/model.pt, tau.json)
         python scripts/z13_train.py --t2               (Т2: 300 итераций на 500 эпизодах, C и D на одних пакетах, B)
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

DATA = PROJECT_ROOT / "runs/z13_data"
OUT = PROJECT_ROOT / "runs/z13"
ITERS, BATCH, LR_READOUT, ANCHOR = 4000, 16, 3e-3, 1.0
SUCCESS_SCALE, PRECISION, MIN_INTERVENTIONS = 0.25, 0.9, 50


class Episodes:
    """Все эпизоды набора видео в памяти; пример — (эпизод, номер кадра поиска, цель)."""

    def __init__(self, vids: list[str]) -> None:
        hx, ms, off, lab, meta = [], [], [0], [], []
        for v in vids:
            # Each compressed array is decoded once, rather than once per label.
            with np.load(DATA / f"{v}.npz") as archive:
                d = {key: archive[key] for key in archive.files}
            base_f, base_e = off[-1], len(off) - 1
            o = d["ep_offset"]
            n_ep = len(o) - 1
            if len(d["ex_mask"]) != n_ep or int(o[-1]) != len(d["fr_hex"]):
                raise ValueError(f"Malformed episode offsets/masks in {v}")
            hx.append(d["fr_hex"]); ms.append(d["ex_mask"])
            off.extend((o[1:] + base_f).tolist())
            for i in range(len(d["lab_kind"])):
                e = int(d["lab_ep"][i])
                if not 0 <= e < n_ep or not 0 <= int(d["lab_pos"][i]) < int(o[e + 1] - o[e]) - 4:
                    raise ValueError(f"Label points outside its search episode: {v}, label {i}")
                lab.append((base_e + e, int(d["lab_pos"][i]), int(d["lab_kind"][i]), int(d["lab_col"][i]), d["lab_center"][i], d["lab_gtbox"][i], d["lab_pred"][i], d["lab_crop"][i], bool(d["ep_pert"][e])))
            meta.append((v, len(o) - 1))
        self.hex = np.concatenate(hx) if hx else np.zeros((0, 721), np.float16)
        self.mask = np.concatenate(ms) if ms else np.zeros((0, 721), bool)
        self.off = np.asarray(off, np.int64)
        self.labels = lab; self.meta = meta

    def n_frames(self, e: int) -> int:
        return int(self.off[e + 1] - self.off[e])

    def batch(self, torch, dev, eps: list[int], upto: list[int]):
        """→ hexes (B, T, 721): кадры эпизодов до upto (номер кадра поиска) включительно; короткие дополняются последним кадром."""
        from tracking.z13_readout import N_EXEMPLAR
        T = N_EXEMPLAR + max(upto) + 1
        idx = np.stack([self.off[e] + np.minimum(np.arange(T), self.n_frames(e) - 1) for e in eps])
        hexes = torch.from_numpy(self.hex[idx].astype(np.float32)).to(dev)
        masks = torch.from_numpy(self.mask[eps]).to(dev)
        return hexes, masks


def split_videos():
    from scripts.fly_lifecycle_alive import sets
    S = sets(PROJECT_ROOT / "runs/fly_signatures", PROJECT_ROOT / "runs/fly_signatures_val")
    vids = sorted(v for v, *_ in S["train"] if (DATA / f"{v}.npz").exists())
    if len(vids) != 130:
        raise ValueError(f"Protocol requires all 130 train videos, found {len(vids)}")
    rng = np.random.default_rng(0); perm = rng.permutation(len(vids))
    tau_v = sorted(vids[i] for i in perm[:26]); fit_v = sorted(vids[i] for i in perm[26:])
    return fit_v, tau_v


def build(arm: str, torch):
    from scripts.run_bdd_video_rate_experiment import make_motion_estimator
    from tracking.foveated_eyes import EyeBank
    from tracking.z13_readout import FlyBackbone, HexGeometry, make_modules
    Readout, HexGRU = make_modules(torch)
    est = make_motion_estimator("flyvis_tiled", 30.0, 0.02)
    bank = EyeBank(est); geom = HexGeometry(bank.points_local, bank.eye_size)
    dev = est.flyvis.device
    if arm == "B":
        bb = HexGRU(geom.neighbors).to(dev)
    else:
        bb = FlyBackbone(est, trainable=(arm == "D"))
    ro = Readout(bb.out_dim).to(dev)
    return est, bank, geom, bb, ro, dev


def backbone_groups(arm, bb):
    if arm == "B":
        return [{"params": list(bb.parameters()), "lr": LR_READOUT}]
    return bb.param_groups() if arm == "D" else []


def fit_stats(torch, data, bb, ro, dev, rng, n=64):
    """Стандартизация признаков считывателя по train (до обучения)."""
    from tracking.z13_readout import N_EXEMPLAR
    available = getattr(data, "stats_episodes", np.arange(len(data.off) - 1))
    eps = rng.choice(available, size=min(n, len(available)), replace=False).tolist()
    feats = []
    with torch.no_grad():
        for k in range(0, len(eps), 16):
            ch = eps[k:k + 16]; hx, _ = data.batch(torch, dev, ch, [0] * len(ch))
            st = bb.init(hx[:, 0:1]); last = hx[:, 0]
            for j in range(1, N_EXEMPLAR + 1):
                st, f = bb.step(st, last, hx[:, j]); last = hx[:, j]; feats.append(f.transpose(1, 2).reshape(-1, f.shape[1]))
    F = torch.cat(feats)
    ro.mu.copy_(F.mean(0)); ro.sd.copy_(F.std(0).clamp(min=1e-3))


def train(arm, torch, data, bb, ro, dev, iters, seed=0, log=None, probe=None, control=None, identity=None, checkpoint=None):
    from tracking.z13_readout import run_sequence
    from tracking.z13_checkpoint import TrainingPaused, restore_checkpoint, save_checkpoint
    rng = np.random.default_rng(seed)
    usable = [i for i, l in enumerate(data.labels) if l[2] >= 0]
    opt = torch.optim.Adam([{"params": list(ro.parameters()), "lr": LR_READOUT}] + backbone_groups(arm, bb))
    hist = []; start = 0; t0 = time.time(); last_saved = t0
    if checkpoint is not None:
        checkpoint = Path(checkpoint)
        if checkpoint.exists():
            if control is None or not control.resume:
                raise ValueError(f"Checkpoint already exists; use --resume: {checkpoint}")
            start, hist = restore_checkpoint(torch, checkpoint, arm, bb, ro, opt, rng, identity)
        else:
            save_checkpoint(torch, checkpoint, arm, bb, ro, opt, rng, 0, hist, identity)
    if control:
        control.status(stage="training", arm=arm, step=start, total=iters, checkpoint=str(checkpoint))
        control.check()
    for it in range(start, iters):
        pick = rng.choice(usable, size=BATCH, replace=False)
        L = [data.labels[i] for i in pick]
        eps = [l[0] for l in L]; pos = [l[1] for l in L]
        tgt = torch.as_tensor([l[3] if l[2] == 1 else 721 for l in L], device=dev)
        hx, ms = data.batch(torch, dev, eps, pos)
        logits = run_sequence(torch, bb, ro, hx, ms)   # (B, S, 722)
        sel = logits[torch.arange(len(L), device=dev), torch.as_tensor(pos, device=dev)]
        obj = torch.nn.functional.cross_entropy(sel, tgt)
        pen = bb.penalty() if hasattr(bb, "penalty") else torch.zeros((), device=dev)
        rec = {"it": it + 1, "object": float(obj.detach()), "penalty": float(pen.detach())}
        if probe is not None and (it % 25 == 0):
            rec.update(probe(obj, pen))
        opt.zero_grad(); (obj + (ANCHOR * pen if arm == "D" else 0.0)).backward(); opt.step()
        if hasattr(bb, "clamp"):
            bb.clamp()
        hist.append(rec)
        if checkpoint is not None and ((it + 1) % control.every == 0 or time.time() - last_saved >= 60 or it + 1 == iters or control.stopped()):
            save_checkpoint(torch, checkpoint, arm, bb, ro, opt, rng, it + 1, hist, identity)
            last_saved = time.time()
            control.status(step=it + 1, checkpoint_verified=True)
        if control:
            control.check()
        if (it + 1) % 50 == 0 or it == 0:
            el = time.time() - t0; msg = f"[TRAIN {arm}] {it + 1}/{iters} объектная {np.mean([h['object'] for h in hist[-50:]]):.3f} штраф {rec['penalty']:.5f} | {el / 60:.1f} мин, осталось ~{el / (it + 1) * (iters - it - 1) / 60:.1f} мин"
            print(msg, flush=True)
            if log:
                log.write(msg + "\n"); log.flush()
    return hist


def evaluate(torch, data, bb, ro, dev, geom, eye_size, label_ids, shuffle=False, seed=1, control=None):
    """Прогон эпизодов без градиента: для каждой метки — вероятность «найден», оценка центра (кадр), ошибки."""
    from tracking.z13_readout import center_estimate, run_sequence, to_frame
    by_ep = {}
    for i in label_ids:
        by_ep.setdefault(data.labels[i][0], []).append(i)
    eps = sorted(by_ep); rows = []
    perm = np.random.default_rng(seed).permutation(len(eps)) if shuffle else np.arange(len(eps))
    with torch.no_grad():
        for k in range(0, len(eps), 16):
            if control:
                control.check()
            ch = eps[k:k + 16]; mx = [max(data.labels[i][1] for i in by_ep[e]) for e in ch]
            hx, ms = data.batch(torch, dev, ch, mx)
            if shuffle:   # образцы из других эпизодов (первые 4 кадра и рамка образца)
                src = [eps[perm[k + j]] for j in range(len(ch))]
                hx2, ms2 = data.batch(torch, dev, src, [0] * len(src))
                hx[:, :4] = hx2[:, :4]; ms = ms2
            P = torch.softmax(run_sequence(torch, bb, ro, hx, ms), -1).cpu().numpy()
            for j, e in enumerate(ch):
                for i in by_ep[e]:
                    l = data.labels[i]; p = P[j, l[1]]; pres = 1.0 - float(p[721])
                    c = np.asarray(to_frame(center_estimate(p[:721], geom), tuple(int(v) for v in l[7]), eye_size))
                    row = {"label": i, "kind": l[2], "pert": l[8], "p_present": pres, "center": c.tolist(), "pred": np.asarray(l[6], float).tolist()}
                    if l[2] == 1:
                        gt = np.asarray(l[4], float); b = np.asarray(l[5], float); scale = max(b[2] - b[0], b[3] - b[1])
                        row.update(err=float(np.linalg.norm(c - gt)), err_pred=float(np.linalg.norm(np.asarray(l[6], float) - gt)), scale=float(scale),
                                   col_hit=int(np.argmax(p[:721])) == l[3])
                    rows.append(row)
    return rows


def success(r) -> bool:
    return r["kind"] == 1 and r["err"] <= SUCCESS_SCALE * r["scale"] and r["err"] < r["err_pred"]


def choose_tau(rows):
    """Наименьший τ с долей успешных вмешательств ≥ 0,9 при ≥ 50 вмешательствах (kind 0 — ошибка, kind −1 — отдельно)."""
    lab = [r for r in rows if r["kind"] >= 0]
    cand = sorted({round(r["p_present"], 4) for r in lab})
    best = None
    for tau in cand:
        iv = [r for r in lab if r["p_present"] >= tau]
        if len(iv) < MIN_INTERVENTIONS:
            break
        prec = sum(success(r) for r in iv) / len(iv)
        if prec >= PRECISION:
            best = tau; break
    rep = {"tau": best}
    if best is not None:
        iv = [r for r in lab if r["p_present"] >= best]; pos = [r for r in iv if r["kind"] == 1]
        unk = [r for r in rows if r["kind"] == -1]
        rep.update(interventions=len(iv), precision=sum(success(r) for r in iv) / len(iv), on_negative=sum(r["kind"] == 0 for r in iv),
                   median_err_rel=float(np.median([r["err"] / r["scale"] for r in pos])) if pos else None,
                   median_shift_px=float(np.median([np.linalg.norm(np.asarray(r["center"]) - np.asarray(r["pred"])) for r in iv])),
                   unknown_intervention_share=(sum(r["p_present"] >= best for r in unk) / len(unk)) if unk else None, labelled=len(lab), unknown=len(unk))
    return rep


def save_model(path, arm, bb, ro, tau_rep):
    import torch
    st = {"arm": arm, "readout": ro.state_dict(), "tau": tau_rep}
    if arm == "B":
        st["backbone"] = bb.state_dict()
    elif arm == "D":
        st["network"] = {k: v.detach().cpu() for k, v in bb.net.state_dict().items()}
    torch.save(st, path)


def main() -> int:
    global OUT
    import torch
    from tracking.z13_checkpoint import RunControl, TrainingPaused, provenance, atomic_json
    ap = argparse.ArgumentParser(); ap.add_argument("--arm", choices=["B", "C", "D"]); ap.add_argument("--t2", action="store_true"); ap.add_argument("--iters", type=int, default=ITERS)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--stop-file", type=Path)
    ap.add_argument("--deadline", help="ISO timestamp with UTC offset")
    ap.add_argument("--output-root", type=Path, default=OUT)
    a = ap.parse_args()
    if not a.t2 and a.arm is None:
        ap.error("--arm or --t2 is required")
    OUT = a.output_root.resolve()
    OUT.relative_to(PROJECT_ROOT)
    control = RunControl(OUT, a.stop_file, a.deadline, a.resume)
    control.check()
    torch.manual_seed(0)
    fit_v, tau_v = split_videos()
    OUT.mkdir(parents=True, exist_ok=True)
    identity = provenance(PROJECT_ROOT, DATA, fit_v + tau_v, {
        "fit": fit_v, "tau": tau_v, "iters": a.iters, "batch": BATCH,
        "lr_readout": LR_READOUT, "anchor": ANCHOR, "seed": 0,
        "t2": a.t2, "success_scale": SUCCESS_SCALE, "precision": PRECISION,
        "minimum_interventions": MIN_INTERVENTIONS,
    })
    manifest = OUT / ("t2_manifest.json" if a.t2 else f"{a.arm}_manifest.json")
    if manifest.exists():
        if not a.resume or json.loads(manifest.read_text(encoding="utf-8")) != identity:
            raise ValueError("Existing run requires --resume with identical code/data/settings")
    else:
        atomic_json(manifest, identity)
    if a.t2:
        return t2(torch, fit_v, control, identity)
    d = OUT / a.arm; d.mkdir(parents=True, exist_ok=True)
    (d / "split.json").write_text(json.dumps({"fit": fit_v, "tau": tau_v}, ensure_ascii=False, indent=1), encoding="utf-8")
    est, bank, geom, bb, ro, dev = build(a.arm, torch)
    data = Episodes(fit_v); print(f"[DATA] обучение: видео {len(fit_v)}, эпизодов {len(data.off) - 1}, меток {len(data.labels)}", flush=True)
    checkpoint = d / "checkpoint.pt"
    if not checkpoint.exists():
        fit_stats(torch, data, bb, ro, dev, np.random.default_rng(0))
    with (d / "train_log.txt").open("a" if a.resume else "w", encoding="utf-8") as log:
        hist = train(a.arm, torch, data, bb, ro, dev, a.iters, log=log, control=control, identity=identity, checkpoint=checkpoint)
    del data
    tdata = Episodes(tau_v)
    ids = [i for i, l in enumerate(tdata.labels) if not l[8]]   # эпизоды без возмущения
    control.status(stage="calibration")
    rows = evaluate(torch, tdata, bb, ro, dev, geom, bank.eye_size, ids, control=control)
    rep = choose_tau(rows)
    (d / "tau.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    (d / "loss.json").write_text(json.dumps(hist), encoding="utf-8")
    save_model(d / "model.pt", a.arm, bb, ro, rep)
    control.status(state="completed", stage="model_ready", arm=a.arm)
    print(f"[TAU] {a.arm}: {rep}\n[OK]", flush=True)
    return 0


def t2(torch, fit_v, control=None, identity=None) -> int:
    """Т2: 300 итераций на 500 эпизодах train; C и D — одни и те же пакеты; объектная потеря, штраф и градиенты раздельно;
    отложенные 100 эпизодов — объектная потеря C и D; перемешивание образцов — доля попаданий в колонку у D."""
    from tracking.z13_readout import run_sequence
    d = OUT / "t2"; d.mkdir(parents=True, exist_ok=True)
    progress_file = d / "arms.json"
    rep = json.loads(progress_file.read_text(encoding="utf-8")) if control and control.resume and progress_file.exists() else {}
    rng = np.random.default_rng(0)
    vids = list(rng.permutation(fit_v)[:20])
    full = Episodes(vids)
    eps = rng.permutation(len(full.off) - 1)
    tr_e, ho_e = set(eps[:500].tolist()), set(eps[500:600].tolist())
    for arm in ("C", "D", "B"):
        if control:
            control.check()
        if arm in rep:
            continue
        torch.manual_seed(0)
        est, bank, geom, bb, ro, dev = build(arm, torch)
        sub = Episodes.__new__(Episodes); sub.__dict__ = dict(full.__dict__); sub.labels = [l for l in full.labels if l[0] in tr_e]
        sub.stats_episodes = np.asarray(sorted(tr_e))
        checkpoint = d / f"checkpoint_{arm}.pt" if control else None
        if checkpoint is None or not checkpoint.exists():
            fit_stats(torch, sub, bb, ro, dev, np.random.default_rng(0))

        def probe(obj, pen, bb=bb, arm=arm):
            if arm != "D":
                return {}
            g_obj = torch.autograd.grad(obj, bb.params, retain_graph=True, allow_unused=True)
            g_pen = torch.autograd.grad(pen, bb.params, retain_graph=True, allow_unused=True) if pen.requires_grad else [None] * len(bb.params)
            nrm = lambda gs: float(torch.sqrt(sum((g ** 2).sum() for g in gs if g is not None))) if any(g is not None for g in gs) else 0.0
            return {"grad_object": nrm(g_obj), "grad_penalty": nrm(g_pen)}
        t0 = time.time()
        hist = train(arm, torch, sub, bb, ro, dev, 300, probe=probe, control=control, identity=identity, checkpoint=checkpoint)
        ho = [i for i, l in enumerate(full.labels) if l[0] in ho_e and l[2] >= 0]
        losses = []
        with torch.no_grad():
            for k in range(0, len(ho), 16):
                if control:
                    control.status(stage="t2_heldout", arm=arm)
                    control.check()
                L = [full.labels[i] for i in ho[k:k + 16]]
                hx, ms = full.batch(torch, dev, [l[0] for l in L], [l[1] for l in L])
                lg = run_sequence(torch, bb, ro, hx, ms)[torch.arange(len(L), device=dev), torch.as_tensor([l[1] for l in L], device=dev)]
                losses.append(float(torch.nn.functional.cross_entropy(lg, torch.as_tensor([l[3] if l[2] == 1 else 721 for l in L], device=dev), reduction="sum")))
        r = {"seconds": time.time() - t0, "object_first50": float(np.mean([h["object"] for h in hist[:50]])), "object_last50": float(np.mean([h["object"] for h in hist[-50:]])),
             "heldout_object": sum(losses) / max(len(ho), 1), "heldout_labels": len(ho)}
        if arm == "D":
            r["grad"] = [{k: h[k] for k in ("it", "grad_object", "grad_penalty") if k in h} for h in hist if "grad_object" in h]
            r["drift_median_rel"] = float(np.median(np.concatenate([((p.detach() - q).abs() / (q.abs() + 1e-6)).flatten().cpu().numpy() for p, q in zip(bb.params, bb.p0)])))
            hol = [i for i in ho if full.labels[i][2] == 1]
            for nm, sh in (("true", False), ("shuffled", True)):
                rows = evaluate(torch, full, bb, ro, dev, geom, bank.eye_size, hol, shuffle=sh, control=control)
                r[f"col_hit_{nm}"] = float(np.mean([x["col_hit"] for x in rows])) if rows else None
        rows_ho = evaluate(torch, full, bb, ro, dev, geom, bank.eye_size, [i for i in ho if not full.labels[i][8]], control=control)
        tr_ = choose_tau(rows_ho); r["tau_heldout"] = tr_
        if tr_["tau"] is None:   # только для технической Т3 (запуск, время, счёт вмешательств), не для оценки
            tr_ = dict(tr_, tau=0.5, technical=True)
        save_model(d / f"model_{arm}.pt", arm, bb, ro, tr_)
        rep[arm] = r
        if control:
            from tracking.z13_checkpoint import atomic_json
            atomic_json(progress_file, rep)
        print(f"[T2] {arm}: {r if arm != 'D' else {k: v for k, v in r.items() if k != 'grad'}}", flush=True)
        del bb, ro
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    D, C = rep["D"], rep["C"]
    g = [x for x in D["grad"] if x.get("grad_object") is not None]
    rep["conditions"] = {"loss_falls_D": D["object_last50"] < D["object_first50"], "core_moves_D": D["drift_median_rel"] > 0,
                         "grad_object_ge_penalty_D": bool(np.median([x["grad_object"] for x in g]) >= np.median([x["grad_penalty"] for x in g])) if g else False,
                         "heldout_D_below_C": D["heldout_object"] < C["heldout_object"],
                         "shuffle_drops_D": (D["col_hit_shuffled"] or 0) < (D["col_hit_true"] or 0)}
    (d / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[T2] условия: {rep['conditions']}\n[OK]", flush=True)
    if control:
        control.status(state="completed", stage="t2_finished", conditions=rep["conditions"])
    return 0 if all(rep["conditions"].values()) else 2


if __name__ == "__main__":
    from tracking.z13_checkpoint import TrainingPaused
    try:
        raise SystemExit(main())
    except TrainingPaused as exc:
        print(f"[PAUSED] {exc}", flush=True)
        raise SystemExit(75)
