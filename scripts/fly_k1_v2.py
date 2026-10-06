"""К1 v2: дообучение ядра мухи на нормализованном входе по единому протоколу (карточка fly_finetune_k1_core_v1).

Отличия от пилота К1 (fly_finetune_k1_pilot_v1, обучались только вход и декодер на штатном входе):
  * вход — lcn24 (fly_input_norm_v1: с ним муха лучше нулевого движения ночью), не обучается;
  * обучаются свободные параметры сети (постоянные времени, потенциалы покоя, силы синапсов — 734 числа) и декодер,
    с притяжением к решению Sintel: λ·среднее((p − p0) / (|p0| + 0,1))²;
  * единый протокол: обучение и оценка — fade-in на первом кадре, точка отсчёта — выход декодера после fade-in
    (с lcn24 точка отсчёта почти не важна, fly_reference_point_v1), цель — DIS в ближайшем узле; на обучении
    разгон WARM кадров без градиента, затем LEN кадров с градиентом; масштаб — обучаемый, начало 3,0 (по train).
Ветви: decoder (контроль: декодер + масштаб, сеть штатная) и core (сеть + декодер + масштаб).
Оценка: 17 видео dev пилота К1, метрика Д4 (перенос рамки), штатная муха + lcn24 при масштабе 3,0 — опорная;
плюс отклик на известные скорости 1–8 px/кадр на кадрах dev (сдвиг без отражённых краёв).

Запуск: python scripts/fly_k1_v2.py [--steps 2000] [--eval-only]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import scripts.fly_finetune_k1 as K  # noqa: E402
from scripts.fly_noise_probe import lcn  # noqa: E402

OUT = PROJECT_ROOT / "runs" / "fly_k1_v2"
WARM, LEN, SEG_PER_VIDEO = 20, 12, 8
GAIN0 = 3.0
ANCHOR = 1.0
ARMS = ("decoder", "core")
FREE = ("nodes_bias", "nodes_time_const", "edges_syn_strength")
SPEEDS = [1.0, 2.0, 4.0, 8.0]


def status(**kw):
    kw["time"] = time.strftime("%H:%M:%S")
    (OUT / "status.json").write_text(json.dumps(kw, ensure_ascii=False), encoding="utf-8")


def build_train(fly, vids, rng, tf):
    segs = []
    pts = None
    for i, v in enumerate(vids):
        dq = np.load(K.TC / v / "motion/dis_quarter.npz")
        nf = len(dq["vectors"])
        starts = np.sort(rng.choice(np.arange(1, nf - WARM - LEN - 1), SEG_PER_VIDEO, replace=False))
        want = {s + j for s in starts for j in range(WARM + LEN)}
        buf = {}
        for f, g in enumerate(K.read_gray(PROJECT_ROOT / "data/bdd100k/videos/train" / f"{v}.mov")):
            if f in want:
                buf[f] = np.clip(fly.tiles(tf(g)), 0, 255).round().astype(np.uint8)
            if f > starts[-1] + WARM + LEN:
                break
        if pts is None:
            pts, _ = fly.points_scale()
        idx = K.nearest_dis(dq["points"], pts)
        for s in starts:
            fr = [buf.get(s + j) for j in range(WARM + LEN)]
            if any(x is None for x in fr):
                continue
            segs.append((np.stack(fr), np.nan_to_num(dq["vectors"][s + WARM:s + WARM + LEN][:, idx].astype(np.float32))))
        print(f"[DATA] {i + 1}/{len(vids)} {v}: отрезков {len(segs)}", flush=True)
        status(stage="data", done=i + 1, total=len(vids))
    return segs


def train_arm(arm, fly, segs, steps, rng, log):
    t = fly.torch; dec, net = fly.base.decoder, fly.net
    _, sc = fly.points_scale(); sc = t.tensor(sc, device=fly.dev)
    log_gain = t.tensor(math.log(GAIN0), device=fly.dev, requires_grad=True)
    dec.eval()
    for p in dec.parameters():
        p.requires_grad_(True)
    groups = [{"params": [log_gain], "lr": 1e-2}, {"params": list(dec.parameters()), "lr": 1e-3}]
    net_params, p0 = [], []
    if arm == "core":
        for name, p in net.named_parameters():
            if name in FREE:   # свободные 734 параметра; знаки и числа синапсов (коннектом) закреплены
                p.requires_grad_(True); net_params.append(p); p0.append(p.detach().clone())
        # шаг по масштабу параметров (медианы: потенциалы покоя ≈ 0,54, постоянные времени ≈ 0,02, силы синапсов ≈ 0,018):
        # за 2000 шагов Adam сдвигает параметр не больше чем на 2000·lr — порядка десятков процентов в худшем случае
        LR = {"nodes_bias": 1e-4, "nodes_time_const": 5e-6, "edges_syn_strength": 5e-6}
        names = [nm for nm, _ in net.named_parameters() if nm in FREE]
        for nm, p in zip(names, net_params):
            groups.append({"params": [p], "lr": LR[nm]})
    opt = t.optim.Adam(groups)
    started = time.time(); hist = []
    for it in range(steps):
        frames, tgt = segs[rng.integers(len(segs))]
        cart = t.from_numpy(frames).to(fly.dev, t.float32).unsqueeze(2) / 255.0
        tgt = t.from_numpy(tgt).to(fly.dev)
        with t.no_grad():
            h0 = fly.render(cart[0]); state = fly.fade_in(h0); last = h0
            act0 = state.nodes.activity.unsqueeze(1)
            for j in range(1, WARM):
                cur = fly.render(cart[j]); state, _ = fly.step(state, last, cur); last = cur
        base = dec(act0)[:, 0]
        loss = 0.0
        for j in range(LEN):
            cur = fly.render(cart[WARM + j]); state, act = fly.step(state, last, cur); last = cur
            err = K.to_px(dec(act).mean(1) - base, sc, log_gain) - tgt[j]
            loss = loss + t.sqrt((err ** 2).sum(1) + 1.0).mean()
        loss = loss / LEN
        reg = sum(((p - q) / (q.abs() + 0.1)).pow(2).mean() for p, q in zip(net_params, p0)) / max(len(net_params), 1) if net_params else t.zeros((), device=fly.dev)
        opt.zero_grad(); (loss + ANCHOR * reg).backward(); opt.step()
        if net_params:
            with t.no_grad():
                net.clamp()
        hist.append(float(loss))
        if (it + 1) % 50 == 0 or it == 0:
            el = time.time() - started; eta = el / (it + 1) * (steps - it - 1)
            msg = f"[TRAIN {arm}] {it + 1}/{steps} потеря {np.mean(hist[-50:]):.3f} притяжение {float(reg):.4f} масштаб {float(log_gain.exp()):.2f} | {el / 60:.1f} мин, осталось ~{eta / 60:.1f} мин"
            print(msg, flush=True); log.write(msg + "\n"); log.flush()
            status(stage="train", arm=arm, step=it + 1, total=steps, eta_min=eta / 60)
    drift = {}
    if net_params:
        for (name, p), q in zip([(n, p) for n, p in net.named_parameters() if n in FREE], p0):
            drift[name] = float(((p.detach() - q).abs() / (q.abs() + 1e-6)).median())
    out = {"log_gain": float(log_gain.detach()), "decoder": {k: v.detach().clone() for k, v in dec.state_dict().items()},
           "network": {k: v.detach().clone() for k, v in net.state_dict().items()} if net_params else None,
           "loss_hist": hist, "drift_median_rel": drift}
    for p in list(dec.parameters()) + net_params:
        p.requires_grad_(False)
    return out


def fields_for_video(fly, path, tf, configs):
    """configs: имя → (network_state | None, decoder_state, log_gain). Одна симуляция на каждую различную сеть."""
    t = fly.torch; dec, net = fly.base.decoder, fly.net
    stock_net = {k: v.clone() for k, v in net.state_dict().items()}
    nets = {}
    for name, (ns, ds, lg) in configs.items():
        nets.setdefault("stock" if ns is None else name, ns)
    raw = [g for g in K.read_gray(path)]
    out = {}
    for nk, ns in nets.items():
        net.load_state_dict(stock_net if ns is None else ns)
        users = [n for n, (s, _, _) in configs.items() if ("stock" if s is None else n) == nk]
        res = {n: [] for n in users}; state = last = None; base = {}
        with t.no_grad():
            for g in raw:
                cart = t.from_numpy(np.clip(fly.tiles(tf(g)), 0, 255).round().astype(np.float32)).to(fly.dev).unsqueeze(1) / 255.0
                cur = fly.render(cart)
                if state is None:
                    state = fly.fade_in(cur); _, sc = fly.points_scale(); sct = t.tensor(sc, device=fly.dev)
                    for n in users:
                        dec.load_state_dict(configs[n][1]); base[n] = dec(state.nodes.activity.unsqueeze(1))[:, 0]
                        res[n].append(np.zeros((6 * 721, 2), np.float32))
                else:
                    state, act = fly.step(state, last, cur)
                    for n in users:
                        dec.load_state_dict(configs[n][1])
                        res[n].append(K.to_px(dec(act).mean(1) - base[n], sct, t.tensor(configs[n][2], device=fly.dev)).cpu().numpy())
                last = cur
        for n in users:
            out[n] = np.stack(res[n])
        if hasattr(tf, "reset"):
            pass
    net.load_state_dict(stock_net)
    return out


def speed_check(fly, configs, fps):
    """Отклик на известные скорости: ошибка/ноль на колонку при обученном масштабе (без подбора), кадры dev, без отражённых краёв."""
    import cv2
    t = fly.torch; dec, net = fly.base.decoder, fly.net
    stock_net = {k: v.clone() for k, v in net.state_dict().items()}
    sel = json.loads((K.TC / "selection.json").read_text(encoding="utf-8"))["videos"]
    rng = np.random.default_rng(5); frames = []
    for v in sel["dev"][20:26]:
        g = [x for _, x in zip(range(300), K.read_gray(PROJECT_ROOT / "data/bdd100k/videos/train" / f"{v}.mov"))]
        frames += [g[i] for i in rng.choice(len(g), 2, replace=False)]
    tf = lcn(24.0); res = {}
    for name, (ns, ds, lg) in configs.items():
        net.load_state_dict(stock_net if ns is None else ns); dec.load_state_dict(ds)
        per = {}
        for sp in SPEEDS:
            errs, zero, par = [], [], []
            for g in frames:
                a = rng.uniform(0, 2 * np.pi); u = np.array([np.cos(a), np.sin(a)]); vel = u * sp
                H, W = g.shape
                seq = [g] * 21 + [cv2.warpAffine(g, np.float32([[1, 0, vel[0] * j], [0, 1, vel[1] * j]]), (W, H), borderMode=cv2.BORDER_CONSTANT) for j in range(1, 9)]
                tiles = [np.clip(fly.tiles(tf(x)), 0, 255).round().astype(np.float32) for x in seq]
                pts, sc = fly.points_scale(); sct = t.tensor(sc, device=fly.dev); end = pts - vel * 8
                ok = (pts[:, 0] > 24) & (pts[:, 0] < W - 24) & (pts[:, 1] > 24) & (pts[:, 1] < H - 24) & (end[:, 0] > 24) & (end[:, 0] < W - 24) & (end[:, 1] > 24) & (end[:, 1] < H - 24)
                with t.no_grad():
                    cart = lambda x: t.from_numpy(x).to(fly.dev).unsqueeze(1) / 255.0
                    cur = fly.render(cart(tiles[0])); st = fly.fade_in(cur); last = cur; base = dec(st.nodes.activity.unsqueeze(1))[:, 0]
                    for j in range(1, len(tiles)):
                        cur = fly.render(cart(tiles[j])); st, act = fly.step(st, last, cur); last = cur
                        if j > 20:
                            fl = K.to_px(dec(act).mean(1) - base, sct, t.tensor(lg, device=fly.dev)).cpu().numpy()[ok]
                            errs.append(np.linalg.norm(fl - vel, axis=1).mean()); zero.append(sp); par.append(float((fl @ u).mean()))
            per[str(sp)] = {"epe_ratio": float(np.mean(errs) / np.mean(zero)), "parallel_over_speed": float(np.mean(par) / sp)}
        res[name] = per
        print(f"[SPEED] {name}: " + " ".join(f"{sp}: ошибка/ноль {per[str(sp)]['epe_ratio']:.3f}, проекция/скорость {per[str(sp)]['parallel_over_speed']:.2f};" for sp in SPEEDS), flush=True)
    net.load_state_dict(stock_net)
    return res


def main() -> int:
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="2 видео обучения, 20 шагов, 1 видео оценки")
    a = ap.parse_args()
    import torch
    OUT.mkdir(parents=True, exist_ok=True)
    fps = json.loads((K.TC / "0001542f-5ce3cf52/motion/dis_quarter.json").read_text(encoding="utf-8"))["fps"]
    fly = K.Fly(fps, gain=1.0)
    stock_dec = {k: v.detach().clone() for k, v in fly.base.decoder.state_dict().items()}
    stock_net = {k: v.detach().clone() for k, v in fly.net.state_dict().items()}
    meta = json.loads((PROJECT_ROOT / "runs/fly_finetune_k1/summary.json").read_text(encoding="utf-8"))["meta"]
    tr, ev, cond = meta["train"], meta["eval"], meta["cond"]
    if a.smoke:
        OUT = PROJECT_ROOT / "runs" / "fly_k1_v2_smoke"; OUT.mkdir(parents=True, exist_ok=True)
        tr, ev, a.steps = tr[:1] + tr[-1:], ev[1:2], 20
    tf = lcn(24.0)
    arms_path = OUT / "arms.pt"
    if a.eval_only and arms_path.exists():
        arms = torch.load(arms_path, weights_only=False)
    else:
        segs = build_train(fly, tr, np.random.default_rng(0), tf)
        arms = {}
        with open(OUT / "train_log.txt", "w", encoding="utf-8") as log:
            for arm in ARMS:
                fly.base.decoder.load_state_dict(stock_dec); fly.net.load_state_dict(stock_net)
                arms[arm] = train_arm(arm, fly, segs, a.steps, np.random.default_rng(1), log)
                torch.save(arms, arms_path)
        fly.base.decoder.load_state_dict(stock_dec); fly.net.load_state_dict(stock_net)
        del segs
    configs = {"stock": (None, stock_dec, math.log(GAIN0))}
    for arm in ARMS:
        configs[arm] = (arms[arm]["network"], arms[arm]["decoder"], arms[arm]["log_gain"])
    spd = speed_check(fly, configs, fps)
    (OUT / "speed.json").write_text(json.dumps(spd, ensure_ascii=False, indent=1), encoding="utf-8")
    res_path = OUT / "per_video.json"
    res = json.loads(res_path.read_text(encoding="utf-8")) if res_path.exists() else {}
    pts, _ = fly.points_scale(); t0 = time.time()
    for i, v in enumerate(ev):
        if v in res:
            continue
        fields = fields_for_video(fly, PROJECT_ROOT / "data/bdd100k/videos/train" / f"{v}.mov", tf, configs)
        dq = np.load(K.TC / v / "motion/dis_quarter.npz")
        nf = min(len(dq["vectors"]), *(len(x) for x in fields.values()))
        res[v] = K.box_errors(fields, pts, (dq["points"], dq["vectors"]), K.TC, "train", v, nf)
        res_path.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        el = time.time() - t0
        print(f"[EVAL] {i + 1}/{len(ev)} {v} ({cond[v]}): " + " | ".join(f"{k} {r['median']:.2f}" for k, r in res[v].items()) + f" | осталось ~{el / (i + 1) * (len(ev) - i - 1) / 60:.0f} мин", flush=True)
        status(stage="eval", done=i + 1, total=len(ev))
    rng = np.random.default_rng(1); summ = {}
    for g in ("night", "all"):
        vs = [v for v in ev if g == "all" or cond[v] == "night"]
        s = {k: float(np.mean([res[v][k]["median"] for v in vs])) for k in res[vs[0]]}; s["videos"] = len(vs)
        for a_, b_ in (("core", "stock"), ("core", "decoder"), ("decoder", "stock"), ("core", "zero"), ("stock", "zero")):
            d = np.array([res[v][a_]["median"] - res[v][b_]["median"] for v in vs]); bs = d[rng.integers(0, len(d), (5000, len(d)))].mean(1)
            s[f"{a_}-{b_}"] = {"mean": float(d.mean()), "ci95": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]}
        summ[g] = s
        print(f"== {g} ({len(vs)}): ноль {s['zero']:.2f} DIS {s['dis']:.2f} штатная+lcn24 {s['stock']:.2f} декодер {s['decoder']:.2f} ядро {s['core']:.2f} | ядро − штатная {s['core-stock']['mean']:+.3f} [{s['core-stock']['ci95'][0]:+.3f}; {s['core-stock']['ci95'][1]:+.3f}] | ядро − декодер {s['core-decoder']['mean']:+.3f} [{s['core-decoder']['ci95'][0]:+.3f}; {s['core-decoder']['ci95'][1]:+.3f}]", flush=True)
    arms_info = {k: {"gain": math.exp(v["log_gain"]), "final_loss": float(np.mean(v["loss_hist"][-100:])), "first_loss": float(np.mean(v["loss_hist"][:50])), "drift_median_rel": v["drift_median_rel"]} for k, v in arms.items()}
    (OUT / "summary.json").write_text(json.dumps({"summary": summ, "arms": arms_info, "speed": spd, "meta": {"train": tr, "eval": ev, "warm": WARM, "len": LEN, "anchor": ANCHOR, "steps": a.steps}}, ensure_ascii=False, indent=1), encoding="utf-8")
    status(stage="done")
    print(f"[OK] {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
