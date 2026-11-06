"""К1, ступень 0 (пилот): дообучение входа и декодера мухи на видео BDD (карточка fly_finetune_k1_pilot_v1).

Сеть FlyVis (модель ансамбля 000, 734 параметра) заморожена. Обучаются, по ветвям:
  scale          — один общий масштаб потока (контроль: выигрыш «только от масштаба»);
  decoder        — штатный декодер потока (7427 весов) + масштаб, вход штатный;
  input_decoder  — обучаемый вход перед глазом мухи (8 параметров: адаптация фоторецептора и
                   центр-окружение ламины, как в tracking/retina.py, но дифференцируемо; при старте
                   вход в точности штатный) + декодер + масштаб; градиент ко входу идёт через
                   замороженную сеть.
Учитель — DIS (runs/train_cache/<id>/motion/dis_quarter.npz) в ближайшем узле сетки к каждой колонке,
функция потерь — Шарбонье от ошибки конечной точки, px за кадр 30 Гц. Режим стенда: 6 плиток,
линейная пересэмплировка, среднее по 2 внутренним шагам, вычитание потока после fade-in.

Обучение: 20 видео train (10 ночь + 10 день, порядок selection.json), отрезки по 6 кадров разгона
без градиента + 12 кадров с градиентом (усечённое обратное распространение во времени).
Оценка: все видео dev (отложены от обучения мухи), ночь/день по атрибутам; метрика — как в
scripts/d4_fly_decoder_refit.py: перенос эталонной рамки полем по сырым кадрам между соседними
размеченными кадрами, расстояние центров, медиана по видео; среднее медиан по группе;
парный бутстрэп по видео относительно штатной мухи.

Критерии (записаны до прогона): ночью ошибка ≥ 30 % ниже штатной мухи; днём не хуже;
< 10 % — ступень 1 не запускать.

Запуск: python scripts/fly_finetune_k1.py [--check] [--steps 1500] [--eval-only]
  --check — сверка штатного прогона этого кода с кэшем стенда (val7, flyvis_tiled.npz) и выход.
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

OUT = PROJECT_ROOT / "runs" / "fly_finetune_k1"
TC = PROJECT_ROOT / "runs" / "train_cache"
CATS = {"pedestrian", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"}
N_TRAIN_PER_COND = 10
SEG_PER_VIDEO = 10
WARM, LEN = 6, 12
ARMS = ("scale", "decoder", "input_decoder")
STAND_GAIN = 38.84157   # DEFAULT_GAINS["flyvis_tiled"]; Д4 сравнивал кэш с усилением 1,0


def status(**kw):
    OUT.mkdir(parents=True, exist_ok=True)
    kw["time"] = time.strftime("%H:%M:%S")
    (OUT / "status.json").write_text(json.dumps(kw, ensure_ascii=False), encoding="utf-8")


def conditions(vids, split):
    from datasets.subset_filter import filter_videos_by_subset, load_video_attributes
    attrs = load_video_attributes(str(PROJECT_ROOT / f"data/bdd100k/labels/bdd100k_labels_images_{split}.json"))
    cond = {}
    for c in ("normal", "night"):
        for v in filter_videos_by_subset(attrs, vids, c):
            cond.setdefault(v, c)
    return cond


class Fly:
    """Муха стенда (FlyVisMotionEstimator: tiled, linear, mean) в виде, пригодном для градиента."""

    def __init__(self, fps: float, model="flow/0000/000", layout=None, gain=None):
        import torch
        from tracking.flyvis_motion import FlyVisMotionEstimator, TileLayout
        self.torch = torch
        # усиление стенда для плиток (DEFAULT_GAINS в run_bdd_video_rate_experiment.py): «штатная муха» = муха трекера
        self.base = FlyVisMotionEstimator(model=model, mode="tiled", source_fps=fps, temporal_resampling="linear", flow_aggregation="mean", flow_gain=STAND_GAIN if gain is None else gain,
                                          layout=layout or TileLayout(2, 3, 0.20))
        self.net, self.eye = self.base.network, self.base.eye
        self.dev = self.base.flyvis.device
        self.H = self.base.eye_height
        self.steps, self.dt = self.base.integration_steps, self.base.dt
        self.bounds = None

    def tiles(self, gray: np.ndarray) -> np.ndarray:
        import cv2
        if self.bounds is None:
            self.base._prepare_layout(gray.shape[1], gray.shape[0]); self.bounds = self.base._bounds
        return np.stack([cv2.resize(gray[y1:y2, x1:x2], (self.H, self.H), interpolation=cv2.INTER_AREA) for x1, y1, x2, y2 in self.bounds])

    def points_scale(self):
        """Координаты колонок (6·721, 2) в пикселях кадра и множитель «выход декодера → px» по плиткам."""
        f = self.base._points_and_vectors(self.torch.zeros((len(self.bounds), 2, 721), device=self.dev))
        sc = np.array([self.base.flow_gain * (y2 - y1) / self.base.kernel_area for _, y1, _, y2 in self.bounds], np.float32)
        return f.points_xy, sc

    def render(self, cart):  # (6,1,H,W) → (6,1,721)
        return self.eye(cart)

    def fade_in(self, hex0):
        with self.torch.no_grad():
            return self.net.fade_in_state(self.base.fade_in_seconds, self.dt, hex0[:, 0])

    def step(self, state, last_hex, cur_hex):
        t = self.torch
        alpha = t.arange(1, self.steps + 1, device=self.dev, dtype=t.float32).view(1, self.steps, 1, 1) / self.steps
        stim = last_hex * (1 - alpha) + cur_hex * alpha
        # свой буфер на каждый шаг: штатный Stimulus переиспользует буфер на месте, это ломает граф градиента
        st = self.net.stimulus
        idx = t.as_tensor(np.asarray(st.input_index), device=self.dev)
        buf = t.zeros((stim.shape[0], stim.shape[1], st.n_nodes), device=self.dev, dtype=t.float32)
        buf[:, :, idx] = stim.expand(-1, -1, idx.shape[0], -1)
        states = self.net.forward(buf, self.dt, state, as_states=True)
        act = t.stack([s.nodes.activity for s in states], 1)
        return states[-1], act


class FrontEnd:
    """Обучаемый вход: y = b + a1·x^γ + a2·(p − 0,5) + a3·(p − blur(p)), p = x / (x + k·m + d), m — EMA размытой яркости."""

    NAMES = ("alpha", "k", "d", "a1", "gamma", "b", "a2", "a3")

    def __init__(self, torch, dev):
        self.t = torch
        init = {"alpha": math.log(0.105 / (1 - 0.105)), "k": math.log(math.e - 1), "d": math.log(math.exp(0.05) - 1),
                "a1": 1.0, "gamma": 0.0, "b": 0.0, "a2": 0.0, "a3": 0.0}
        self.p = {k: torch.tensor(v, device=dev, requires_grad=True) for k, v in init.items()}
        self.ka, self.kl = self._gauss(16.0, dev), self._gauss(10.0, dev)
        self.m = None

    def _gauss(self, s, dev):
        r = int(3 * s); x = self.t.arange(-r, r + 1, device=dev, dtype=self.t.float32)
        g = self.t.exp(-x ** 2 / (2 * s * s)); return g / g.sum()

    def blur(self, x, g):
        F = self.t.nn.functional; r = (len(g) - 1) // 2
        x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="replicate"), g.view(1, 1, 1, -1))
        return F.conv2d(F.pad(x, (0, 0, r, r), mode="replicate"), g.view(1, 1, -1, 1))

    def reset(self):
        self.m = None

    def __call__(self, x):  # x: (6,1,H,W) в [0,1]
        F, P = self.t.nn.functional, self.p
        bx = self.blur(x, self.ka)
        a = self.t.sigmoid(P["alpha"])
        self.m = bx if self.m is None else (1 - a) * self.m + a * bx
        pr = x / (x + F.softplus(P["k"]) * self.m + F.softplus(P["d"]) + 1e-4)
        lam = pr - self.blur(pr, self.kl)
        y = P["b"] + P["a1"] * (x + 1e-4) ** self.t.exp(P["gamma"]) + P["a2"] * (pr - 0.5) + P["a3"] * lam
        return y.clamp(-0.5, 1.5)

    def detach_state(self):
        if self.m is not None:
            self.m = self.m.detach()

    def values(self):
        return {k: float(v.detach()) for k, v in self.p.items()}


def to_px(dec_flow, sc, log_gain):
    """(плитки, 2, 721) выход декодера минус базовый → (плитки·721, 2) px, ось y вниз."""
    t = dec_flow
    v = t.permute(0, 2, 1) * (sc.view(-1, 1, 1) * log_gain.exp())
    v = v * v.new_tensor([1.0, -1.0])
    return v.reshape(-1, 2)


def read_gray(path):
    import cv2
    cap = cv2.VideoCapture(str(path))
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        yield cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    cap.release()


def nearest_dis(points_dis, pts):
    from scipy.spatial import cKDTree
    return cKDTree(points_dis).query(pts, k=1)[1]


def build_train(fly, vids, rng):
    """Отрезки: тайлы uint8 (WARM+LEN, 6, H, H) и цели DIS в колонках (LEN, 4326, 2)."""
    segs = []
    pts = None
    for i, v in enumerate(vids):
        dq = np.load(TC / v / "motion/dis_quarter.npz")
        nf = len(dq["vectors"])
        starts = np.sort(rng.choice(np.arange(1, nf - WARM - LEN - 1), SEG_PER_VIDEO, replace=False))
        want = {s + j: k for k, s in enumerate(starts) for j in range(WARM + LEN)}
        buf = {}
        for f, g in enumerate(read_gray(PROJECT_ROOT / "data/bdd100k/videos/train" / f"{v}.mov")):
            if f in want:
                buf[f] = fly.tiles(g)
            if f > starts[-1] + WARM + LEN:
                break
        if pts is None:
            pts, _ = fly.points_scale()
        idx = nearest_dis(dq["points"], pts)
        for s in starts:
            fr = [buf.get(s + j) for j in range(WARM + LEN)]
            if any(x is None for x in fr):
                continue
            tgt = dq["vectors"][s + WARM:s + WARM + LEN][:, idx].astype(np.float32)
            segs.append((np.stack(fr), np.nan_to_num(tgt)))
        print(f"[DATA] {i + 1}/{len(vids)} {v}: отрезков {len(segs)}", flush=True)
        status(stage="data", done=i + 1, total=len(vids))
    return segs


def train_arm(arm, fly, segs, steps, rng, log):
    t = fly.torch
    dec = fly.base.decoder
    _, sc = fly.points_scale(); sc = t.tensor(sc, device=fly.dev)
    log_gain = t.zeros((), device=fly.dev, requires_grad=True)
    groups = [{"params": [log_gain], "lr": 1e-2}]
    fe = None
    dec.eval()
    for p in dec.parameters():
        p.requires_grad_(arm != "scale")
    if arm != "scale":
        groups.append({"params": list(dec.parameters()), "lr": 1e-3})
    if arm == "input_decoder":
        fe = FrontEnd(t, fly.dev); groups.append({"params": list(fe.p.values()), "lr": 1e-2})
    opt = t.optim.Adam(groups)
    started = time.time(); hist = []
    for it in range(steps):
        frames, tgt = segs[rng.integers(len(segs))]
        cart = t.from_numpy(frames).to(fly.dev, t.float32).unsqueeze(2) / 255.0   # (T, 6, 1, H, W)
        tgt = t.from_numpy(tgt).to(fly.dev)
        if fe is not None:
            fe.reset()
        inp = (lambda c: fe(c)) if fe is not None else (lambda c: c)
        with t.no_grad():
            h0 = fly.render(inp(cart[0]))
            state = fly.fade_in(h0)
            last = h0
            for j in range(1, WARM):
                cur = fly.render(inp(cart[j])); state, _ = fly.step(state, last, cur); last = cur
            act0 = state.nodes.activity.unsqueeze(1)
        if fe is not None:
            fe.detach_state()
        base_flow = dec(act0)[:, 0]
        loss = 0.0
        for j in range(LEN):
            cur = fly.render(inp(cart[WARM + j]))
            state, act = fly.step(state, last, cur); last = cur
            flow = dec(act).mean(1) - base_flow
            err = to_px(flow, sc, log_gain) - tgt[j]
            loss = loss + t.sqrt((err ** 2).sum(1) + 1.0).mean()
        loss = loss / LEN
        opt.zero_grad(); loss.backward(); opt.step()
        hist.append(float(loss))
        if (it + 1) % 25 == 0 or it == 0:
            el = time.time() - started; eta = el / (it + 1) * (steps - it - 1)
            msg = f"[TRAIN {arm}] {it + 1}/{steps} потеря {np.mean(hist[-25:]):.3f} масштаб {float(log_gain.exp()):.3f} | {el / 60:.1f} мин, осталось ~{eta / 60:.1f} мин"
            print(msg, flush=True); log.write(msg + "\n"); log.flush()
            status(stage="train", arm=arm, step=it + 1, total=steps, loss=float(np.mean(hist[-25:])), eta_min=eta / 60)
    state_dict = {"log_gain": float(log_gain.detach()), "decoder": {k: v.detach().clone() for k, v in dec.state_dict().items()},
                  "frontend": fe.values() if fe else None, "loss_hist": hist}
    for p in dec.parameters():
        p.requires_grad_(False)
    return state_dict


def run_video(fly, path, arms_state, stock_dec_state):
    """Поля движения (F, 4326, 2) для штатной мухи и всех ветвей за один проход по видео."""
    t = fly.torch
    dec = fly.base.decoder
    _, sc = fly.points_scale() if fly.bounds is not None else (None, None)
    pipes = {"stock": (None, stock_dec_state, 0.0)}
    for a, s in arms_state.items():
        pipes[a] = (s["frontend"], s["decoder"], s["log_gain"])
    # одна симуляция сети на каждый различный вход
    inputs = {"stock": None}
    for a, (fe_vals, _, _) in pipes.items():
        if fe_vals is not None:
            inputs[a] = fe_vals
    fes = {}
    for k, vals in inputs.items():
        if vals is None:
            fes[k] = None
        else:
            fe = FrontEnd(t, fly.dev)
            with t.no_grad():
                for n, v in vals.items():
                    fe.p[n].fill_(v)
            fes[k] = fe
    sims = {k: {"state": None, "last": None, "base_act": None} for k in fes}
    fields = {a: [] for a in pipes}
    with t.no_grad():
        for f, g in enumerate(read_gray(path)):
            cart = t.from_numpy(fly.tiles(g)).to(fly.dev, t.float32).unsqueeze(1) / 255.0
            if sc is None:
                _, sc = fly.points_scale()
            sct = t.tensor(sc, device=fly.dev)
            acts = {}
            for k, fe in fes.items():
                cur = fly.render(fe(cart) if fe is not None else cart)
                S = sims[k]
                if S["state"] is None:
                    S["state"] = fly.fade_in(cur); S["base_act"] = S["state"].nodes.activity.unsqueeze(1); acts[k] = None
                else:
                    S["state"], acts[k] = fly.step(S["state"], S["last"], cur)
                S["last"] = cur
            for a, (fe_vals, dstate, lg) in pipes.items():
                k = a if fe_vals is not None else "stock"
                dec.load_state_dict(dstate)
                if acts[k] is None:
                    fields[a].append(np.zeros((6 * 721, 2), np.float32)); continue
                flow = dec(acts[k]).mean(1) - dec(sims[k]["base_act"])[:, 0]
                fields[a].append(to_px(flow, sct, t.tensor(lg, device=fly.dev)).cpu().numpy())
    dec.load_state_dict(stock_dec_state)
    out = {a: np.stack(v) for a, v in fields.items()}
    out["stock_g1"] = out["stock"] / STAND_GAIN   # как в Д4 (кэш стенда без усиления)
    return out


def box_errors(fields, pts, dis, root, split, vid, nf):
    from scripts.evaluate_flyvis_qmatch_gate import propagate_box
    al = json.loads((root / vid / "alignment.json").read_text(encoding="utf-8"))["matches"]
    truth = json.loads((PROJECT_ROOT / "data/bdd100k/labels/box_track_20" / split / f"{vid}.json").read_text(encoding="utf-8"))
    boxes = []
    for m, fr in zip(al, truth):
        boxes.append((int(m["video_frame_index"]), {l["id"]: np.array([l["box2d"]["x1"], l["box2d"]["y1"], l["box2d"]["x2"], l["box2d"]["y2"]], np.float32)
                                                    for l in fr.get("labels") or [] if l.get("box2d") and l.get("category") in CATS}))
    srcs = {"zero": (pts, np.zeros_like(next(iter(fields.values())))), "dis": dis}
    srcs.update({a: (pts, v) for a, v in fields.items()})
    errs = {k: [] for k in srcs}
    for (r0, b0), (r1, b1) in zip(boxes[:-1], boxes[1:]):
        if r1 <= r0 or r1 >= nf:
            continue
        for tid, box in b0.items():
            if tid not in b1:
                continue
            gt_c = (b1[tid][:2] + b1[tid][2:]) / 2
            for k, (p, vec) in srcs.items():
                pred = propagate_box(box, p, vec, r0, r1)[0]
                errs[k].append(float(np.linalg.norm((pred[:2] + pred[2:]) / 2 - gt_c)))
    return {k: {"median": float(np.median(e)), "mean": float(np.mean(e)), "n": len(e)} for k, e in errs.items() if e}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--scale-steps", type=int, default=300)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--eval-limit", type=int, default=0)
    ap.add_argument("--train-videos", type=int, default=N_TRAIN_PER_COND)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    global OUT
    if a.out:
        OUT = PROJECT_ROOT / a.out
    import torch
    OUT.mkdir(parents=True, exist_ok=True)
    fps = json.loads((TC / "0001542f-5ce3cf52/motion/dis_quarter.json").read_text(encoding="utf-8"))["fps"]
    fly = Fly(fps)
    stock_dec = {k: v.detach().clone() for k, v in fly.base.decoder.state_dict().items()}

    if a.check:
        v = "b1c81faa-3df17267"
        ref = np.load(PROJECT_ROOT / "runs/video_rate" / v / "motion/flyvis_tiled.npz")
        import itertools
        fly_c = Fly(json.loads((PROJECT_ROOT / "runs/video_rate" / v / "motion/flyvis_tiled.json").read_text(encoding="utf-8"))["fps"])
        src = itertools.islice(read_gray(PROJECT_ROOT / "data/bdd100k/videos/val" / f"{v}.mov"), 120)
        frames = list(src)
        # прямой проход без записи файла
        t = fly_c.torch; dec = fly_c.base.decoder
        out = []
        with t.no_grad():
            state = last = base_act = None
            for f, g in enumerate(frames):
                cart = t.from_numpy(fly_c.tiles(g)).to(fly_c.dev, t.float32).unsqueeze(1) / 255.0
                cur = fly_c.render(cart)
                if state is None:
                    state = fly_c.fade_in(cur); base_act = state.nodes.activity.unsqueeze(1); out.append(np.zeros((4326, 2), np.float32))
                else:
                    state, act = fly_c.step(state, last, cur)
                    _, sc = fly_c.points_scale()
                    out.append(to_px(dec(act).mean(1) - dec(base_act)[:, 0], t.tensor(sc, device=fly_c.dev), t.zeros((), device=fly_c.dev)).cpu().numpy())
                last = cur
        out = np.stack(out); rv = ref["vectors"][:len(out)]
        pts, _ = fly_c.points_scale()
        m = np.abs(rv) > 0.05; print(f"[CHECK] отношение out/ref: медиана {np.median(out[m] / rv[m]):.3f}, разброс 5–95 % {np.quantile(out[m] / rv[m], [0.05, 0.95])}")
        d = np.abs(out - rv); print(f"[CHECK] max |Δ| {d.max():.4f} px, медиана |ref| {np.median(np.abs(rv)):.4f}, точки совпадают: {np.allclose(pts, ref['points'], atol=1e-3)}")
        return 0

    rng = np.random.default_rng(0)
    sel = json.loads((TC / "selection.json").read_text(encoding="utf-8"))["videos"]
    cond_tr = conditions(sel["train"], "train")
    tr = [v for v in sel["train"] if cond_tr.get(v) == "night"][:a.train_videos] + [v for v in sel["train"] if cond_tr.get(v) == "normal"][:a.train_videos]
    cond_dev = conditions(sel["dev"], "train")
    ev = [v for v in sel["dev"] if cond_dev.get(v) in ("night", "normal")]
    if a.eval_limit:
        ev = [v for v in ev if cond_dev[v] == "night"][:a.eval_limit] + [v for v in ev if cond_dev[v] == "normal"][:a.eval_limit]
    print(f"обучение: {len(tr)} видео (ночь {sum(cond_tr[v] == 'night' for v in tr)}), оценка dev: {len(ev)} (ночь {sum(cond_dev[v] == 'night' for v in ev)})", flush=True)
    meta = {"train": tr, "eval": ev, "cond": {**{v: cond_tr[v] for v in tr}, **{v: cond_dev[v] for v in ev}}, "steps": a.steps, "warm": WARM, "len": LEN}

    arms_path = OUT / "arms.pt"
    if a.eval_only and arms_path.exists():
        arms = torch.load(arms_path, weights_only=False)
    else:
        segs = build_train(fly, tr, rng)
        arms = {}
        with open(OUT / "train_log.txt", "w", encoding="utf-8") as log:
            for arm in ARMS:
                fly.base.decoder.load_state_dict(stock_dec)
                arms[arm] = train_arm(arm, fly, segs, a.scale_steps if arm == "scale" else a.steps, np.random.default_rng(1), log)
                torch.save(arms, arms_path)
        fly.base.decoder.load_state_dict(stock_dec)
        del segs

    res = {}
    pts, _ = fly.points_scale()
    started = time.time()
    for i, v in enumerate(ev):
        fields = run_video(fly, PROJECT_ROOT / "data/bdd100k/videos/train" / f"{v}.mov", arms, stock_dec)
        dq = np.load(TC / v / "motion/dis_quarter.npz")
        nf = min(len(dq["vectors"]), *(len(x) for x in fields.values()))
        res[v] = box_errors(fields, pts, (dq["points"], dq["vectors"]), TC, "train", v, nf)
        el = time.time() - started; eta = el / (i + 1) * (len(ev) - i - 1)
        print(f"[EVAL] {i + 1}/{len(ev)} {v} ({cond_dev[v]}): " + " | ".join(f"{k} {r['median']:.2f}" for k, r in res[v].items()) + f" | осталось ~{eta / 60:.1f} мин", flush=True)
        status(stage="eval", done=i + 1, total=len(ev), eta_min=eta / 60)
        (OUT / "per_video.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")

    groups = {"all": [v for v in ev if v in res], "night": [v for v in ev if v in res and cond_dev[v] == "night"], "day": [v for v in ev if v in res and cond_dev[v] == "normal"]}
    summary = {}
    brng = np.random.default_rng(1)
    for g, vs in groups.items():
        if not vs:
            continue
        keys = list(res[vs[0]])
        s = {k: float(np.mean([res[v][k]["median"] for v in vs])) for k in keys}
        s["videos"] = len(vs)
        for k in keys:
            if k == "stock":
                continue
            dlt = np.array([res[v][k]["median"] - res[v]["stock"]["median"] for v in vs])
            bs = dlt[brng.integers(0, len(dlt), (5000, len(dlt)))].mean(1)
            s[f"{k}-stock"] = {"mean": float(dlt.mean()), "ci95": [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]}
            s[f"reduction_{k}"] = 1 - s[k] / s["stock"]
        summary[g] = s
    arms_info = {k: {"log_gain": v["log_gain"], "frontend": v["frontend"], "final_loss": float(np.mean(v["loss_hist"][-50:]))} for k, v in arms.items()}
    (OUT / "summary.json").write_text(json.dumps({"summary": summary, "arms": arms_info, "meta": meta}, ensure_ascii=False, indent=1), encoding="utf-8")
    for g, s in summary.items():
        print(f"== {g} ({s['videos']} видео): " + " | ".join(f"{k} {s[k]:.2f}" for k in ("zero", "dis", "stock", *ARMS) if k in s)
              + " | снижение к штатной: " + ", ".join(f"{k} {s['reduction_' + k]:.1%}" for k in ARMS if "reduction_" + k in s))
    status(stage="done")
    print(f"[OK] {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
