"""Парное продолжение обучения D (карточка z13_core_continue_v1): D-train (ядро + считыватель) против D-frozen (только считыватель).

Оба варианта начинаются с одного состояния runs/z13_spatial_v2_20261013/D/full/checkpoint.pt (шаг 4000): веса ядра и
пространственного считывателя, нормировка считывателя, якоря притяжения, состояние Adam, генераторы случайных чисел.
D-train восстанавливает Adam целиком (4 группы: считыватель + 3 группы свободных параметров ядра, lr как К1 v2);
D-frozen — ядро заморожено, Adam только группы считывателя, её моменты переносятся из того же состояния (migrate_optimizer).
Пакеты: тот же выбор, что scripts/z13_spatial_train.py:train (генератор из контрольной точки, 16 примеров, мини-пакеты по 4),
для обоих вариантов одинаковый — хеш выбранных примеров пишется на каждом шаге и сверяется. Функция потерь, считыватель,
нормировка, вырезки, скорости обучения не меняются. Ступень 1 — 1000 дополнительных шагов (4000 → 5000) на вариант.
Оценка — один фиксированный набор train26 (эпизоды без возмущения, все помеченные кадры) до обучения и после каждого варианта:
доля полезных центров среди положительных (scripts/z13_train.success), ошибка в долях масштаба, NLL цели, вмешательства при
одинаковой доле отбора (1 / 2 / 5 % всех строк) с отдельным счётом неизвестных; парный бутстрэп по видео.
Запуск: python scripts/z13_continue.py --init | --audit | --technical-cpu | --all --deadline <ISO с поясом>
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RUN = ROOT / "runs/z13_continue_20261017"
SOURCE = ROOT / "runs/z13_spatial_v2_20261013/D/full"
SOURCE_PROTOCOL = ROOT / "runs/z13_spatial_v2_20261013/protocol.json"
VARIANTS = ("D-frozen", "D-train")
EXTRA_STEPS = 1000
COVERAGES = (0.01, 0.02, 0.05)
CODE = ["scripts/z13_continue.py", "scripts/z13_spatial_train.py", "scripts/z13_train.py", "tracking/z13_spatial.py",
        "tracking/z13_readout.py", "tracking/z13_checkpoint.py"]


def sha(path: Path) -> str:
    d = hashlib.sha256()
    with Path(path).open("rb") as s:
        for chunk in iter(lambda: s.read(1 << 20), b""):
            d.update(chunk)
    return d.hexdigest()


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


# ---------- общие части (тестируются в tests/test_z13_continue.py без FlyVis) ----------

def migrate_optimizer(state: dict, n_readout: int) -> dict:
    """Состояние Adam из 4 групп → только группа считывателя (первая), моменты и шаг сохраняются без изменений."""
    g0 = state["param_groups"][0]
    if len(g0["params"]) != n_readout:
        raise ValueError(f"Readout group has {len(g0['params'])} parameters, expected {n_readout}")
    ids = set(g0["params"])
    return {"state": {k: copy.deepcopy(v) for k, v in state["state"].items() if k in ids}, "param_groups": [copy.deepcopy(g0)]}


def pick_batch(rng, usable, batch):
    """Тот же выбор, что scripts/z13_spatial_train.py:train."""
    return rng.choice(usable, size=batch, replace=False)


def batch_hash(picked) -> str:
    return hashlib.sha256(np.asarray(picked, np.int64).tobytes()).hexdigest()[:16]


def train_steps(torch, variant, data, bb, ro, dev, opt, rng, start, stop, history, forward, on_save, control=None,
                batch=16, micro=4, save_every=10):
    """Шаги start → stop. D-frozen: ядро без градиента, штрафа и clamp; D-train — как прежний train()."""
    usable = [i for i, row in enumerate(data.labels) if row[2] >= 0]
    ro.train()
    for step in range(start, stop):
        if control is not None:
            control.check()
        picked = pick_batch(rng, usable, batch)
        obj = 0.
        opt.zero_grad(set_to_none=True)
        for left in range(0, batch, micro):
            labels = [data.labels[i] for i in picked[left:left + micro]]
            hx, masks = data.batch(torch, dev, [r[0] for r in labels], [r[1] for r in labels])
            logits = forward(torch, bb, ro, hx, masks)
            sel = logits[torch.arange(len(labels), device=dev), torch.as_tensor([r[1] for r in labels], device=dev)]
            target = torch.as_tensor([r[3] if r[2] == 1 else 721 for r in labels], device=dev)
            loss = torch.nn.functional.cross_entropy(sel, target)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("Non-finite localization loss")
            (loss * len(labels) / batch).backward()
            obj += float(loss.detach()) * len(labels) / batch
        core_grad = 0.
        pen = torch.zeros((), device=dev)
        if variant == "D-train":
            core_grad = float(sum(p.grad.detach().square().sum() for p in bb.params if p.grad is not None) ** 0.5)
            pen = bb.penalty(); pen.backward()
        elif any(p.grad is not None for p in bb.params):
            raise RuntimeError("Frozen core received gradients")
        if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for g in opt.param_groups for p in g["params"]):
            raise RuntimeError("Non-finite gradients")
        opt.step()
        if variant == "D-train":
            bb.clamp()
        history.append({"it": step + 1, "object": obj, "penalty": float(pen.detach()), "core_object_gradient": core_grad, "batch": batch_hash(picked)})
        if (step + 1) % save_every == 0 or step + 1 == stop or (control is not None and control.stopped()):
            on_save(step + 1)
    return history


def core_fingerprint(params) -> str:
    d = hashlib.sha256()
    for p in params:
        d.update(p.detach().cpu().numpy().tobytes())
    return d.hexdigest()


def coverage_table(rows, coverages=COVERAGES):
    """Одинаковый знаменатель для всех вариантов: все строки набора; отбор — верхние ceil(q·N) по p_present."""
    n = len(rows); order = sorted(range(n), key=lambda i: -rows[i]["p_present"])
    out = {}
    for q in coverages:
        k = max(1, math.ceil(q * n)); sel = [rows[i] for i in order[:k]]
        known = [r for r in sel if r["kind"] >= 0]; useful = sum(bool(r.get("useful")) for r in known)
        out[f"{q:.2f}"] = {"selected": k, "known": len(known), "unknown": k - len(known), "useful": useful, "harmful": len(known) - useful,
                           "useful_share_known": useful / len(known) if known else None}
    return out


def per_video_useful(rows):
    v = {}
    for r in rows:
        if r["kind"] == 1:
            a = v.setdefault(r["video"], [0, 0]); a[0] += bool(r["useful"]); a[1] += 1
    return {k: a[0] / a[1] for k, a in v.items()}


def paired(a: dict, b: dict, n_boot=5000, seed=1):
    vids = sorted(set(a) & set(b)); d = np.array([a[x] - b[x] for x in vids])
    bs = d[np.random.default_rng(seed).integers(0, len(d), (n_boot, len(d)))].mean(1)
    return {"mean_per_video": float(d.mean()), "ci95": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))], "better": int((d > 0).sum()), "n": len(d)}


def summarize(rows):
    pos = [r for r in rows if r["kind"] == 1]; lab = [r for r in rows if r["kind"] >= 0]
    return {"rows": len(rows), "positives": len(pos), "labelled": len(lab),
            "useful_fraction_pooled": float(np.mean([r["useful"] for r in pos])) if pos else None,
            "useful_fraction_mean_video": float(np.mean(list(per_video_useful(rows).values()))) if pos else None,
            "median_error_rel": float(np.median([r["err"] / r["scale"] for r in pos])) if pos else None,
            "nll_labelled": float(np.mean([r["nll"] for r in lab])) if lab else None, "coverage": coverage_table(rows)}


# ---------- части с FlyVis ----------

def evaluate(torch, data, bb, ro, dev, geom, eye_size, control=None):
    """Как scripts/z13_spatial_train.py:predict (эпизоды без возмущения), плюс NLL цели и видео строки."""
    from scripts.z13_train import success
    from tracking.z13_readout import center_estimate, run_sequence, to_frame
    vid_of = []
    for v, n in data.meta:
        vid_of += [v] * n
    by_ep = {}
    for i, lab in enumerate(data.labels):
        if not lab[8]:
            by_ep.setdefault(lab[0], []).append(i)
    eps = sorted(by_ep); rows = []
    ro.eval()
    with torch.no_grad():
        for left in range(0, len(eps), 4):
            if control is not None:
                control.check()
            ch = eps[left:left + 4]
            hx, ms = data.batch(torch, dev, ch, [max(data.labels[i][1] for i in by_ep[e]) for e in ch])
            probs = run_sequence(torch, bb, ro, hx, ms).softmax(-1).cpu().numpy()
            for j, e in enumerate(ch):
                for idx in by_ep[e]:
                    lab = data.labels[idx]; p = probs[j, lab[1]]
                    center = np.asarray(to_frame(center_estimate(p[:721], geom), lab[7], eye_size))
                    tgt = lab[3] if lab[2] == 1 else 721
                    r = {"label": idx, "video": vid_of[e], "kind": int(lab[2]), "p_present": 1 - float(p[721]), "center": center.tolist(),
                         "pred": np.asarray(lab[6], float).tolist(), "nll": float(-np.log(max(p[tgt], 1e-30))) if lab[2] >= 0 else None}
                    if lab[2] == 1:
                        box = np.asarray(lab[5]); t = np.asarray(lab[4])
                        r.update(err=float(np.linalg.norm(center - t)), err_pred=float(np.linalg.norm(np.asarray(lab[6]) - t)),
                                 scale=float(max(box[2] - box[0], box[3] - box[1])))
                        r["useful"] = bool(success(r))
                    elif lab[2] == 0:
                        r["useful"] = False
                    rows.append(r)
            if control is not None and left % 200 == 0:
                control.status(stage="evaluation", episodes_done=left + len(ch), episodes_total=len(eps))
    return rows


def load_source(torch):
    ck = torch.load(SOURCE / "checkpoint.pt", map_location="cpu", weights_only=False)
    return ck


def build_variant(torch, variant, ck):
    import scripts.z13_spatial_train as SPT
    est, bank, geom, bb, ro, dev = SPT.build("D", torch)   # FlyBackbone(trainable=True) + SpatialReadout(34, ...)
    ro.load_state_dict(ck["readout"]); bb.net.load_state_dict(ck["backbone"])
    if len(bb.p0) != len(ck["anchors"]):
        raise ValueError("Anchor count mismatch")
    for dst, val in zip(bb.p0, ck["anchors"]):
        dst.copy_(val.to(dst.device))
    n_ro = len(list(ro.parameters()))
    if variant == "D-train":
        opt = torch.optim.Adam([{"params": list(ro.parameters()), "lr": .003}] + bb.param_groups())
        opt.load_state_dict(ck["optimizer"])
    else:
        for p in bb.params:
            p.requires_grad_(False)
        opt = torch.optim.Adam([{"params": list(ro.parameters()), "lr": .003}])
        opt.load_state_dict(migrate_optimizer(ck["optimizer"], n_ro))
    rng = np.random.default_rng(0); rng.bit_generator.state = copy.deepcopy(ck["generator"])
    random.setstate(ck["rng_python"]); np.random.set_state(ck["rng_numpy"]); torch.set_rng_state(ck["rng_torch"])
    return est, bank, geom, bb, ro, dev, opt, rng


def save_state(torch, path, variant, bb, ro, opt, rng, step, history, ident):
    from tracking.z13_checkpoint import save_checkpoint
    save_checkpoint(torch, path, "D", bb, ro, opt, rng, step, history, {**ident, "variant": variant})


def restore_state(torch, path, variant, bb, ro, opt, rng, ident):
    from tracking.z13_checkpoint import restore_checkpoint
    return restore_checkpoint(torch, path, "D", bb, ro, opt, rng, {**ident, "variant": variant})


def audit(torch) -> dict:
    """Согласованность исходного состояния D: шаги, история, модель, идентичность, оптимизатор."""
    ck = load_source(torch); st = json.loads((SOURCE / "status.json").read_text(encoding="utf-8"))
    loss = json.loads((SOURCE / "loss.json").read_text(encoding="utf-8")); ident = json.loads((SOURCE / "identity.json").read_text(encoding="utf-8"))
    model = torch.load(SOURCE / "model.pt", map_location="cpu", weights_only=False)
    proto = json.loads(SOURCE_PROTOCOL.read_text(encoding="utf-8"))
    eq = lambda a, b: a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
    g = ck["optimizer"]["param_groups"]
    rep = {"source": str(SOURCE.relative_to(ROOT)), "files_sha256": {p.name: sha(p) for p in sorted(SOURCE.iterdir()) if p.is_file()},
           "checkpoint_step": ck["step"], "status_step": st["step"], "status_state": st["state"], "protocol_steps": proto["steps"],
           "history_len": len(ck["history"]), "loss_json_len": len(loss), "history_equals_loss_json": ck["history"] == loss,
           "identity_equals_identity_json": ck["identity"] == ident, "model_readout_equals_checkpoint": eq(model["readout"], ck["readout"]),
           "model_backbone_equals_checkpoint": eq(model["backbone"], ck["backbone"]), "optimizer_groups": len(g),
           "optimizer_group_sizes": [len(x["params"]) for x in g], "optimizer_lrs": [x["lr"] for x in g], "anchors": len(ck["anchors"]),
           "torch_version_checkpoint": ck["torch_version"], "torch_version_now": str(torch.__version__), "fit_videos": len(ident["config"]["job"]["fit"]),
           "predict_videos": len(ident["config"]["job"]["predict"]),
           "code_changed_since_source": {k: (sha(ROOT / k) != v) for k, v in ident["files_sha256"].items() if k.endswith(".py") and (ROOT / k).exists()}}
    rep["passed"] = bool(rep["checkpoint_step"] == rep["status_step"] == rep["protocol_steps"] == rep["history_len"] == rep["loss_json_len"]
                         and rep["history_equals_loss_json"] and rep["identity_equals_identity_json"] and rep["model_readout_equals_checkpoint"]
                         and rep["model_backbone_equals_checkpoint"] and rep["optimizer_groups"] == 4 and rep["torch_version_checkpoint"] == rep["torch_version_now"])
    return rep


def identity(variant=None) -> dict:
    src = json.loads((SOURCE / "identity.json").read_text(encoding="utf-8"))
    return {"source_checkpoint_sha256": sha(SOURCE / "checkpoint.pt"), "source_identity_sha256": sha(SOURCE / "identity.json"),
            "code_sha256": {k: sha(ROOT / k) for k in CODE}, "fit": src["config"]["job"]["fit"], "predict": src["config"]["job"]["predict"],
            "extra_steps": EXTRA_STEPS}


class Console:
    """Состояние для консоли PowerShell: runs/z13_continue_20261017/status.json."""

    def __init__(self, run, deadline):
        self.path = run / "status.json"; self.t0 = time.monotonic(); self.v = {"experiment": "z13_core_continue_v1", "deadline": deadline, "started": now()}

    def __call__(self, **kw):
        from tracking.z13_checkpoint import atomic_json
        self.v.update(kw); self.v["elapsed_s"] = round(time.monotonic() - self.t0); self.v["updated"] = now()
        atomic_json(self.path, self.v)


def run_all(torch, deadline: str, run: Path = RUN):
    from scripts.z13_train import Episodes
    from tracking.z13_checkpoint import RunControl, atomic_json
    run.mkdir(parents=True, exist_ok=True)
    proto = json.loads((run / "protocol.json").read_text(encoding="utf-8"))
    if proto["code_sha256"] != {k: sha(ROOT / k) for k in CODE}:
        raise ValueError("Code changed after protocol was written")
    console = Console(run, deadline); ident = identity()
    done = lambda name: (run / name / "done.json").exists()
    console(state="running", stage="audit")
    a = audit(torch); atomic_json(run / "audit.json", a)
    if not a["passed"]:
        raise RuntimeError("Source audit failed")
    ck = load_source(torch)
    evaluation = None
    if not done("baseline"):   # исходная оценка + сверка с сохранёнными предсказаниями D
        console(stage="baseline evaluation", variant="D (step 4000)")
        est, bank, geom, bb, ro, dev, opt, rng = build_variant(torch, "D-frozen", ck)
        evaluation = Episodes(ident["predict"])
        ctl = RunControl(run / "baseline", run / "stop.request", deadline=deadline)
        rows = evaluate(torch, evaluation, bb, ro, dev, geom, bank.eye_size, ctl)
        saved = {r["label"]: r for r in json.loads((SOURCE / "predictions.json").read_text(encoding="utf-8"))}
        diff_p = max(abs(r["p_present"] - saved[r["label"]]["p_present"]) for r in rows)
        diff_c = max(float(np.abs(np.asarray(r["center"]) - np.asarray(saved[r["label"]]["center"])).max()) for r in rows)
        repro = {"rows": len(rows), "saved_rows": len(saved), "max_abs_p_present": diff_p, "max_abs_center_px": diff_c,
                 "passed": bool(len(rows) == len(saved) and diff_p < 1e-3 and diff_c < 0.5)}
        atomic_json(run / "baseline/reproduction.json", repro)
        if not repro["passed"]:
            raise RuntimeError(f"Saved D predictions not reproduced: {repro}")
        atomic_json(run / "baseline/rows.json", rows); atomic_json(run / "baseline/summary.json", summarize(rows))
        atomic_json(run / "baseline/done.json", {"finished": now()})
        del est, bank, bb, ro, opt; torch.cuda.empty_cache() if torch.cuda.is_available() else None
    from tracking.z13_readout import run_sequence
    for k, variant in enumerate(proto["order"]):
        folder = run / variant
        if done(variant):
            continue
        console(stage="training", variant=variant, variant_index=f"{k + 1}/{len(proto['order'])}", extra_step=0, extra_total=EXTRA_STEPS)
        est, bank, geom, bb, ro, dev, opt, rng = build_variant(torch, variant, ck)
        core_before = core_fingerprint(bb.params)
        ctl = RunControl(folder, run / "stop.request", deadline=deadline)
        cpath = folder / "checkpoint.pt"; folder.mkdir(parents=True, exist_ok=True)
        start, history = ck["step"], []
        if cpath.exists():
            start, history = restore_state(torch, cpath, variant, bb, ro, opt, rng, ident)
        fit = Episodes(ident["fit"]); t0 = time.monotonic(); s0 = start

        def on_save(step):
            save_state(torch, cpath, variant, bb, ro, opt, rng, step, history, ident)
            el = time.monotonic() - t0; rate = el / max(step - s0, 1)
            console(extra_step=step - ck["step"], seconds_per_step=round(rate, 3), eta_variant_s=round(rate * (ck["step"] + EXTRA_STEPS - step)),
                    last_saved=now(), last_loss=history[-1]["object"])
            ctl.status(state="running", step=step, total=ck["step"] + EXTRA_STEPS)

        train_steps(torch, variant, fit, bb, ro, dev, opt, rng, start, ck["step"] + EXTRA_STEPS, history, run_sequence, on_save, ctl)
        del fit
        core_after = core_fingerprint(bb.params)
        drift = float(np.median(np.concatenate([((p.detach() - torch.as_tensor(ck["backbone"][n], device=p.device)).abs() /
                                                 (torch.as_tensor(ck["backbone"][n], device=p.device).abs() + 1e-6)).flatten().cpu().numpy()
                                                for n, p in bb.net.named_parameters() if n in ("nodes_bias", "nodes_time_const", "edges_syn_strength")])))
        console(stage="evaluation", variant=variant)
        evaluation = evaluation or Episodes(ident["predict"])
        rows = evaluate(torch, evaluation, bb, ro, dev, geom, bank.eye_size, ctl)
        atomic_json(folder / "rows.json", rows); atomic_json(folder / "summary.json", summarize(rows)); atomic_json(folder / "loss.json", history)
        atomic_json(folder / "done.json", {"finished": now(), "core_unchanged": core_before == core_after, "core_drift_median_rel_vs_step4000": drift,
                                           "final_step": ck["step"] + EXTRA_STEPS})
        del est, bank, bb, ro, opt; torch.cuda.empty_cache() if torch.cuda.is_available() else None
    compare(run); console(state="completed", stage="report written")


def compare(run: Path = RUN) -> dict:
    from tracking.z13_checkpoint import atomic_json
    rows = {n: json.loads((run / n / "rows.json").read_text(encoding="utf-8")) for n in ("baseline",) + VARIANTS}
    keys = [tuple(r["label"] for r in rows[n]) for n in rows]
    if len(set(keys)) != 1:
        raise ValueError("Evaluation rows differ between variants")
    losses = {v: json.loads((run / v / "loss.json").read_text(encoding="utf-8")) for v in VARIANTS}
    same_batches = [a["batch"] for a in losses["D-frozen"]] == [b["batch"] for b in losses["D-train"]]
    dn = {v: json.loads((run / v / "done.json").read_text(encoding="utf-8")) for v in VARIANTS}
    pv = {n: per_video_useful(rows[n]) for n in rows}
    rep = {"summary": {n: summarize(rows[n]) for n in rows}, "identical_batches": same_batches,
           "frozen_core_unchanged": dn["D-frozen"]["core_unchanged"], "trained_core_changed": not dn["D-train"]["core_unchanged"],
           "trained_core_drift_median_rel": dn["D-train"]["core_drift_median_rel_vs_step4000"],
           "train_loss_last100": {v: float(np.mean([h["object"] for h in losses[v][-100:]])) for v in VARIANTS},
           "paired_useful": {"D-train-D-frozen": paired(pv["D-train"], pv["D-frozen"]), "D-train-baseline": paired(pv["D-train"], pv["baseline"]),
                             "D-frozen-baseline": paired(pv["D-frozen"], pv["baseline"])},
           "scope": "Поисковый результат на многократно использованном train26; не HOTA; интервал с нулём не доказывает равенства"}
    rep["execution_valid"] = bool(same_batches and rep["frozen_core_unchanged"] and rep["trained_core_changed"])
    atomic_json(run / "report.json", rep)
    return rep


def technical_cpu(torch, n_videos=2, steps=2) -> dict:
    """CPU-проверка на реальной D: перенос состояния, равные пакеты, неизменность / изменение ядра, точное восстановление."""
    from scripts.z13_train import Episodes
    from tracking.z13_readout import run_sequence
    import tempfile
    ck = load_source(torch); ident = identity(); fit = Episodes(ident["fit"][:n_videos]); rep = {}
    hist_batches = {}
    for variant in VARIANTS:
        est, bank, geom, bb, ro, dev, opt, rng = build_variant(torch, variant, ck)
        before = core_fingerprint(bb.params); ro0 = copy.deepcopy(ro.state_dict())
        if variant == "D-frozen":
            m = opt.state_dict()["state"]; src = ck["optimizer"]["state"]
            rep["frozen_moments_carried"] = all(torch.equal(m[i]["exp_avg"], src[i]["exp_avg"]) and torch.equal(m[i]["exp_avg_sq"], src[i]["exp_avg_sq"]) for i in m)
        with tempfile.TemporaryDirectory() as tmp:
            cpath = Path(tmp) / "ck.pt"; history = []
            save = lambda s: save_state(torch, cpath, variant, bb, ro, opt, rng, s, history, ident)
            train_steps(torch, variant, fit, bb, ro, dev, opt, rng, ck["step"], ck["step"] + steps, history, run_sequence, save, save_every=1)
            after = core_fingerprint(bb.params); snap = (copy.deepcopy(ro.state_dict()), copy.deepcopy(bb.net.state_dict()), copy.deepcopy(opt.state_dict()), copy.deepcopy(rng.bit_generator.state))
            with torch.no_grad():   # порча и восстановление
                next(ro.parameters()).add_(1.0)
                bb.params[0].add_(1.0)
            restore_state(torch, cpath, variant, bb, ro, opt, rng, ident)
            eq = lambda a, b: a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
            exact = eq(ro.state_dict(), snap[0]) and eq(bb.net.state_dict(), snap[1]) and rng.bit_generator.state == snap[3]
        hist_batches[variant] = [h["batch"] for h in history]
        rep[variant] = {"core_unchanged": before == after, "readout_changed": not eq(ro.state_dict(), ro0), "restore_exact": exact,
                        "loss": [h["object"] for h in history], "core_gradient": [h["core_object_gradient"] for h in history]}
        del est, bank, bb, ro, opt
    rep["identical_batches"] = hist_batches["D-frozen"] == hist_batches["D-train"]
    rep["passed"] = bool(rep["identical_batches"] and rep["frozen_moments_carried"] and rep["D-frozen"]["core_unchanged"] and not rep["D-train"]["core_unchanged"]
                         and rep["D-frozen"]["readout_changed"] and rep["D-train"]["readout_changed"] and rep["D-frozen"]["restore_exact"] and rep["D-train"]["restore_exact"]
                         and all(g > 0 for g in rep["D-train"]["core_gradient"]))
    return rep


def write_protocol(run: Path = RUN):
    from tracking.z13_checkpoint import atomic_json
    run.mkdir(parents=True, exist_ok=True)
    p = run / "protocol.json"
    if p.exists():
        raise SystemExit("protocol.json уже существует — не переписывается")
    proto = {"id": "z13_core_continue_v1", "written": now(), "source": str(SOURCE.relative_to(ROOT)), "source_checkpoint_sha256": sha(SOURCE / "checkpoint.pt"),
             "variants": list(VARIANTS), "order": ["D-frozen", "D-train"], "extra_steps": EXTRA_STEPS, "batch": 16, "micro_batch": 4, "readout_lr": 0.003,
             "core_lr": {"nodes_bias": 1e-4, "nodes_time_const": 5e-6, "edges_syn_strength": 5e-6}, "anchor": 1.0, "save_every_steps": 10,
             "evaluation": "train26 (identity.json job.predict), эпизоды без возмущения, все помеченные кадры; до обучения и после каждого варианта",
             "main_measure": "доля полезных центров среди положительных (scripts/z13_train.success), парный бутстрэп по видео 5000",
             "secondary": ["median err/scale", "NLL цели на помеченных", "вмешательства при доле отбора 1/2/5 % всех строк: known/unknown/useful/harmful", "train loss last 100"],
             "stop": "нечисла в потере или градиентах, градиент в замороженном ядре, расхождение исходных предсказаний (|Δp| ≥ 1e-3 или |Δцентр| ≥ 0,5 px), stop.request, срок",
             "no_repeat": "done.json этапа — этап не повторяется; обучение продолжается с checkpoint.pt варианта",
             "code_sha256": {k: sha(ROOT / k) for k in CODE}}
    atomic_json(p, proto)
    return proto


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", action="store_true"); ap.add_argument("--audit", action="store_true"); ap.add_argument("--technical-cpu", action="store_true")
    ap.add_argument("--all", action="store_true"); ap.add_argument("--compare", action="store_true"); ap.add_argument("--deadline")
    a = ap.parse_args()
    if a.technical_cpu or a.audit or a.init:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        import torch
        torch.cuda.is_available = lambda: False
    import torch
    torch.set_num_threads(2)
    from tracking.z13_checkpoint import atomic_json
    if a.init:
        print(json.dumps(write_protocol(), ensure_ascii=False)[:300]); return
    if a.audit:
        r = audit(torch); atomic_json(RUN / "audit.json", r); print(json.dumps(r, ensure_ascii=False, indent=1)); return
    if a.technical_cpu:
        r = technical_cpu(torch); atomic_json(RUN / "technical_cpu.json", r); print(json.dumps(r, ensure_ascii=False)); return
    if a.compare:
        print(json.dumps(compare(), ensure_ascii=False)[:2000]); return
    if a.all:
        if not a.deadline or datetime.fromisoformat(a.deadline).tzinfo is None:
            ap.error("--deadline с часовым поясом обязателен")
        from tracking.gpu_lock import GpuLock
        with GpuLock("flyvis_fovea_gpu"):
            run_all(torch, a.deadline)


if __name__ == "__main__":
    from tracking.z13_checkpoint import TrainingPaused
    try:
        main()
    except TrainingPaused as exc:
        print(str(exc), flush=True); raise SystemExit(75)
