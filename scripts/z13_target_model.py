"""Проверка «муха против входа глаза» для онлайн-модели цели (карточка z13_target_model_probe_v1; по DiMP, без обучения ядра).

Узел: для каждого эпизода — линейная модель цели в замкнутом виде (гребневая регрессия), подогнанная только по наблюдениям
трекера на последнем кадре образца: цель 1 — колонки внутри рамки трека (ex_mask), 0 — остальные колонки вырезки (включая
соседей). Применение к кадру поиска: балл колонки = модель(признаки колонки); центр — колонка-максимум и её 6 соседей
(веса softmax(балл / ст. откл. баллов) только по этим 7 колонкам); оценка присутствия — высота пика в ст. откл. баллов кадра.
Разметка будущего кадра — только для оценки (полезный центр: scripts/z13_train.success).
Признаки (одинаковое правило: признак колонки + среднее по 6 соседям, стандартизация по train-подвыборке, одна сетка λ):
  R0 — вход глаза с той же историей эпизода: текущий кадр, разность с прошлым, скользящие средние τ = 2 и 8 кадров (8 чисел);
  C0 — FlyVis с исходной инициализацией конфигурации (до обучения на Sintel), заморожена;  C — штатная FlyVis;
  D — сохранённая обученная D (runs/z13_spatial_v2_20261013/D/full/model.pt); у мухи 34 выходных типа (68 чисел).
Выбор λ — на 20 видео train104 (одна сетка, максимум доли полезных центров); оценка — те же строки train26, что
runs/z13_continue_20261017/baseline/rows.json (прежний считыватель D) и runs/z13_spatial_v2_20261013/C/full/predictions.json (C).
Контроль прогноза Калмана: центр = прогноз (полезным по определению не бывает) + доля центров нового узла у центра вырезки.
Выход — runs/z13_target_model_20261017/report.json. Запуск: python scripts/z13_target_model.py [--cpu --smoke]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RUN = ROOT / "runs/z13_target_model_20261017"
ARMS = ("R0", "C0", "C", "D")
LAMBDAS = (1e-3, 1e-2, 1e-1, 1.0, 10.0)
LAMBDA_VIDEOS = 20
MIN_SHIFT = 0.02


def neighbor_aug(f, nb):
    """(B, d, 721) → (B, 2d, 721): признак колонки и среднее по её 6 соседям."""
    return np.concatenate([f, f[:, :, nb[:, 1:]].mean(-1)], 1)


def receptor_features(hexes):
    """(B, T, 721) → список по кадрам (B, 4, 721): кадр, разность с прошлым, EMA τ=2 и τ=8 по всей истории эпизода."""
    out = []; e2 = hexes[:, 0].copy(); e8 = hexes[:, 0].copy(); prev = hexes[:, 0]
    for t in range(hexes.shape[1]):
        h = hexes[:, t]; e2 += (h - e2) / 2.0; e8 += (h - e8) / 8.0
        out.append(np.stack([h, h - prev, e2.copy(), e8.copy()], 1)); prev = h
    return out


def fit_predict(x_ex, mask, x_search, lambdas):
    """Гребневая модель цели по одному кадру образца. x_ex (721, d), mask (721,), x_search (S, 721, d) → баллы (L, S, 721)."""
    mu = x_ex.mean(0); xc = x_ex - mu; y = mask.astype(np.float64); yc = y - y.mean()
    g = xc.T @ xc / len(xc); b = xc.T @ yc / len(xc); out = []
    for lam in lambdas:
        w = np.linalg.solve(g + lam * np.eye(len(g)), b)
        out.append((x_search - mu) @ w + y.mean())
    return np.stack(out)


def center_and_presence(score, geom):
    k = int(np.argmax(score)); nb = np.unique(geom.neighbors[k]); s = score[nb]; sd = float(score.std()) + 1e-9
    w = np.exp((s - s.max()) / sd); c = (geom.points[nb] * w[:, None]).sum(0) / w.sum()
    return c, float((score.max() - score.mean()) / sd)


def build_arm(arm, torch, ckD):
    from scripts.run_bdd_video_rate_experiment import make_motion_estimator
    from tracking.foveated_eyes import EyeBank
    from tracking.z13_readout import FlyBackbone, HexGeometry
    est = make_motion_estimator("flyvis_tiled", 30.0, 0.02)
    bank = EyeBank(est); geom = HexGeometry(bank.points_local, bank.eye_size)
    if arm == "R0":
        return bank, geom, None
    if arm == "C0":
        est.network = est.flyvis.NetworkView("flow/0000/000").network(checkpoint="best").init(eval=True)
        est.network.requires_grad_(False)
    if arm == "D":
        est.network.load_state_dict({k: v.to(est.flyvis.device) for k, v in ckD["backbone"].items()})
    return bank, geom, FlyBackbone(est, trainable=False)


def episode_features(torch, arm, bb, hexes_t, nb, frames_needed):
    """Признаки по кадрам эпизодов пакета: dict кадр → (B, d, 721) numpy (только нужные кадры)."""
    from tracking.z13_readout import N_EXEMPLAR
    T = hexes_t.shape[1]; out = {}
    if arm == "R0":
        rf = receptor_features(hexes_t.cpu().numpy().astype(np.float64))
        return {t: neighbor_aug(rf[t], nb) for t in frames_needed}
    with torch.no_grad():
        st = bb.init(hexes_t[:, 0:1]); last = hexes_t[:, 0]
        for t in range(1, T):
            st, f = bb.step(st, last, hexes_t[:, t]); last = hexes_t[:, t]
            if t in frames_needed:
                out[t] = neighbor_aug(f.double().cpu().numpy(), nb)
    if (N_EXEMPLAR - 1) == 0:
        raise ValueError("exemplar frame must be stepped")
    return out


def run_set(torch, arm, bb, geom, eye_size, data, lambdas, stats, status, label=""):
    """Строки оценки по всем помеченным кадрам эпизодов без возмущения; на каждую строку — результат для каждого λ."""
    from scripts.z13_train import success
    from tracking.z13_readout import N_EXEMPLAR, to_frame
    vid_of = []
    for v, n in data.meta:
        vid_of += [v] * n
    by_ep = {}
    for i, lab in enumerate(data.labels):
        if not lab[8]:
            by_ep.setdefault(lab[0], []).append(i)
    eps = sorted(by_ep); rows = []; dev = bb.dev if bb is not None else "cpu"
    for left in range(0, len(eps), 8):
        ch = eps[left:left + 8]; mx = [max(data.labels[i][1] for i in by_ep[e]) for e in ch]
        hx, ms = data.batch(torch, dev, ch, mx)
        need = {N_EXEMPLAR - 1} | {N_EXEMPLAR + data.labels[i][1] for e in ch for i in by_ep[e]}
        F = episode_features(torch, arm, bb, hx, geom.neighbors, need)
        mask = ms.cpu().numpy()
        for j, e in enumerate(ch):
            xe = (F[N_EXEMPLAR - 1][j].T - stats[0]) / stats[1]
            idx = by_ep[e]; xs = np.stack([(F[N_EXEMPLAR + data.labels[i][1]][j].T - stats[0]) / stats[1] for i in idx])
            sc = fit_predict(xe, mask[j], xs, lambdas)
            for s_i, i in enumerate(idx):
                lab = data.labels[i]; r = {"label": i, "video": vid_of[e], "kind": int(lab[2]), "pos": int(lab[1]), "by_lambda": []}
                for l_i in range(len(lambdas)):
                    c_eye, pres = center_and_presence(sc[l_i, s_i], geom)
                    c = np.asarray(to_frame(c_eye, lab[7], eye_size)); x = {"p_present": pres, "center": c.tolist(), "pred": np.asarray(lab[6], float).tolist(),
                                                                         "near_crop_center": bool(np.linalg.norm(c_eye - eye_size / 2) < geom.spacing)}
                    if lab[2] == 1:
                        box = np.asarray(lab[5]); t_ = np.asarray(lab[4])
                        x.update(kind=1, err=float(np.linalg.norm(c - t_)), err_pred=float(np.linalg.norm(np.asarray(lab[6]) - t_)), scale=float(max(box[2] - box[0], box[3] - box[1])))
                        x["useful"] = bool(success(x))
                    r["by_lambda"].append(x)
                rows.append(r)
        status(stage=f"{arm}: {label}", episodes_done=min(left + 8, len(eps)), episodes_total=len(eps))
    return rows


def feature_stats(torch, arm, bb, geom, data, n_eps=200):
    """Среднее и ст. откл. признаков по кадрам образца train-подвыборки (до выбора λ и оценки)."""
    from tracking.z13_readout import N_EXEMPLAR
    rng = np.random.default_rng(0); eps = sorted(rng.choice(len(data.off) - 1, size=min(n_eps, len(data.off) - 1), replace=False).tolist())
    dev = bb.dev if bb is not None else "cpu"; xs = []
    for left in range(0, len(eps), 8):
        hx, _ = data.batch(torch, dev, eps[left:left + 8], [0] * len(eps[left:left + 8]))
        F = episode_features(torch, arm, bb, hx, geom.neighbors, {N_EXEMPLAR - 1})
        xs.append(F[N_EXEMPLAR - 1].transpose(0, 2, 1).reshape(-1, F[N_EXEMPLAR - 1].shape[1]))
    X = np.concatenate(xs)
    return X.mean(0), X.std(0) + 1e-6


def useful_rows(rows, l_i):
    return [{"video": r["video"], "kind": r["kind"], "label": r["label"], "pos": r["pos"], **r["by_lambda"][l_i]} for r in rows]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cpu", action="store_true"); ap.add_argument("--smoke", action="store_true"); a = ap.parse_args()
    if a.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        import torch
        torch.cuda.is_available = lambda: False
    import torch
    torch.set_num_threads(4)
    from scripts.z13_continue import Console, paired, per_video_useful, summarize
    from scripts.z13_train import Episodes
    from tracking.z13_checkpoint import atomic_json
    run = RUN / ("smoke" if a.smoke else ""); run.mkdir(parents=True, exist_ok=True)
    src = json.loads((ROOT / "runs/z13_spatial_v2_20261013/D/full/identity.json").read_text(encoding="utf-8"))["config"]["job"]
    fit_v, eval_v = src["fit"][:LAMBDA_VIDEOS], src["predict"]
    if a.smoke:
        fit_v, eval_v = fit_v[:1], eval_v[:1]
    console = Console(run, "нет (короткая проверка; останов — stop.request)")
    ckD = torch.load(ROOT / "runs/z13_spatial_v2_20261013/D/full/model.pt", map_location="cpu", weights_only=False)
    fit, ev = Episodes(fit_v), Episodes(eval_v)
    res = {}
    for k, arm in enumerate(ARMS):
        done = run / arm / "rows.json"
        if done.exists():
            res[arm] = json.loads((run / arm / "choice.json").read_text(encoding="utf-8")); continue
        console(state="running", variant=arm, variant_index=f"{k + 1}/{len(ARMS)}", stage="build")
        t0 = time.monotonic(); bank, geom, bb = build_arm(arm, torch, ckD)
        stats = feature_stats(torch, arm, bb, geom, fit)
        frows = run_set(torch, arm, bb, geom, bank.eye_size, fit, LAMBDAS, stats, console, "выбор λ (train104, 20 видео)")
        fr = [np.mean([x["by_lambda"][l]["useful"] for x in frows if x["kind"] == 1]) for l in range(len(LAMBDAS))]
        l_best = int(np.argmax(fr))
        erows = run_set(torch, arm, bb, geom, bank.eye_size, ev, LAMBDAS[l_best:l_best + 1], stats, console, "оценка train26")
        choice = {"lambda": LAMBDAS[l_best], "fit_useful_by_lambda": dict(zip(map(str, LAMBDAS), map(float, fr))), "seconds": time.monotonic() - t0}
        atomic_json(run / arm / "choice.json", choice); atomic_json(run / arm / "rows.json", useful_rows(erows, 0)); res[arm] = choice
        del bb, bank
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    rows = {arm: json.loads((run / arm / "rows.json").read_text(encoding="utf-8")) for arm in ARMS}
    labels = {tuple(sorted(r["label"] for r in rows[arm])) for arm in ARMS}
    if len(labels) != 1:
        raise ValueError("Different evaluation rows between arms")
    prior = {}
    if not a.smoke:
        b = json.loads((ROOT / "runs/z13_continue_20261017/baseline/rows.json").read_text(encoding="utf-8"))
        prior["D_prior"] = b
        cp = {r["label"]: r for r in json.loads((ROOT / "runs/z13_spatial_v2_20261013/C/full/predictions.json").read_text(encoding="utf-8"))}
        from scripts.z13_train import success
        vid = {r["label"]: r["video"] for r in b}
        prior["C_prior"] = [{"label": l, "video": vid[l], "kind": cp[l]["kind"], "p_present": cp[l]["p_present"], **({"useful": bool(success(cp[l])), "err": cp[l]["err"], "scale": cp[l]["scale"]} if cp[l]["kind"] == 1 else {"useful": False})}
                            for l in vid]
        if {r["label"] for r in b} != set(next(iter(labels))):
            raise ValueError("Prior readout rows differ from evaluation rows")
    allrows = {**{f"{arm}_new": rows[arm] for arm in ARMS}, **prior}
    pv = {k: per_video_useful(v) for k, v in allrows.items()}
    short = {k: per_video_useful([r for r in v if r.get("pos", 0) <= 3]) for k, v in rows.items()}
    late = {k: per_video_useful([r for r in v if r.get("pos", 0) > 3]) for k, v in rows.items()}

    def summ(v):
        pos = [r for r in v if r["kind"] == 1]
        return {"positives": len(pos), "useful_pooled": float(np.mean([r["useful"] for r in pos])), "useful_mean_video": float(np.mean(list(per_video_useful(v).values()))),
                "median_err_rel": float(np.median([r["err"] / r["scale"] for r in pos])), "near_crop_center": float(np.mean([r.get("near_crop_center", False) for r in pos])) if "near_crop_center" in pos[0] else None}
    rep = {"lambda_choice": res, "summary": {k: summ(v) for k, v in allrows.items()}, "kalman_control": "центр = прогноз: полезных 0 по определению (err < err_pred не выполняется)",
           "paired": {}, "by_history": {}, "min_shift": MIN_SHIFT}
    comps = [("D_new", "D_prior"), ("D_new", "R0_new"), ("C_new", "R0_new"), ("C_new", "C0_new"), ("D_new", "C_new"), ("C_new", "C_prior")]
    for x, y in comps:
        if x in pv and y in pv:
            rep["paired"][f"{x}-{y}"] = paired(pv[x], pv[y])
    for nm, part in (("search_pos_le3", short), ("search_pos_gt3", late)):
        rep["by_history"][nm] = {f"{x}_new-R0_new": paired(part[x], part["R0"]) for x in ("C", "D") if part.get(x) and part.get("R0")}   # ключи rows — имена вариантов (исправлено 17.10)
    def verdict(k):
        if k not in rep["paired"]:
            return None
        d = rep["paired"][k]
        return "полезно" if d["ci95"][0] > 0 and d["mean_per_video"] >= MIN_SHIFT else ("не полезно" if d["ci95"][1] < MIN_SHIFT else "неубедительно")
    rep["verdict"] = {"node_vs_prior_readout_D": verdict("D_new-D_prior"), "fly_vs_eye_input_D": verdict("D_new-R0_new"), "fly_vs_eye_input_C": verdict("C_new-R0_new")}
    atomic_json(run / "report.json", rep); console(state="completed", stage="report written")
    print(json.dumps({"summary": rep["summary"], "paired": rep["paired"], "verdict": rep["verdict"]}, ensure_ascii=False)[:3000])


if __name__ == "__main__":
    from tracking.gpu_lock import GpuLock
    with GpuLock("flyvis_fovea_gpu"):
        main()
