"""Где в мухе теряется информация о движении (карточка fly_stage_localization_v1; план автора 06.10, шаг 4).

Искусственный равномерный сдвиг кадров BDD dev без отражённых краёв (BORDER_CONSTANT, только колонки, у которых
источник изображения на всём пути внутри кадра с запасом), скорость 1 / 2 / 4 / 8 px/кадр и контраст ×0,25 / ×0,5 / ×1
(вокруг средней яркости кадра) меняются раздельно. Протокол: fade-in → STATIC кадров неподвижной сцены → LEN кадров сдвига;
точка отсчёта — установившаяся (средняя активность за последние 5 кадров покоя; fly_reference_point_v1).

Ступени (активность последнего внутреннего шага кадра минус точка отсчёта):
  photo   — R1–R6;
  early   — входы детекторов движения: L1, L2, L3, Mi1, Tm3, Mi4, Mi9 (ON) и Tm1, Tm2, Tm4, Tm9 (OFF);
  t4t5    — T4a–d, T5a–d;
  outputs — все 34 выходных типа (вход декодера);
  decoder — выход декодера Sintel (px при усилении 1) — без считывания, только общий масштаб.
Признаки колонки: типы ступени в колонке и 6 соседях, текущий и прошлый кадр. Считывание — гребневая регрессия
скорости (vx, vy), обучение на чётных отрезках (все контрасты), проверка на нечётных; R² по скорости × контрасту.
Отдельно — устойчивость к контрасту: обучение только на контрасте ×1, проверка на ×0,25.
Входы: raw и lcn24 (scripts/fly_noise_probe.py).

Запуск: python scripts/fly_stage_probe.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import scripts.fly_finetune_k1 as K  # noqa: E402
from scripts.fly_noise_probe import lcn  # noqa: E402
from scripts.fly_scale_probe import make_fly  # noqa: E402

OUT = PROJECT_ROOT / "runs" / "fly_stage_probe"
SPEEDS = [1.0, 2.0, 4.0, 8.0]
CONTRASTS = [0.25, 0.5, 1.0]
STATIC, LEN, MARGIN = 30, 8, 24
PER_TILE = 40
STAGES = {
    "photo": ["R1", "R2", "R3", "R4", "R5", "R6"],
    "early": ["L1", "L2", "L3", "Mi1", "Tm3", "Mi4", "Mi9", "Tm1", "Tm2", "Tm4", "Tm9"],
    "t4t5": ["T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d"],
}
HEX_NB = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]


def neighbours(u, v):
    pos = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(u, v))}
    return np.array([[pos.get((int(a) + du, int(b) + dv), i) for du, dv in HEX_NB] for i, (a, b) in enumerate(zip(u, v))])


def ridge_r2(Xtr, Ytr, Xte, Yte, alpha=10.0):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    A = (Xtr - mu) / sd; B = (Xte - mu) / sd
    A1 = np.c_[A, np.ones(len(A))]
    W = np.linalg.solve(A1.T @ A1 + alpha * np.eye(A1.shape[1]), A1.T @ Ytr)
    P = np.c_[B, np.ones(len(B))] @ W
    return float(1 - ((Yte - P) ** 2).sum() / ((Yte - Yte.mean(0)) ** 2).sum())


def main() -> int:
    import cv2
    OUT.mkdir(parents=True, exist_ok=True)
    fps = json.loads((K.TC / "0001542f-5ce3cf52/motion/dis_quarter.json").read_text(encoding="utf-8"))["fps"]
    sel = json.loads((K.TC / "selection.json").read_text(encoding="utf-8"))["videos"]
    rng = np.random.default_rng(0)
    frames = []
    for v in sel["dev"][:6]:
        g = [x for _, x in zip(range(300), K.read_gray(PROJECT_ROOT / "data/bdd100k/videos/train" / f"{v}.mov"))]
        frames += [g[i] for i in rng.choice(len(g), 2, replace=False)]
    fly = make_fly(fps, "2x3", 1)
    t = fly.torch; dec = fly.base.decoder
    li = fly.net.stimulus.layer_index
    c = fly.net.connectome
    any_t = li["T4a"]; nb = neighbours(c.nodes.u[:][any_t], c.nodes.v[:][any_t])
    out_types = [x.decode() for x in c.output_cell_types[:]]
    stage_idx = {k: np.stack([np.asarray(li[x]) for x in tl]) for k, tl in {**STAGES, "outputs": out_types}.items()}   # (типы, 721)
    res = {}
    t0 = time.time()
    for inp in ("raw", "lcn24"):
        tf = (lambda g: g.astype(np.float32)) if inp == "raw" else lcn(24.0)
        data = {k: {"X": [], "Y": [], "seg": [], "sp": [], "ct": []} for k in (*stage_idx, "decoder")}
        seg_id = 0
        for ci, ct in enumerate(CONTRASTS):
            for sp in SPEEDS:
                for g in frames:
                    a = rng.uniform(0, 2 * np.pi); vel = np.array([np.cos(a), np.sin(a)]) * sp
                    m = float(g.mean()); gc = np.clip(m + ct * (g.astype(np.float32) - m), 0, 255).astype(np.uint8)
                    H, W = g.shape
                    seq = [gc] * (1 + STATIC) + [cv2.warpAffine(gc, np.float32([[1, 0, vel[0] * j], [0, 1, vel[1] * j]]), (W, H), borderMode=cv2.BORDER_CONSTANT) for j in range(1, LEN + 1)]
                    tiles = [fly.tiles(tf(x)) for x in seq]
                    pts, sc = fly.points_scale(); sct = t.tensor(sc, device=fly.dev)
                    end = pts - vel * LEN
                    ok = ((pts[:, 0] > MARGIN) & (pts[:, 0] < W - MARGIN) & (pts[:, 1] > MARGIN) & (pts[:, 1] < H - MARGIN)
                          & (end[:, 0] > MARGIN) & (end[:, 0] < W - MARGIN) & (end[:, 1] > MARGIN) & (end[:, 1] < H - MARGIN)).reshape(6, 721)
                    acts, decs, hist_a, hist_d = [], [], [], []
                    with t.no_grad():
                        cart = lambda x: t.from_numpy(x).to(fly.dev, t.float32).unsqueeze(1) / 255.0
                        cur = fly.render(cart(tiles[0])); state = fly.fade_in(cur); last = cur
                        for j in range(1, len(tiles)):
                            cur = fly.render(cart(tiles[j])); state, act = fly.step(state, last, cur); last = cur
                            a_np = act[:, -1].cpu().numpy(); d_np = dec(act).mean(1).cpu().numpy()   # (6, nodes), (6, 2, 721)
                            if STATIC - 5 < j <= STATIC:
                                hist_a.append(a_np); hist_d.append(d_np)
                            if j >= STATIC:
                                acts.append(a_np); decs.append(d_np)
                    ref_a, ref_d = np.mean(hist_a, 0), np.mean(hist_d, 0)
                    picks = [rng.choice(np.flatnonzero(ok[k]), min(PER_TILE, ok[k].sum()), replace=False) if ok[k].any() else np.array([], int) for k in range(6)]
                    for j in range(1, len(acts)):          # j-й кадр сдвига, прошлый — j−1 (нулевой — последний кадр покоя)
                        A_now, A_prev = acts[j] - ref_a, acts[j - 1] - ref_a
                        for stg, idx in stage_idx.items():
                            feats = []
                            for k in range(6):
                                if not len(picks[k]):
                                    continue
                                cols = np.concatenate([picks[k][:, None], nb[picks[k]]], 1)       # (n, 7)
                                fn = A_now[k][idx][:, cols]; fp = A_prev[k][idx][:, cols]          # (типы, n, 7)
                                feats.append(np.concatenate([fn.transpose(1, 0, 2).reshape(len(cols), -1), fp.transpose(1, 0, 2).reshape(len(cols), -1)], 1))
                            X = np.concatenate(feats).astype(np.float32)
                            D = data[stg]; D["X"].append(X); D["Y"].append(np.broadcast_to(vel, (len(X), 2))); D["seg"].append(np.full(len(X), seg_id)); D["sp"].append(np.full(len(X), sp)); D["ct"].append(np.full(len(X), ct))
                        px = K.to_px(t.from_numpy(decs[j] - ref_d).to(fly.dev), sct, t.zeros((), device=fly.dev)).cpu().numpy().reshape(6, 721, 2)
                        X = np.concatenate([px[k][picks[k]] for k in range(6)])
                        D = data["decoder"]; D["X"].append(X); D["Y"].append(np.broadcast_to(vel, (len(X), 2))); D["seg"].append(np.full(len(X), seg_id)); D["sp"].append(np.full(len(X), sp)); D["ct"].append(np.full(len(X), ct))
                    seg_id += 1
                print(f"[STAGE] {inp} контраст {ct} скорость {sp}: готово | {time.time() - t0:.0f} с", flush=True)
                K.status(stage="stage_probe", input=inp, contrast=ct, speed=sp)
        res[inp] = {}
        for stg, D in data.items():
            X, Y, seg, sp_, ct_ = (np.concatenate(D[k]) for k in ("X", "Y", "seg", "sp", "ct"))
            tr, te = seg % 2 == 0, seg % 2 == 1
            r = {}
            if stg == "decoder":
                gk = float((X[tr] * Y[tr]).sum() / (X[tr] ** 2).sum())
                r2 = lambda m: float(1 - ((Y[m] - gk * X[m]) ** 2).sum() / ((Y[m] - Y[m].mean(0)) ** 2).sum())
                r["all"] = r2(te)
                for ct in CONTRASTS:
                    for sp in SPEEDS:
                        r[f"c{ct}_s{sp}"] = r2(te & (ct_ == ct) & (sp_ == sp))
                c1 = tr & (ct_ == 1.0); gk1 = float((X[c1] * Y[c1]).sum() / (X[c1] ** 2).sum())
                m = te & (ct_ == 0.25); r["train_c1_test_c0.25"] = float(1 - ((Y[m] - gk1 * X[m]) ** 2).sum() / ((Y[m] - Y[m].mean(0)) ** 2).sum())
            else:
                r["all"] = ridge_r2(X[tr], Y[tr], X[te], Y[te])
                mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
                A1 = np.c_[(X[tr] - mu) / sd, np.ones(tr.sum())]
                Wt = np.linalg.solve(A1.T @ A1 + 10.0 * np.eye(A1.shape[1]), A1.T @ Y[tr])
                for ct in CONTRASTS:
                    for sp in SPEEDS:
                        m = te & (ct_ == ct) & (sp_ == sp)
                        P = np.c_[(X[m] - mu) / sd, np.ones(m.sum())] @ Wt
                        r[f"c{ct}_s{sp}"] = float(1 - ((Y[m] - P) ** 2).sum() / ((Y[m] - Y[m].mean(0)) ** 2).sum())
                c1 = tr & (ct_ == 1.0); m = te & (ct_ == 0.25)
                r["train_c1_test_c0.25"] = ridge_r2(X[c1], Y[c1], X[m], Y[m])
            res[inp][stg] = r
            print(f"[STAGE] {inp} {stg:8s}: R² {r['all']:.3f} | по контрасту (скорость 4): " + " ".join(f"{ct}:{r[f'c{ct}_s4.0']:.3f}" for ct in CONTRASTS)
                  + f" | обучение ×1 → проверка ×0,25: {r['train_c1_test_c0.25']:.3f}", flush=True)
        (OUT / "stages.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[OK] {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
