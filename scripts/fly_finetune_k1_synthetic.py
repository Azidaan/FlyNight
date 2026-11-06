"""К1, контроль процедуры: работает ли дообучение декодера там, где движение мухе заведомо доступно.

Кадры BDD (те же 20 видео train, отложенные — dev) сдвигаются как целое с постоянной скоростью
1–6 px/кадр в случайном направлении (cv2.warpAffine, край — отражение); цель — эта скорость во всех
колонках. Ветви обучаются той же train_arm из scripts/fly_finetune_k1.py; ошибка — средняя ошибка
конечной точки по колонкам на отложенных отрезках, px/кадр. Если дообученный декодер здесь ниже
нулевого движения, а на BDD с DIS — нет, отказ пилота не в процедуре, а в сигнале мухи на реальном видео.

Запуск: python scripts/fly_finetune_k1_synthetic.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import scripts.fly_finetune_k1 as K  # noqa: E402
from tracking.flyvis_geometry_protocol import add_geometry_arguments, ensure_geometry_output, resolve_output, require_arm_geometry, require_empirical_gain_geometry

OUT = PROJECT_ROOT / "runs" / "fly_finetune_k1" / "synthetic.json"


def segments(fly, vids, split, rng, per_video):
    import cv2
    segs = []
    for v in vids:
        frames = list(zip(range(400), K.read_gray(PROJECT_ROOT / "data/bdd100k/videos" / split / f"{v}.mov")))
        for f in rng.choice(len(frames), per_video, replace=False):
            g = frames[f][1]
            speed = rng.uniform(1, 6); ang = rng.uniform(0, 2 * np.pi); vel = np.array([np.cos(ang), np.sin(ang)]) * speed
            seq = []
            for j in range(K.WARM + K.LEN):
                M = np.float32([[1, 0, vel[0] * j], [0, 1, vel[1] * j]])
                seq.append(fly.tiles(cv2.warpAffine(g, M, (g.shape[1], g.shape[0]), borderMode=cv2.BORDER_REFLECT)))
            tgt = np.broadcast_to(vel.astype(np.float32), (K.LEN, 6 * 721, 2)).copy()
            segs.append((np.stack(seq), tgt))
    return segs


def evaluate(fly, segs, state, stock):
    if state is not None:
        require_arm_geometry({"synthetic": state}, fly.base.geometry_version)
    t = fly.torch; dec = fly.base.decoder
    _, sc = fly.points_scale(); sc = t.tensor(sc, device=fly.dev)
    dec.load_state_dict(state["decoder"] if state else stock)
    lg = t.tensor(state["log_gain"] if state else 0.0, device=fly.dev)
    fe = None
    if state and state.get("frontend"):
        fe = K.FrontEnd(t, fly.dev)
        with t.no_grad():
            for n, x in state["frontend"].items():
                fe.p[n].fill_(x)
    errs, zero = [], []
    with t.no_grad():
        for frames, tgt in segs:
            cart = t.from_numpy(frames).to(fly.dev, t.float32).unsqueeze(2) / 255.0
            if fe is not None:
                fe.reset()
            inp = (lambda c: fe(c)) if fe is not None else (lambda c: c)
            h0 = fly.render(inp(cart[0])); st = fly.fade_in(h0); last = h0
            base = st.nodes.activity.unsqueeze(1)
            for j in range(1, K.WARM + K.LEN):
                cur = fly.render(inp(cart[j])); st, act = fly.step(st, last, cur); last = cur
                if j >= K.WARM:
                    v = K.to_px(dec(act).mean(1) - dec(base)[:, 0], sc, lg).cpu().numpy()
                    errs.append(np.linalg.norm(v - tgt[j - K.WARM], axis=1).mean())
                    zero.append(np.linalg.norm(tgt[j - K.WARM], axis=1).mean())
    dec.load_state_dict(stock)
    return float(np.mean(errs)), float(np.mean(zero))


def main() -> int:
    global OUT
    import argparse
    parser = argparse.ArgumentParser(); add_geometry_arguments(parser); args = parser.parse_args()
    require_empirical_gain_geometry(args.geometry_version, 'fly_finetune_k1_synthetic')
    OUT = resolve_output(args.out, OUT, PROJECT_ROOT)
    ensure_geometry_output(OUT, args.geometry_version, file=True)
    selection_meta = K.OUT / "summary.json"
    K.OUT = OUT.parent
    fps = json.loads((K.TC / "0001542f-5ce3cf52/motion/dis_quarter.json").read_text(encoding="utf-8"))["fps"]
    fly = K.Fly(fps, geometry_version=args.geometry_version)
    stock = {k: v.detach().clone() for k, v in fly.base.decoder.state_dict().items()}
    meta = json.loads(selection_meta.read_text(encoding="utf-8"))["meta"] if selection_meta.exists() else None
    sel = json.loads((K.TC / "selection.json").read_text(encoding="utf-8"))["videos"]
    tr = meta["train"] if meta else sel["train"][:20]
    ev = sel["dev"][:10]
    rng = np.random.default_rng(0)
    seg_tr = segments(fly, tr, "train", rng, 10); seg_ev = segments(fly, ev, "train", rng, 4)
    print(f"[DATA] обучение {len(seg_tr)} отрезков, проверка {len(seg_ev)}", flush=True)
    res = {"geometry_version": args.geometry_version}
    e, z = evaluate(fly, seg_ev, None, stock); res["zero"] = z; res["stock"] = e
    print(f"[SYN] нулевое движение {z:.3f} | штатная муха (усиление стенда) {e:.3f}", flush=True)
    with open(OUT.with_suffix(".log"), "w", encoding="utf-8") as log:
        for arm, steps in (("scale", 300), ("decoder", 600)):
            fly.base.decoder.load_state_dict(stock)
            st = K.train_arm(arm, fly, seg_tr, steps, np.random.default_rng(1), log)
            res[arm], _ = evaluate(fly, seg_ev, st, stock)
            res[f"{arm}_gain"] = float(np.exp(st["log_gain"]))
            print(f"[SYN] {arm}: {res[arm]:.3f} (усиление {res[f'{arm}_gain']:.3f})", flush=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[OK] {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
