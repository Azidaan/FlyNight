"""Сравнение настоящей мухи с мухой на перемешанном коннектоме (карточка fly_connectome_null_v1).

Оба отчёта — scripts/eval_fly_reid.py --ladder-only на одних и тех же роликах, треках и парах
(трекер и сетчатка одинаковы; меняется только коннектом). Парный бутстреп по видео разности AP
для голов с мухой; разрез dev+test по средней яркости ролика (< 60/255).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_bdd_video_rate_experiment import save_json  # noqa: E402

HEADS = ("geometry_fly", "geometry_hexin_fly", "geometry_appearance_hexin_fly")


def boot(d: np.ndarray) -> dict:
    rng = np.random.default_rng(42); b = d[rng.integers(0, len(d), (5000, len(d)))].mean(1)
    return {"mean": float(d.mean()), "ci95": [float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))], "n": int(len(d)), "better": int((d > 0).sum()), "worse": int((d < 0).sum())}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intact", type=Path, default=PROJECT_ROOT / "runs/fly_reid_controls/report.json")
    parser.add_argument("--null", type=Path, default=PROJECT_ROOT / "runs/fly_reid_null/report.json")
    parser.add_argument("--luminance", type=Path, default=PROJECT_ROOT / "runs/motion_reliability_lamina/luminance.json")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "runs/fly_reid_null/compare.json")
    args = parser.parse_args()
    I = json.loads(args.intact.read_text(encoding="utf-8"))["ladder"]; N = json.loads(args.null.read_text(encoding="utf-8"))["ladder"]
    lum = json.loads(args.luminance.read_text(encoding="utf-8"))
    out = {"aggregate": {}, "bootstrap": {}}
    for head in HEADS:
        out["aggregate"][head] = {s: {"intact": I[head][s], "null": N[head][s]} for s in ("dev", "test", "val7")}
        for s in ("dev", "test", "val7"):
            a, b = N[head][f"{s}_by_video"], I[head][f"{s}_by_video"]
            out["bootstrap"][f"{head}|{s}"] = boot(np.asarray([b[v] - a[v] for v in a if v in b]))
        for grp, test in (("dark", lambda x: x < 60), ("bright", lambda x: x >= 60)):
            d = [I[head][f"{s}_by_video"][v] - N[head][f"{s}_by_video"][v] for s in ("dev", "test") for v in N[head][f"{s}_by_video"] if v in I[head][f"{s}_by_video"] and v in lum and test(lum[v])]
            out["bootstrap"][f"{head}|{grp}"] = boot(np.asarray(d))
        for k in (f"{head}|dev", f"{head}|test", f"{head}|val7", f"{head}|dark", f"{head}|bright"):
            r = out["bootstrap"][k]
            print(f"[NULL] настоящая − перемешанная {k:40s} {r['mean']:+.4f} [{r['ci95'][0]:+.4f}; {r['ci95'][1]:+.4f}] {r['better']}/{r['worse']}", flush=True)
        print(f"[AP] {head:32s} " + " | ".join(f"{s}: настоящая {I[head][s]:.4f} перемешанная {N[head][s]:.4f}" for s in ("dev", "test", "val7")), flush=True)
    for head in ("geometry_hexin", "geometry"):
        out["aggregate"][head] = {s: {"intact": I[head][s], "null": N[head][s]} for s in ("test",)}  # контроль: без мухи отчёты должны совпасть
    save_json(args.output, out)
    print(f"[OK] {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
