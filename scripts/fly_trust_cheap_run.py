"""fly_motion_trust_cheap_v1: подбор дешёвого доверия к движению мухи на одних видео и проверка на других.

Подбор: 18 видео dev (чётные позиции selection.json, runs/motion_scheme_search/trust_split.json), сетка
w ∈ {0,1; 0,2; 0,3; 0,5; 1,0} × m ∈ {0; 3; 6} (вес поправки × w, запрет поправки при < m колонок в рамке), схема 5 Гц.
Выбор — по средней парной разности HOTA с «без движения» на подборе. Проверка выбранного варианта: 17 других видео dev и
7 видео val7 (в подборе не участвуют); критерий — парная разность HOTA на 24 отложенных видео > 0, ДИ > 0.
Выход: runs/motion_scheme_search/trust_cheap.json.

Запуск: python scripts/fly_trust_cheap_run.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT = PROJECT_ROOT / "runs/motion_scheme_search"
PY = sys.executable
GRID = [f"fly@w{w}m{m}" for w in (0.1, 0.2, 0.3, 0.5, 1.0) for m in (0, 3, 6)]


def run(set_, vids, sources, tag):
    cmd = [PY, "-u", str(PROJECT_ROOT / "scripts/motion_scheme_search.py"), "--set", set_, "--schemes", "r5_all", "--sources", *sources, "--tag", tag]
    if vids:
        cmd += ["--videos", *vids]
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)
    return json.loads((OUT / f"{set_}{tag}.json").read_text(encoding="utf-8"))


def diffs(res, key, vids):
    base = res["r5_all_none"]
    return np.array([res[key][v]["HOTA"] - base[v]["HOTA"] for v in vids if v in res.get(key, {}) and v in base])


def boot(d, seed=1):
    rng = np.random.default_rng(seed); bs = d[rng.integers(0, len(d), (5000, len(d)))].mean(1)
    return {"mean": float(d.mean()), "ci95": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))], "wins": int((d > 0).sum()), "n": int(len(d))}


def main() -> int:
    split = json.loads((OUT / "trust_split.json").read_text(encoding="utf-8"))
    tune = run("dev", split["tune"], ["none"] + GRID, "_trust_tune")
    sel = {g: boot(diffs(tune, f"r5_all_{g}", split["tune"])) for g in GRID}
    for g, r in sel.items():
        print(f"[TUNE] {g}: {r['mean']:+.3f} [{r['ci95'][0]:+.3f}; {r['ci95'][1]:+.3f}] лучше {r['wins']}/{r['n']}", flush=True)
    best = max(GRID, key=lambda g: sel[g]["mean"])
    print(f"[BEST] {best}: {sel[best]['mean']:+.3f}", flush=True)
    test_dev = run("dev", split["test"], ["none", "dis", best], "_trust_test")
    test_val = run("val7", None, ["none", "dis", best], "_trust_test")
    d_dev, d_val = diffs(test_dev, f"r5_all_{best}", split["test"]), diffs(test_val, f"r5_all_{best}", list(test_val["r5_all_none"]))
    held = boot(np.concatenate([d_dev, d_val]))
    dis_held = boot(np.concatenate([diffs(test_dev, "r5_all_dis", split["test"]), diffs(test_val, "r5_all_dis", list(test_val["r5_all_none"]))]))
    out = {"grid_tune": sel, "best": best, "held_out_24": held, "held_out_dev17": boot(d_dev), "held_out_val7": boot(d_val), "dis_held_out_24": dis_held,
           "passed": bool(held["ci95"][0] > 0)}
    (OUT / "trust_cheap.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"== {best} на 24 отложенных: {held['mean']:+.3f} [{held['ci95'][0]:+.3f}; {held['ci95'][1]:+.3f}], лучше {held['wins']}/{held['n']} | DIS: {dis_held['mean']:+.3f} | пройден: {out['passed']}", flush=True)
    print("[OK]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
