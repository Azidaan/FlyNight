"""Д5: держатся ли выводы о мухе на нескольких моделях ансамбля FlyVis (карточка fly_ensemble_members_v1).

Для каждой модели — отчёт scripts/eval_fly_reid.py --ladder-only (одни и те же ролики, треки и пары; меняется
только модель мухи). Модель 000 — runs/fly_reid_controls/report.json. Эффекты (по видео, AP):
  fly_over_geometry = geometry_fly − geometry
  fly_over_input    = geometry_hexin_fly − geometry_hexin
По каждой модели — бутстреп по видео; по ансамблю — двухуровневый бутстреп (модели → видео);
разрез dev+test по средней яркости ролика (< 60/255), как в fly_matched_controls_v1.

Запуск: python scripts/compare_ensemble_members.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

REPORTS = {"000": PROJECT_ROOT / "runs/fly_reid_controls/report.json",
           "012": PROJECT_ROOT / "runs/fly_reid_m012/report.json",
           "025": PROJECT_ROOT / "runs/fly_reid_m025/report.json",
           "037": PROJECT_ROOT / "runs/fly_reid_m037/report.json"}
EFFECTS = {"fly_over_geometry": ("geometry_fly", "geometry"), "fly_over_input": ("geometry_hexin_fly", "geometry_hexin")}
OUT = PROJECT_ROOT / "runs" / "fly_ensemble_members" / "compare.json"


def per_video(ladder, head, split):
    return ladder[head][f"{split}_by_video"]


def main() -> int:
    lum = json.loads((PROJECT_ROOT / "runs/motion_reliability_lamina/luminance.json").read_text(encoding="utf-8"))
    members = {m: json.loads(p.read_text(encoding="utf-8"))["ladder"] for m, p in REPORTS.items() if p.exists()}
    rng = np.random.default_rng(0)
    out = {"members": list(members), "per_member": {}, "ensemble": {}}
    diffs = {}  # (effect, subset) -> member -> {video: d}
    for m, L in members.items():
        out["per_member"][m] = {}
        for eff, (a, b) in EFFECTS.items():
            for split in ("test", "dev", "val7"):
                A, B = per_video(L, a, split), per_video(L, b, split)
                d = {v: float(A[v] - B[v]) for v in A if v in B}
                diffs.setdefault((eff, split), {})[m] = d
            dt = {**diffs[(eff, "dev")][m], **diffs[(eff, "test")][m]}
            diffs.setdefault((eff, "dark"), {})[m] = {v: x for v, x in dt.items() if v in lum and lum[v] < 60}
            diffs.setdefault((eff, "bright"), {})[m] = {v: x for v, x in dt.items() if v in lum and lum[v] >= 60}
    for (eff, sub), by_m in diffs.items():
        for m, d in by_m.items():
            x = np.array(list(d.values()))
            if not len(x):
                continue
            b = x[rng.integers(0, len(x), (5000, len(x)))].mean(1)
            out["per_member"][m][f"{eff}|{sub}"] = {"mean": float(x.mean()), "ci95": [float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))], "better": int((x > 0).sum()), "worse": int((x < 0).sum()), "n": int(len(x))}
        ms = [m for m in by_m if by_m[m]]
        vids = sorted(set.intersection(*[set(by_m[m]) for m in ms])) if ms else []
        if len(ms) < 2 or not vids:
            continue
        M = np.array([[by_m[m][v] for v in vids] for m in ms])   # (members, videos)
        bs = []
        for _ in range(5000):
            mi = rng.integers(0, len(ms), len(ms)); vi = rng.integers(0, len(vids), len(vids))
            bs.append(M[mi][:, vi].mean())
        out["ensemble"][f"{eff}|{sub}"] = {"mean": float(M.mean()), "ci95": [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))],
                                            "member_means": {m: float(M[i].mean()) for i, m in enumerate(ms)},
                                            "sign_agree": int(sum(np.sign(M[i].mean()) == np.sign(M.mean()) for i in range(len(ms)))), "members": len(ms)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    for k, v in out["ensemble"].items():
        print(f"{k:28s} ансамбль {v['mean']:+.4f} [{v['ci95'][0]:+.4f}; {v['ci95'][1]:+.4f}]  по моделям " + ", ".join(f"{m} {x:+.4f}" for m, x in v["member_means"].items()) + f"  знак совпал {v['sign_agree']}/{v['members']}")
    print(f"[OK] {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
