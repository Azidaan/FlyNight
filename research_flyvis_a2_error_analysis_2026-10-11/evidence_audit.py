"""Read-only evidence snapshot and mathematical examples, not a tracking experiment.

Uses NumPy on CPU. Does not import the FlyVis runner, train a model, invoke
TrackEval, alter project sources, or read unfinished history archives.
"""
from __future__ import annotations

import ast
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
SOURCES = [
    "AGENTS.md",
    "scripts/life_fly_history.py",
    "scripts/build_fly_signatures.py",
    "scripts/life_filter_head.py",
    "scripts/life_filter_boost.py",
    "scripts/chain_filter_head.py",
    "scripts/output_filter_oracle.py",
    "tracking/foveated_eyes.py",
    "evaluation/hota_eval.py",
    "datasets/subset_filter.py",
    "runs/signature_overlap_reconcile/report.json",
    "runs/signature_overlap_reconcile/count_check.json",
    "runs/newval_long_false_oracle/report.json",
    "runs/newval_oracle/report.json",
    "runs/newval_error_analysis/report.json",
    "runs/output_filter_oracle/report.json",
    "runs/life_fly_history/confirm_videos.json",
]


def load(rel):
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def main():
    snapshot = {"read_at_utc": datetime.now(timezone.utc).isoformat(),
                "kind": "source review and numerical counterexamples; no new HOTA",
                "sources": {}}
    for rel in SOURCES:
        path = ROOT / rel
        content = path.read_bytes()
        snapshot["sources"][rel] = {
            "sha256": hashlib.sha256(content).hexdigest(),
            "bytes": len(content),
            "mtime_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
        }

    # Compile only the stateless stats() function, never execute module imports/main().
    source = (ROOT / "scripts/life_fly_history.py").read_text(encoding="utf-8")
    module = ast.parse(source)
    fn = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "stats")
    scope = {"np": np}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "isolated_stats", "exec"), scope)
    a = np.asarray([0, 0, 0, 0, 1, 1, 0, 1], np.float32)[:, None]
    b = np.asarray([0, 0, 0, 1, 0, 0, 1, 1], np.float32)[:, None]
    sa, sb = scope["stats"](a), scope["stats"](b)
    assert not np.array_equal(a, b)
    assert np.allclose(sa, sb, atol=1e-7, rtol=0)
    assert float(a[:5].mean()) == float(b[:5].mean())
    assert float(a[-5:].mean()) == float(b[-5:].mean())
    snapshot["representation_counterexample"] = {
        "scope": "one channel; demonstrates non-injectivity, not an observed cause of failure",
        "a": a[:, 0].tolist(), "b": b[:, 0].tolist(),
        "six_stats_a": sa.tolist(), "six_stats_b": sb.tolist(),
        "max_difference": float(np.max(np.abs(sa - sb))),
        "first5_mean_both": float(a[:5].mean()),
        "last5_mean_both": float(a[-5:].mean()),
        "lag2_product_mean_a": float((a[:-2] * a[2:]).mean()),
        "lag2_product_mean_b": float((b[:-2] * b[2:]).mean()),
    }

    n = 64
    chosen = np.unique(np.round(np.linspace(0, n - 1, min(n, 32))).astype(int))
    omitted = next(i for i in range(n) if i not in set(chosen.tolist()))
    impulse = np.zeros((n, 1), np.float32)
    impulse[omitted] = 1
    snapshot["thinning_counterexample"] = {
        "n_steps": n, "kept_indices": chosen.tolist(), "impulse_at": omitted,
        "max_before": float(impulse.max()), "max_after": float(impulse[chosen].max()),
        "scope": "mathematical example only; no claim that useful FlyVis events are missed in data",
    }

    snapshot["feature_dimensions_from_code"] = {
        "base_nofly": 751, "history_input_channels": 721 + 256 + 40,
        "history_nofly": 1 + 6 * (721 + 256 + 40),
        "total_nofly": 751 + 1 + 6 * (721 + 256 + 40),
        "old_fly_means": 2 * 34 * 19, "history_fly": 6 * 34 * 19,
        "total_fly": 751 + 1 + 6 * (721 + 256 + 40) + 2 * 34 * 19 + 6 * 34 * 19,
    }

    long = load("runs/newval_long_false_oracle/report.json")
    snapshot["existing_long_false_oracle"] = long
    ng = long["oracle"]["night"]
    cnt = long["counts"]
    snapshot["derived_existing_results"] = {
        "night_long_oracle_delta_pooled": ng["oracle_false_gt5"] - ng["none"],
        "night_long_oracle_relative_percent": 100 * (ng["oracle_false_gt5"] / ng["none"] - 1),
        "night_long_to_full_delta_ratio": (ng["oracle_false_gt5"] - ng["none"]) / (ng["oracle_full"] - ng["none"]),
        "long_false_share_lives_all97": cnt["false_gt5"]["lives"] / (cnt["false_gt5"]["lives"] + cnt["false_le5"]["lives"]),
        "long_false_share_fp_boxes_all97": cnt["false_gt5"]["fp_boxes"] / (cnt["false_gt5"]["fp_boxes"] + cnt["false_le5"]["fp_boxes"]),
    }

    snapshot["reconciled_populations"] = load("runs/signature_overlap_reconcile/report.json")
    snapshot["historical_chain_oracle_removed_box_overlaps"] = load("runs/output_filter_oracle/report.json")["removed"]
    snapshot["confirm_manifest"] = load("runs/life_fly_history/confirm_videos.json")
    history = load("experiments/project_history.json")
    snapshot["a2_card_at_read_time"] = next(x for x in history["experiments"] if x["id"] == "life_fly_history_v1")

    # Only the finished pilot known before this review; no sample-level labels used.
    pilot = "019efe88-bbc2a4ac"
    hp = ROOT / "runs/fly_signatures_hist" / pilot / "history.npz"
    sp = hp.with_name("signatures.npz")
    with np.load(hp) as z, np.load(sp) as s:
        steps = z["n_steps"]
        lengths = np.diff(z["offsets"])
        snapshot["pilot_metadata"] = {
            "video_id": pilot, "lives_all_including_unlabeled": len(steps),
            "n_steps_min": int(steps.min()), "n_steps_max": int(steps.max()),
            "lives_thinned": int((steps > 32).sum()),
            "kept_count_equals_min_n_32": bool(np.array_equal(lengths, np.minimum(steps, 32))),
            "n_steps_equals_end_minus_start": bool(np.array_equal(steps, s["end"] - s["start"])),
            "scope": "one pilot; not representative of all 165 videos or labeled lives",
        }
    target = OUT / "EVIDENCE.json"
    target.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"saved": str(target), "counterexample": snapshot["representation_counterexample"],
                      "dimensions": snapshot["feature_dimensions_from_code"],
                      "pilot_metadata": snapshot["pilot_metadata"],
                      "derived": snapshot["derived_existing_results"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
