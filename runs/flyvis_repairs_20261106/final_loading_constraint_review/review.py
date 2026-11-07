"""Independent source/evidence verification; no network/model construction."""
from pathlib import Path
import hashlib
import json

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
EVIDENCE = ROOT / "runs/flyvis_repairs_20261106/loading_constraint_followup"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    before = json.loads((EVIDENCE / "report.json").read_text(encoding="utf-8"))
    after = json.loads((EVIDENCE / "after_report.json").read_text(encoding="utf-8"))
    hashes = {
        "tracking/flyvis_checkpoint.py": "0280e2ef363f85a2a9455fcd2a2a8148e23daf2bf3d1c394eb6c93cff07fd30e",
        "tracking/flyvis_connectome_swap.py": "de6a52ba397f7464ec6f75aa8a9c7cbd03456ecf14ccb51742d235e42b93fa32",
    }
    assert all(sha(ROOT / p) == value for p, value in hashes.items())
    assert all(sha(ROOT / p) == value for p, value in after["protected_sha256"].items())
    assert after["stock_count"] == 50 and len(after["stock"]) == 50
    assert all(row["validated"] and row["input_unchanged"] and row["live_storage_unchanged"] for row in after["stock"])
    with np.load(EVIDENCE / "outputs.npz", allow_pickle=False) as old, \
            np.load(EVIDENCE / "after_outputs.npz", allow_pickle=False) as new:
        assert np.array_equal(old["stock_frozen"], new["stock"])
        assert np.array_equal(new["old_transfer"], new["new_transfer"])
        assert np.array_equal(old["negative_strength_trainable_official_clamp"], old["zero_strength_frozen_control"])
        difference = float(np.max(np.abs(old["negative_strength_frozen"] - old["negative_strength_trainable_official_clamp"])))
        assert difference == before["comparisons"]["negative_frozen_vs_clamped_max_abs"]
        nonfinite = int(np.sum(~np.isfinite(old["finite_log_count_semantic_overflow_frozen"])))
        assert nonfinite == 61531
    report = {
        "reviewer": "/root/motion_fix",
        "conclusion": "PASS within the declared bounds",
        "material_findings": [],
        "source_sha256": hashes,
        "protected_weights_and_config_hashes_independently_verified": len(after["protected_sha256"]),
        "independently_recomputed_negative_strength_activity_difference": difference,
        "independently_recomputed_overflow_nonfinite_activity_elements": nonfinite,
        "saved_stock_activity_exact": True,
        "saved_transfer_activity_exact": True,
        "source_review": {
            "mutation_order": "All source/state/domain checks precede live load_state_dict and copy_; mapped transfer tensors are staged first.",
            "declared_clamp": "None, non_negative and explicit two-element bounds match runtime Network.clamp. No new tau bound is inferred.",
            "shallow_copy": "Official InitialDistribution/Value/Normal/Lognormal are plain Python objects; semantic_values reads only raw_values, replaced on the copy with a detached clone. The original parameter object/storage is untouched.",
            "fixed_connectome": "Complete destination state supplies fixed sign/count to validation without copying source fixed connectivity.",
            "numpy_optimizer_guard": "Read-only inspection of the three-line numeric NumPy branch found no mutation or finite-path numerical change; nonnumeric object/string arrays and full Adam type validation are outside its claim.",
        },
        "limitations": [
            "Independent check recomputed stored arrays and hashes; no new 50-model simulation was run.",
            "Finite semantic values can still overflow later products; the new guard does not prove global dynamical stability.",
            "Shallow-copy reasoning covers the installed official distributions, not arbitrary custom getters with side effects.",
            "No training, scientific quality metric, GPU initialization or production edit was performed by this review.",
        ],
        "evidence_sha256": {name: sha(EVIDENCE / name) for name in ("report.json", "after_report.json", "outputs.npz", "after_outputs.npz", "verify_after.py")},
        "review_script_sha256": sha(__file__),
    }
    target = OUT / "report.json"
    assert not target.exists()
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
