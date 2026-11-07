"""Independent cache byte/output controls on the original generated CPU videos."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts import run_bdd_video_rate_experiment as module


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    out = Path(__file__).resolve().parent
    before = out / "synthetic_inputs"
    assert not (out / "after_report.json").exists()
    first, other = before / "right.avi", before / "left.avi"
    target = out / "native_after/right"
    options = dict(source="farneback", fps=5., max_dt=.02, force=False, device="cpu")
    meta = module.build_motion_cache(video=first, output=target, **options)
    arrays = module.load_motion_cache(target / "farneback.npz", 1.)
    original = module.load_motion_cache(before / "cache_from_right/farneback.npz", 1.)
    assert all(np.array_equal(a, b) for a, b in zip(arrays, original))
    cache_before = [(target / ("farneback" + ext)).read_bytes() for ext in (".npz", ".json")]
    assert module.build_motion_cache(video=first, output=target, **options) == meta
    messages = []
    for force in (False, True):
        try:
            module.build_motion_cache(video=other, output=target, **dict(options, force=force))
        except RuntimeError as exc:
            messages.append(str(exc))
        else:
            raise AssertionError("Changed video reused the old cache")
    cache_after = [(target / ("farneback" + ext)).read_bytes() for ext in (".npz", ".json")]
    assert cache_before == cache_after
    fresh = out / "native_after/left"
    module.build_motion_cache(video=other, output=fresh, **options)
    second = module.load_motion_cache(fresh / "farneback.npz", 1.)
    baseline_second = module.load_motion_cache(before / "fresh_left/farneback.npz", 1.)
    assert all(np.array_equal(a, b) for a, b in zip(second, baseline_second))
    import torch
    assert not torch.cuda.is_initialized()
    report = {
        "source_sha256": digest(module.__file__),
        "baseline_source_sha256": digest(out / "baseline/run_bdd_video_rate_experiment.py"),
        "right_native_points_and_vectors_exact_to_before": True,
        "left_native_points_and_vectors_exact_to_before": True,
        "same_video_metadata_exact": True,
        "changed_video_rejected_force_false_and_true": messages,
        "old_cache_bytes_exact_after_rejection": cache_before == cache_after,
        "new_video_in_new_output_works": True,
        "frames_per_video": len(arrays[1]),
        "points_per_frame": len(arrays[0]),
        "cuda_initialized": torch.cuda.is_initialized(),
        "limitations": ["Old caches without video_identity remain readable but do not prove video provenance.",
                        "The existing two-file npz/json save is not a transaction; disk/power-loss atomicity is not claimed.",
                        "Video identity is not checkpoint/decoder identity and not a scientific quality metric."]
    }
    (out / "after_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
