"""Bounded CPU-only audit; never edit production or use existing experiment caches."""
from pathlib import Path
import hashlib
import json
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts import run_bdd_video_rate_experiment as module


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_video(path, step):
    rng = np.random.default_rng(1771)
    texture = rng.integers(0, 256, (64, 96, 3), dtype=np.uint8)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 5., (96, 64))
    assert writer.isOpened()
    for index in range(5):
        writer.write(np.roll(texture, step * index, axis=1))
    writer.release()


def main():
    out = Path(__file__).resolve().parent
    work = out / "synthetic_inputs"
    work.mkdir(exist_ok=True)
    assert not (out / "before_report.json").exists(), "Never replace evidence"
    a, b = work / "right.avi", work / "left.avi"
    make_video(a, 2)
    make_video(b, -3)
    shared, fresh = work / "cache_from_right", work / "fresh_left"
    kwargs = dict(source="farneback", fps=5., max_dt=.02, force=False)
    first = module.build_motion_cache(video=a, output=shared, **kwargs)
    array_path = shared / "farneback.npz"
    before_bytes = sha(array_path)
    same = module.build_motion_cache(video=a, output=shared, **kwargs)
    original = module.load_motion_cache(array_path, 1.)
    switched = module.build_motion_cache(video=b, output=shared, **kwargs)
    wrongly_reused = module.load_motion_cache(array_path, 1.)
    control = module.build_motion_cache(video=b, output=fresh, **kwargs)
    correct = module.load_motion_cache(fresh / "farneback.npz", 1.)
    assert first == same == switched
    assert before_bytes == sha(array_path)
    assert np.array_equal(original[1], wrongly_reused[1])
    assert not np.allclose(wrongly_reused[1], correct[1])
    import torch
    assert not torch.cuda.is_initialized()
    report = {
        "source_path": str(Path(module.__file__).relative_to(ROOT)),
        "source_sha256": sha(module.__file__),
        "cpu_only_cuda_initialized": torch.cuda.is_initialized(),
        "source": "farneback", "frame_count": 5, "fps": 5.,
        "right_sha256": sha(a), "left_sha256": sha(b),
        "same_video_control_exact": first == same,
        "other_video_silently_reuses_metadata": first == switched,
        "other_video_silently_reuses_vectors": bool(np.array_equal(original[1], wrongly_reused[1])),
        "cache_bytes_preserved": before_bytes == sha(array_path),
        "fresh_other_video_differs": not bool(np.allclose(wrongly_reused[1], correct[1])),
        "max_abs_wrong_vs_correct": float(np.max(np.abs(wrongly_reused[1] - correct[1]))),
        "cached_last_median_dx": float(np.median(wrongly_reused[1][-1, :, 0])),
        "fresh_last_median_dx": float(np.median(correct[1][-1, :, 0])),
        "metadata_keys": sorted(first),
        "actual_callers": ["scripts/run_bdd_video_rate_experiment.py:main", "scripts/build_train_caches.py:main"],
        "scope": "Cache identity defect; no HOTA or scientific quality claim. No production edits.",
    }
    (out / "before_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
