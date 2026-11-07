"""Reproduce the three differing tracker outputs with one component changed at a time."""
from pathlib import Path
from unittest.mock import patch
import hashlib
import json
import os
import subprocess
import sys
import types
import zipfile

from compare_video_run import compare_array
from system_memory_guard import guard_entrypoint

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PARENT = "f18b3ef8ebe398fa490d90d946b13313eb6a11cd"
BEFORE_ASSIGNMENT_FIX = "e060d8539accdb934939b92a1e29d2246795858b"


def load_revision(commit, path, name):
    source = subprocess.check_output(["git", "show", f"{commit}:{path}"], cwd=ROOT)
    module = types.ModuleType(name)
    module.__file__ = str(ROOT / path)
    sys.modules[name] = module
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module, hashlib.sha256(source).hexdigest()


def compare_repeats(runner, video, fps, directory):
    import cv2
    import numpy as np
    from tracking.b1_deterministic_runtime import deterministic_call

    capture = cv2.VideoCapture(str(video))
    frames = []
    try:
        for _ in range(12):
            ok, frame = capture.read()
            if not ok:
                raise ValueError("Incomplete frame sample")
            frames.append(frame)
    finally:
        capture.release()
    result = {"frames": len(frames), "video_sha256": runner.sha256(video), "fields": {}}
    for source in ("flyvis_full", "flyvis_tiled"):
        estimator = runner.make_motion_estimator(source, fps, .02, device="cuda:0")
        def compute():
            estimator.reset()
            return np.stack([estimator.observe(frame).vectors_xy.copy() for frame in frames])
        conditions = {}
        for name, function in (("default", compute), ("deterministic", deterministic_call(compute))):
            values = [function() for _ in range(3)]
            if not all(np.isfinite(field).all() for field in values):
                raise ValueError(f"Nonfinite field for {source}")
            conditions[name] = {"sha256": [hashlib.sha256(value.tobytes()).hexdigest() for value in values],
                                "max_abs_difference": max(float(np.abs(v - values[0]).max()) for v in values)}
        result["fields"][source] = conditions
    (directory / "repeatability.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)


def main():
    job = guard_entrypoint(timeout=1200)
    sys.path.insert(0, str(ROOT))
    if sys.argv[1:] == ["--repeatability"]:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import cv2
    import torch
    from tracking.flyvis_tracker import WeightedGeometryAssignment
    from tracking.gpu_lock import CANONICAL_GPU_LOCK_NAME, GpuLock
    from scripts import run_bdd_video_rate_experiment as runner

    run = HERE / "uncapped_checks" / "9f322655eab14129a260aeb69166448d"
    fresh = run / "video"
    summary = json.loads((fresh / "summary.json").read_bytes())
    video = Path(summary["protocol"]["video_path"])
    vid = summary["protocol"]["video_id"]
    saved = ROOT / "runs" / "video_rate" / vid
    if runner.sha256(video) != summary["protocol"]["video_sha256"]:
        raise ValueError("Reference video differs")
    torch.set_num_threads(2)
    cv2.setNumThreads(4)
    if sys.argv[1:] == ["--repeatability"]:
        with GpuLock(CANONICAL_GPU_LOCK_NAME):
            compare_repeats(runner, video, summary["protocol"]["source_fps"], job.run_dir)
        return
    parent, parent_sha = load_revision(PARENT, "scripts/run_bdd_video_rate_experiment.py", "parent_video_runner")
    previous, old_sha = load_revision(BEFORE_ASSIGNMENT_FIX, "tracking/flyvis_tracker.py", "before_assignment_fix")
    # Both sides use the same CUDA device. The parent's numeric CLI selector failed in the original run.
    motion = job.run_dir / "motion"
    field_comparison = {}
    with GpuLock(CANONICAL_GPU_LOCK_NAME):
        for source in ("flyvis_full", "flyvis_tiled"):
            parent.build_motion_cache(source, video, motion, fps=summary["protocol"]["source_fps"],
                                      max_dt=.02, force=False, device="cuda:0")
            with zipfile.ZipFile(motion / (source + ".npz")) as a, zipfile.ZipFile(fresh / "motion" / (source + ".npz")) as b:
                field_comparison[source] = {key: compare_array(a, b, key) for key in ("points", "vectors")}
    detections = runner.load_detection_cache(fresh / "detections" / "raw_video.npz")
    truth = json.loads((runner.DATA_ROOT / "labels" / "box_track_20" / "val" / (vid + ".json")).read_bytes())
    matched = summary["alignment"]["matches"]
    outcomes = {}
    for name in ("constant_velocity", "flyvis_tiled_calibrated", "flyvis_full_legacy"):
        rel = Path("results") / name / (vid + ".json")
        reference, actual = [json.loads((d / rel).read_bytes()) for d in (saved, fresh)]
        def compare(rows):
            return {label: {"length_matches": len(rows) == len(other),
                            "different_frames": sum(left != right for left, right in zip(rows, other))}
                    for label, other in (("september", reference), ("fresh", actual))}
        results = {}
        for label, directory in (("september_fields", saved / "motion"), ("fresh_fields", fresh / "motion")):
            rows, _ = parent.run_tracker(name, detections, truth, matched, directory)
            results["parent_entry_current_assignment_" + label] = compare(rows)
            with patch.object(WeightedGeometryAssignment, "assign", previous.WeightedGeometryAssignment.assign):
                rows, _ = parent.run_tracker(name, detections, truth, matched, directory)
            results["parent_entry_previous_assignment_" + label] = compare(rows)
        outcomes[name] = results
    report = dict(run_id=run.name, parent_commit=PARENT, parent_runner_git_bytes_sha256=parent_sha,
                  previous_assignment_commit=BEFORE_ASSIGNMENT_FIX, previous_tracker_git_bytes_sha256=old_sha,
                  predecessor_fields_vs_fresh=field_comparison, tracking=outcomes)
    (job.run_dir / "diagnostic.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
