"""Compare a complete fresh video run with the saved benchmark, in bounded blocks."""
from pathlib import Path
import json
import sys
import zipfile

from system_memory_guard import guard_entrypoint

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def compare_array(first, second, key):
    import numpy as np
    from numpy.lib.format import read_array_header_1_0, read_array_header_2_0, read_magic

    def header(stream):
        version = read_magic(stream)
        return {(1, 0): read_array_header_1_0, (2, 0): read_array_header_2_0}[version](stream)

    with first.open(key + ".npy") as left, second.open(key + ".npy") as right:
        shape, order, dtype = header(left)
        if (shape, order, dtype) != header(right):
            return {"same_layout": False}
        changed, maximum = 0, 0.0
        while True:
            a = np.frombuffer(left.read(65536 * dtype.itemsize), dtype=dtype)
            b = np.frombuffer(right.read(65536 * dtype.itemsize), dtype=dtype)
            if a.size != b.size:
                raise ValueError(f"Unequal payload lengths for {key}")
            if not a.size:
                break
            changed += int(np.count_nonzero(a != b))
            maximum = max(maximum, float(np.abs(a.astype(np.float64) - b).max()))
        return {"same_layout": True, "different_values": changed, "max_abs_difference": maximum}


def main():
    job = guard_entrypoint()
    run = (HERE / "uncapped_checks" / sys.argv[1]).resolve()
    if run.parent != (HERE / "uncapped_checks").resolve():
        raise ValueError("Invalid run path")
    new = run / "video"
    summary = json.loads((new / "summary.json").read_bytes())
    video_id = summary["protocol"]["video_id"]
    old = ROOT / "runs" / "video_rate" / video_id
    saved = json.loads((old / "summary.json").read_bytes())
    if saved["protocol"]["video_sha256"] != summary["protocol"]["video_sha256"]:
        raise ValueError("Reference video differs")
    caches = {"detections/raw_video.npz": ("frame_count", "frame_idx", "cls", "conf", "xyxy")}
    caches.update({f"motion/{name}.npz": ("points", "vectors") for name in summary["motion_sources"]})
    arrays = {}
    for name, fields in caches.items():
        with zipfile.ZipFile(old / name) as left, zipfile.ZipFile(new / name) as right:
            arrays[name] = {key: compare_array(left, right, key) for key in fields}

    trackers = {}
    for name in summary["trackers"]:
        rows = Path("results") / name / (video_id + ".json")
        previous = json.loads((old / rows).read_bytes())
        current = json.loads((new / rows).read_bytes())
        changed = [a["frameIndex"] for a, b in zip(previous, current) if a != b]
        old_metrics = saved["trackers"][name]["metrics"]
        new_metrics = summary["trackers"][name]["metrics"]
        trackers[name] = {"rows_identical": previous == current, "frames_old": len(previous),
                          "frames_new": len(current), "changed_frames": changed,
                          "metric_deltas": {k: new_metrics[k] - old_metrics[k] for k in
                                            ("HOTA", "DetA", "AssA", "IDF1", "MOTA", "IDSw", "CLR_FP", "CLR_FN", "Frag")}}
    report = {"run_id": run.name, "saved_reference": old.relative_to(ROOT).as_posix(), "arrays": arrays,
              "trackers": trackers}
    (job.run_dir / "comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({name: row["rows_identical"] for name, row in trackers.items()}), flush=True)


if __name__ == "__main__":
    main()
