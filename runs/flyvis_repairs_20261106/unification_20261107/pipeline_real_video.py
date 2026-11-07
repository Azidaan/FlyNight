"""Run the normal BDD100K video CLI with its defaults in a new, fingerprinted directory."""
from pathlib import Path
import json
import math
import sys

from run_fingerprint import ROOT, begin, finish
from system_memory_guard import guard_entrypoint


def main():
    job = guard_entrypoint(timeout=1800)
    sys.path.insert(0, str(ROOT))
    from scripts import run_bdd_video_rate_experiment as runner
    from tracking.gpu_lock import CANONICAL_GPU_LOCK_NAME, GpuLock
    import cv2
    import torch

    output = job.run_dir / "video"
    video_id = runner.DEFAULT_VIDEO_ID
    sys.argv = [runner.__file__, "--video-id", video_id, "--output", str(output)]
    args = runner.parse_args()
    video, labels, images = runner.resolve_paths(args)
    torch.set_num_threads(2)
    cv2.setNumThreads(4)
    protocol = begin(job.run_dir, video, args.weights, dict(
        cli=[str(arg) for arg in sys.argv], parameters=vars(args) | {"weights": str(args.weights), "output": str(output)},
        torch_threads=torch.get_num_threads(), opencv_threads=cv2.getNumThreads()),
        launcher=Path(__file__).name, arguments=(), extra_inputs=[labels, *images], result_file="pipeline_real_video.json")

    with GpuLock(CANONICAL_GPU_LOCK_NAME):
        if not torch.cuda.is_available():
            raise RuntimeError("The default video CLI requires CUDA device 0")
        if runner.main() != 0:
            raise RuntimeError("Video pipeline failed")

    summary = json.loads((output / "summary.json").read_bytes())
    raw_frames = summary["protocol"]["raw_frames"]
    if raw_frames <= 0 or summary["protocol"]["labeled_frames"] != len(images):
        raise ValueError("Video and reference counts differ")
    if summary["protocol"]["video_sha256"] != protocol["input_sha256"][video.relative_to(ROOT).as_posix()]:
        raise ValueError("Processed video differs from the recorded input")
    if set(summary["motion_sources"]) != set(args.motion_sources) or set(summary["trackers"]) != set(args.trackers):
        raise ValueError("Missing motion sources or tracker results")
    for source in summary["motion_sources"].values():
        if source["frame_count"] != raw_frames:
            raise ValueError("Motion field length differs from the video")
    metrics = {}
    reference_frames = json.loads(labels.read_bytes())
    for name, record in summary["trackers"].items():
        rows = json.loads((output / "results" / name / f"{video_id}.json").read_bytes())
        if len(rows) != len(reference_frames) or any(
            row["name"] != ref["name"] or row["frameIndex"] != ref["frameIndex"]
            for row, ref in zip(rows, reference_frames)
        ):
            raise ValueError(f"Misaligned output for {name}")
        if not any(row["labels"] for row in rows):
            raise ValueError(f"Empty tracker output for {name}")
        metrics[name] = {key: record["metrics"][key] for key in ("HOTA", "AssA", "DetA", "IDF1", "MOTA")}
        if not all(math.isfinite(value) for value in metrics[name].values()):
            raise ValueError(f"Non-finite metrics for {name}")
    report = dict(passed=True, video_id=video_id, raw_frames=raw_frames, reference_frames=len(images),
                  detections=summary["detector"]["detections"], motion_sources=args.motion_sources,
                  metrics=metrics, gpu=summary["detector"]["software"]["gpu"])
    (job.run_dir / "pipeline_real_video.json").write_text(json.dumps(report, indent=2, allow_nan=False),
                                                          encoding="utf-8")
    finish(job.run_dir, protocol, extra_output_files=(p for p in output.rglob("*") if p.is_file()),
           scope="Default video CLI with fresh detection and motion caches, tracking and evaluation on one full validation video.")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
