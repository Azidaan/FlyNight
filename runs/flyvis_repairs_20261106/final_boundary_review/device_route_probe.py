"""Trace the real CLI route up to model construction, without loading a model."""
from pathlib import Path
from unittest.mock import patch
import hashlib
import inspect
import json
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts import run_bdd_video_rate_experiment as module
from tracking.flyvis_motion import FlyVisMotionEstimator, GEOMETRY_VERSION, SYNTHETIC_FLOW_GAIN, resolve_flow_gain


class StopBeforeModel(Exception):
    pass


def main():
    out = Path(__file__).resolve().parent
    target = out / "device_route"
    target.mkdir(exist_ok=True)
    truth = target / "truth.json"
    truth.write_text('[{"name":"frame.jpg"}]', encoding="utf-8")
    calls = []

    def factory(*args, **kwargs):
        calls.append({"args": list(args), "kwargs": kwargs})
        raise StopBeforeModel

    argv = ["run_bdd_video_rate_experiment", "--device", "cpu", "--motion-only",
            "--motion-sources", "flyvis_full", "--output", str(target / "output")]
    with patch.object(sys, "argv", argv), \
            patch.object(module, "resolve_paths", return_value=(out / "synthetic_inputs/right.avi", truth, [])), \
            patch.object(module, "build_alignment", return_value=([{}], {"fps": 5., "decoded_frame_count": 5})), \
            patch.object(module, "build_detection_cache", return_value={}) as detector, \
            patch.object(module, "load_detection_cache", return_value={"frame_count": 5}), \
            patch.object(module, "make_motion_estimator", factory):
        try:
            module.main()
        except StopBeforeModel:
            pass
        else:
            raise AssertionError("No model-construction boundary reached")
    assert detector.call_args.kwargs["device"] == "cpu"
    assert len(calls) == 1 and "device" not in calls[0]["kwargs"]
    public_defaults = inspect.signature(FlyVisMotionEstimator).parameters
    assert public_defaults["geometry_version"].default == "legacy_v1" == GEOMETRY_VERSION
    assert resolve_flow_gain(None, "legacy_v1") == SYNTHETIC_FLOW_GAIN
    try:
        resolve_flow_gain(None, "resize_v2")
    except ValueError:
        research_rejected = True
    else:
        research_rejected = False
    import torch
    assert not torch.cuda.is_initialized()
    report = {
        "source_sha256": hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
        "argv": argv,
        "detector_requested_device": detector.call_args.kwargs["device"],
        "motion_factory_calls": calls,
        "motion_factory_effective_device_default": inspect.signature(module.make_motion_estimator).parameters["device"].default,
        "public_geometry_default": public_defaults["geometry_version"].default,
        "public_gain_default": public_defaults["flow_gain"].default,
        "public_resolved_legacy_gain": resolve_flow_gain(None, "legacy_v1"),
        "implicit_research_gain_rejected": research_rejected,
        "cuda_initialized": torch.cuda.is_initialized(),
        "scope": "No runtime activation; detection/alignment I/O mocked. Actual parse/main/build_motion_cache route traced. Device option scope needs owner decision.",
    }
    path = out / "device_route_report.json"
    assert not path.exists()
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
