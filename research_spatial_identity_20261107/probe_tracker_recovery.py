"""Recover missing tracker fields on one fixed video and check archived parity."""
from __future__ import annotations

import json
import os
import sys
import time
from collections import deque
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = "2"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from research_spatial_identity_20261107.audit_readiness import digest, read, save_new, OUT


def main():
    started = time.perf_counter()
    assert not (OUT / "tracker_recovery_probe.json").exists()
    plan = read(OUT / "tracker_recovery_probe_protocol.json")
    archive_rows = read(OUT / "readiness_rows.json")
    video = plan["video"]
    selected = [r for r in archive_rows if r["video"] == video and r["split"] == "evaluate26"]
    assert selected
    raw_path = ROOT / "runs/train_cache" / video / "detections/raw_video.npz"
    episode_path = ROOT / "runs/z13_data" / (video + ".npz")
    sources = [raw_path, episode_path, Path(__file__), ROOT / "tracking/xywh_bytetrack.py",
               ROOT / ".venv/Lib/site-packages/ultralytics/trackers/byte_tracker.py",
               ROOT / ".venv/Lib/site-packages/ultralytics/trackers/bot_sort.py"]
    fingerprints = {p.relative_to(ROOT).as_posix(): digest(p) for p in sources}
    with np.load(raw_path, allow_pickle=False) as z:
        boxes, conf, classes, frame_ids = [z[k] for k in ["xyxy", "conf", "cls", "frame_idx"]]
    with np.load(episode_path, allow_pickle=False) as z:
        d = {k: z[k] for k in ["ep_tid", "ep_offset", "fr_frame", "fr_crop", "ex_mask",
                               "lab_ep", "lab_pos", "lab_pred"]}
    meta_rows = read(ROOT / "runs/z13_temporal_pair_readiness_20261025/inputs" / (video + ".json"))["rows"]
    metadata = {r["uid"]: r for r in meta_rows}
    by_frame = {}
    for r in selected:
        by_frame.setdefault(r["current_frame"], []).append(r)

    import torch
    from ultralytics.trackers.basetrack import BaseTrack, TrackState
    from tracking.xywh_bytetrack import XYWHByteTrack
    from tracking.offline_association import _DetSlice, make_tracker_args
    from tracking.foveated_eyes import crop_box, box_to_crop_frame
    from tracking.z13_readout import HexGeometry

    torch.set_num_threads(2)
    g = read(ROOT / "runs/z13_local_peak_20261021/geometry.json")
    geom = HexGeometry(np.asarray(g["points"], np.float32), g["eye_size"])
    frame_context = {"frame": -1}
    histories = {}
    recovered = {}
    native_matches = {}

    class IndexedDetections(_DetSlice):
        def __init__(self, xyxy, scores, class_ids, raw_ids):
            super().__init__(xyxy, scores, class_ids)
            self.raw_ids = np.asarray(raw_ids)

        def __getitem__(self, mask):
            return IndexedDetections(self._xyxy[mask], self.conf[mask], self.cls[mask], self.raw_ids[mask])

    class CaptureTracker(XYWHByteTrack):
        def init_track(self, results, img=None):
            tracks = super().init_track(results, img)
            for t, idx in zip(tracks, results.raw_ids):
                t.audit_raw_id = int(idx)
            return tracks

        def _pre_first_associate(self, pool, unconfirmed, img, results_high):
            super()._pre_first_associate(pool, unconfirmed, img, results_high)
            frame = frame_context["frame"]
            lookup = {int(t.track_id): t for t in pool}
            for r in by_frame.get(frame, []):
                m = metadata[r["uid"]]
                tid = m["tid"]
                if tid not in lookup:
                    recovered[r["uid"]] = {"uid": r["uid"], "missing_track": tid}
                    continue
                t = lookup[tid]
                h = list(histories.get(tid, []))
                if h:
                    h = [h[0]] * max(0, 4 - len(h)) + h[-4:]
                episode = r["episode"]
                first = int(d["ep_offset"][episode])
                current_crop = crop_box(tuple(map(float, t.mean[:4])), (1280, 720))
                li = np.flatnonzero((d["lab_ep"] == episode) &
                                    (d["lab_pos"] == r["current_array_index"] - first - 4))
                assert len(li) == 1
                pred_error = float(np.max(np.abs(t.mean[:2] - d["lab_pred"][li[0]])))
                prefix_frames_ok = [x[0] for x in h] == d["fr_frame"][first:first + 4].tolist()
                prefix_crops_ok = [list(x[2]) for x in h] == d["fr_crop"][first:first + 4].tolist()
                mask_ok = bool(h) and np.array_equal(
                    geom.mask_in_box(box_to_crop_frame(h[-1][1], h[-1][2], geom.eye_size)), d["ex_mask"][episode])
                recovered[r["uid"]] = {"uid": r["uid"], "tid": tid, "frame": frame,
                    "mean_xywh_velocity": t.mean.tolist(), "covariance": t.covariance.tolist(),
                    "observed_history": [{"frame": x[0], "box_xyxy": x[1].tolist(), "crop": list(x[2])} for x in h],
                    "current_crop_exact": list(current_crop) == m["current_crop"],
                    "observed_frames_exact": prefix_frames_ok, "observed_crops_exact": prefix_crops_ok,
                    "observed_mask_exact": mask_ok, "predicted_center_max_difference": pred_error}

        def _apply_matches(self, matches, tracks, detections, activated, refind):
            frame = frame_context["frame"]
            for i, j in matches:
                native_matches[(frame, int(tracks[i].track_id))] = int(detections[j].audit_raw_id)
            return super()._apply_matches(matches, tracks, detections, activated, refind)

    BaseTrack.reset_id()
    tracker = CaptureTracker(make_tracker_args(track_high_thresh=.25, track_low_thresh=.1,
                                               track_buffer=30, match_thresh=.8))
    last = max(by_frame)
    order = np.argsort(frame_ids, kind="stable")
    sorted_frames = frame_ids[order]
    for frame in range(last + 1):
        if time.perf_counter() - started > plan["wall_limit_seconds"]:
            raise TimeoutError("Bounded one-video recovery probe exceeded its limit")
        lo, hi = np.searchsorted(sorted_frames, [frame, frame + 1])
        ids = order[lo:hi]
        frame_context["frame"] = frame
        tracker.update(IndexedDetections(boxes[ids], conf[ids], classes[ids], ids))
        for t in tracker.tracked_stracks:
            if t.frame_id == tracker.frame_id:
                histories.setdefault(int(t.track_id), deque(maxlen=4)).append(
                    (frame, np.asarray(t.xyxy, np.float64).copy(),
                     crop_box(tuple(map(float, t.mean[:4])), (1280, 720))))
        for r in by_frame.get(frame, []):
            rec = recovered[r["uid"]]
            raw_id = native_matches.get((frame, metadata[r["uid"]]["tid"]))
            high = ids[conf[ids] >= .25]
            rec["native_assignment_raw_detection"] = raw_id
            where = np.flatnonzero(high == raw_id) if raw_id is not None else []
            rec["native_assignment_high_index"] = int(where[0]) if len(where) else None
            rec["native_assignment_is_one_of_candidates"] = bool(len(where) and int(where[0]) in r["detector_candidates"])
    rows = [recovered[r["uid"]] for r in selected]
    checks = ["current_crop_exact", "observed_frames_exact", "observed_crops_exact", "observed_mask_exact"]
    parity = all(all(r.get(k, False) for k in checks) and
                 r.get("predicted_center_max_difference", float("inf")) <= plan["center_tolerance_px"] for r in rows)
    assert not torch.cuda.is_initialized()
    assert all(digest(ROOT / name) == h for name, h in fingerprints.items())
    save_new("tracker_recovery_probe.json", {"complete": True, "video": video,
        "cases": len(rows), "parity_passed": parity, "checks": checks,
        "center_tolerance_px": plan["center_tolerance_px"], "rows": rows,
        "source_sha256": fingerprints, "seconds": time.perf_counter() - started,
        "gpu_initialized": False, "scope": "One-video execution parity only, no scientific comparison or HOTA."})
    print(json.dumps({"cases": len(rows), "parity_passed": parity,
                      "seconds": time.perf_counter() - started}), flush=True)


if __name__ == "__main__":
    main()
