"""Export actual assignments and pre-association state, checking archived parity."""
import os
import sys
import time
from collections import deque
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = "2"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research_spatial_identity_20261107.run_common import RUN, read, sha, write, status, verified_protocol, guard


def export(video, selected):
    import numpy as np
    import torch
    from ultralytics.trackers.utils import matching
    from ultralytics.trackers.basetrack import BaseTrack
    from tracking.xywh_bytetrack import XYWHByteTrack
    from tracking.offline_association import _DetSlice, make_tracker_args
    from tracking.foveated_eyes import crop_box, box_to_crop_frame
    from tracking.z13_readout import HexGeometry

    folder = RUN / "tracker"
    dest = folder / (video + ".json")
    if dest.exists():
        report = read(dest)
        assert report["passed"]
        for name, h in report["sha256"].items():
            assert sha(ROOT / name) == h
        return report
    started = time.perf_counter()
    raw = ROOT / "runs/train_cache" / video / "detections/raw_video.npz"
    episode_file = ROOT / "runs/z13_data" / (video + ".npz")
    archived = read(ROOT / "runs/z13_temporal_pair_readiness_20261025/protocol.json")["sha256"]
    for source in (raw, episode_file):
        assert sha(source) == archived[source.relative_to(ROOT).as_posix()]
    with np.load(raw, allow_pickle=False) as z:
        boxes, confidence, classes, frame_ids = [z[k] for k in ("xyxy", "conf", "cls", "frame_idx")]
    with np.load(episode_file, allow_pickle=False) as z:
        d = {k: z[k] for k in ("ep_tid", "ep_offset", "fr_frame", "fr_crop", "ex_mask", "lab_ep", "lab_pos", "lab_pred")}
    source_meta = ROOT / "runs/z13_temporal_pair_readiness_20261025/inputs" / (video + ".json")
    metadata = {r["uid"]: r for r in read(source_meta)["rows"]}
    geometry = read(ROOT / "runs/z13_local_peak_20261021/geometry.json")
    geom = HexGeometry(np.asarray(geometry["points"], np.float32), geometry["eye_size"])
    by_frame = {}
    for r in selected:
        by_frame.setdefault(r["current_frame"], []).append(r)
    history, outputs, matches_by_frame, matrices = {}, {}, {}, {}
    context = {"frame": -1}

    class Detections(_DetSlice):
        def __init__(self, b, s, c, raw_ids):
            super().__init__(b, s, c)
            self.raw_ids = np.asarray(raw_ids)

        def __getitem__(self, mask):
            return Detections(self._xyxy[mask], self.conf[mask], self.cls[mask], self.raw_ids[mask])

    class ExportTracker(XYWHByteTrack):
        def init_track(self, results, img=None):
            tracks = super().init_track(results, img)
            for t, index in zip(tracks, results.raw_ids):
                t.export_raw_id = int(index)
            return tracks

        def _pre_first_associate(self, pool, unconfirmed, img, high):
            super()._pre_first_associate(pool, unconfirmed, img, high)
            frame = context["frame"]
            lookup = {int(t.track_id): t for t in pool}
            for r in by_frame.get(frame, []):
                m = metadata[r["uid"]]
                assert m["tid"] in lookup, (r["uid"], "missing native track")
                t = lookup[m["tid"]]
                h = list(history.get(m["tid"], []))
                assert h
                h = [h[0]] * max(0, 4 - len(h)) + h[-4:]
                e = r["episode"]
                first = int(d["ep_offset"][e])
                label = np.flatnonzero((d["lab_ep"] == e) & (d["lab_pos"] == r["current_array_index"] - first - 4))
                assert len(label) == 1
                current_crop = crop_box(tuple(t.mean[:4]), (1280, 720))
                checks = {
                    "current_crop": list(current_crop) == m["current_crop"],
                    "observed_frames": [x[0] for x in h] == d["fr_frame"][first:first+4].tolist(),
                    "observed_crops": [list(x[2]) for x in h] == d["fr_crop"][first:first+4].tolist(),
                    "observed_mask": bool(np.array_equal(geom.mask_in_box(box_to_crop_frame(h[-1][1], h[-1][2], geom.eye_size)), d["ex_mask"][e])),
                }
                center_difference = float(np.max(np.abs(t.mean[:2] - d["lab_pred"][label[0]])))
                assert all(checks.values()) and center_difference <= 1e-4, (video, r["uid"], checks, center_difference)
                high_ids = high.raw_ids
                assert len(high_ids) == int(((frame_ids == frame) & (confidence >= .25)).sum())
                outputs[r["uid"]] = {
                    "uid": r["uid"], "video": video, "frame": frame, "tid": m["tid"],
                    "episode": e, "split": r["split"], "mean": t.mean.tolist(), "covariance": t.covariance.tolist(),
                    "history": [{"frame": x[0], "box": x[1].tolist(), "crop": list(x[2])} for x in h],
                    "current_crop": list(current_crop), "checks": checks, "center_difference": center_difference,
                    "candidate_high_ids": m["current_choices"],
                    "candidate_raw_ids": [int(high_ids[j]) for j in m["current_choices"]],
                    "candidate_boxes": [boxes[high_ids[j]].astype(float).tolist() for j in m["current_choices"]],
                }

        def _first_association(self, pool, detections, activated, refind):
            frame = context["frame"]
            if frame not in by_frame:
                return super()._first_association(pool, detections, activated, refind)
            # Same matrix, solver and application as the native implementation; retain the inputs and outputs.
            cost = self.get_dists(pool, detections)
            matches, ut, ud = matching.linear_assignment(cost, thresh=self.args.match_thresh)
            snapshots = {"track_ids": [int(t.track_id) for t in pool],
                         "track_classes": [int(t.cls) for t in pool],
                         "track_states": [int(t.state) for t in pool],
                         "detection_raw_ids": [int(t.export_raw_id) for t in detections],
                         "matches": np.asarray(matches).reshape(-1, 2).tolist(),
                         "unmatched_tracks": list(map(int, ut)), "unmatched_detections": list(map(int, ud)),
                         "threshold": float(self.args.match_thresh)}
            matrices["cost_" + str(frame)] = cost.copy()
            matrices["meta_" + str(frame)] = snapshots
            self._apply_matches(matches, pool, detections, activated, refind)
            return ut, ud

        def _apply_matches(self, matches, tracks, detections, activated, refind):
            if context["frame"] in by_frame:
                for i, j in matches:
                    matches_by_frame[(context["frame"], int(tracks[i].track_id))] = int(detections[j].export_raw_id)
            return super()._apply_matches(matches, tracks, detections, activated, refind)

    BaseTrack.reset_id()
    tracker = ExportTracker(make_tracker_args(track_high_thresh=.25, track_low_thresh=.1, track_buffer=30, match_thresh=.8))
    order = np.argsort(frame_ids, kind="stable")
    ordered_frames = frame_ids[order]
    for f in range(max(by_frame) + 1):
        context["frame"] = f
        lo, hi = np.searchsorted(ordered_frames, [f, f+1])
        ids = order[lo:hi]
        tracker.update(Detections(boxes[ids], confidence[ids], classes[ids], ids))
        for t in tracker.tracked_stracks:
            if t.frame_id == tracker.frame_id:
                history.setdefault(int(t.track_id), deque(maxlen=4)).append(
                    (f, np.asarray(t.xyxy, np.float64).copy(), crop_box(tuple(t.mean[:4]), (1280, 720))))
        for r in by_frame.get(f, []):
            row = outputs[r["uid"]]
            row["native_assignment_raw_id"] = matches_by_frame.get((f, row["tid"]))
            snapshot = matrices["meta_" + str(f)]
            assigned = {snapshot["detection_raw_ids"][j]: snapshot["track_ids"][i] for i, j in snapshot["matches"]}
            row["candidate_assigned_track_ids"] = [assigned.get(i) for i in row["candidate_raw_ids"]]
            row["association_snapshot_frame"] = f
    folder.mkdir(parents=True, exist_ok=True)
    npz = folder / (video + ".npz")
    assert not npz.exists()
    with npz.open("xb") as stream:
        np.savez_compressed(stream, **{k: v for k, v in matrices.items() if k.startswith("cost_")})
    snapshots = {k.removeprefix("meta_"): v for k, v in matrices.items() if k.startswith("meta_")}
    report = {"passed": True, "video": video, "rows": [outputs[r["uid"]] for r in selected],
              "snapshots": snapshots, "seconds": time.perf_counter()-started,
              "sha256": {str(p.relative_to(ROOT).as_posix()): sha(p) for p in (raw, episode_file, source_meta, npz)}}
    write(dest, report, new=True)
    assert not torch.cuda.is_initialized()
    return report


def main(obs):
    import torch
    torch.set_num_threads(2)
    verified_protocol()
    rows = read(ROOT / "research_spatial_identity_20261107/readiness_rows.json")
    by_video = {}
    for r in rows:
        by_video.setdefault(r["video"], []).append(r)
    # Resolve the two questioned cases first, then all other videos without outcome selection.
    order = ["0001542f-ec815219"] + sorted(set(by_video) - {"0001542f-ec815219"})
    receipts = {}
    for i, video in enumerate(order):
        report = export(video, by_video[video])
        receipts[video] = {"sha256": sha(RUN/"tracker"/(video+".json")), "cases": len(report["rows"])}
        status("tracker_export", complete=i+1, total=len(order), video=video, guard=obs.read())
        print("tracker", i+1, len(order), video, len(report["rows"]), round(report["seconds"], 2), flush=True)
    verified_protocol()
    write(RUN / "tracker_export_completed.json", {"passed": True, "videos": receipts, "cases": len(rows)}, new=True)


if __name__ == "__main__":
    main(guard())
