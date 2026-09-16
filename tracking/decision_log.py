"""Журнал решений XYWHByteTrack для выборок событийной уверенности (цель 4).

Обвязка не меняет ни одного решения трекера: она записывает предсказания
треков до обновления, совпадения, рождения и переходы состояний с исходными
индексами детекций. Разметка сюда не передаётся.
"""

from __future__ import annotations

import numpy as np
from ultralytics.trackers.basetrack import TrackState

from tracking.confidence_audit import IndexedDetections
from tracking.xywh_bytetrack import XYWHByteTrack

STATE_NAMES = {TrackState.New: "new", TrackState.Tracked: "tracked", TrackState.Lost: "lost", TrackState.Removed: "removed"}

__all__ = ["AuditedXYWHByteTrack", "IndexedDetections", "STATE_NAMES"]


class AuditedXYWHByteTrack(XYWHByteTrack):
    """XYWHByteTrack с журналом: predictions (до обновления) и events (в ходе обновления)."""

    def __init__(self, args) -> None:
        super().__init__(args)
        self.events: list[dict] = []
        self.predictions: list[dict] = []
        self.stage = ""

    def update_with_log(self, results: IndexedDetections) -> dict:
        output = self.update(results)
        return {"frame_id": int(self.frame_id), "predictions": list(self.predictions), "events": list(self.events), "output": np.array(output, copy=True)}

    def update(self, results, img=None, feats=None, **kwargs):
        if not isinstance(results, IndexedDetections):
            raise TypeError("Журнал решений требует IndexedDetections с исходными индексами")
        self.events, self.predictions = [], []
        before = {int(t.track_id): t.state for t in self.tracked_stracks + self.lost_stracks}
        output = super().update(results, img, feats, **kwargs)
        now = {int(t.track_id): t for t in self.tracked_stracks + self.lost_stracks + self.removed_stracks}
        for tid, state in before.items():
            track = now.get(tid)
            if track is not None and track.state != state:
                self.events.append({"event": "state_transition", "track_id": tid, "from": STATE_NAMES[state], "to": STATE_NAMES[track.state]})
        consumed = {e["det_index"] for e in self.events if "det_index" in e}
        for idx, score in zip(results.indices, results.conf):
            if int(idx) not in consumed:
                reason = "score_filter" if score <= self.args.track_low_thresh else "unmatched_or_not_initialized"
                self.events.append({"event": "rejected", "det_index": int(idx), "score": float(score), "reason": reason})
        return output

    # --- предсказания до обновления -------------------------------------------------
    def multi_predict(self, tracks) -> None:
        super().multi_predict(tracks)
        frame = int(self.frame_id)
        for track in tracks:
            mean = np.asarray(track.mean, dtype=np.float64)
            self.predictions.append({
                "track_id": int(track.track_id),
                "xywh": mean[:4].tolist(),
                "velocity": mean[4:8].tolist(),
                "state": STATE_NAMES[track.state],
                "activated": bool(track.is_activated),
                "age": int(frame - track.start_frame),
                "hits": int(track.tracklet_len),
                "since_update": int(frame - track.frame_id),
                "last_score": float(track.score),
                "class_id": int(track.cls),
            })

    # --- события ассоциации ------------------------------------------------------
    def init_track(self, results, img=None):
        tracks = super().init_track(results, img)
        for track, idx in zip(tracks, results.indices):
            track.idx = int(idx)
        return tracks

    def _first_association(self, *args):
        self.stage = "first"
        return super()._first_association(*args)

    def _second_association(self, *args):
        self.stage = "second"
        return super()._second_association(*args)

    def _apply_match(self, track, det, activated, refind):
        event = "continued" if track.state == TrackState.Tracked else "reactivated"
        self.events.append({"event": event, "stage": self.stage, "track_id": int(track.track_id), "det_index": int(det.idx), "score": float(det.score)})
        return super()._apply_match(track, det, activated, refind)

    def _unconfirmed_association(self, unconfirmed, u_detection, detections, activated, removed):
        result = super()._unconfirmed_association(unconfirmed, u_detection, detections, activated, removed)
        for track in unconfirmed:
            if track.frame_id == self.frame_id and track.is_activated:
                self.events.append({"event": "confirmed", "track_id": int(track.track_id), "det_index": int(track.idx), "score": float(track.score)})
        return result

    def _init_new_tracks(self, u_detection, detections, activated, refind=None):
        before = {id(detections[i]) for i in u_detection}
        super()._init_new_tracks(u_detection, detections, activated, refind)
        for track in activated:
            if id(track) in before and track.state == TrackState.Tracked and track.start_frame == self.frame_id:
                self.events.append({"event": "created", "track_id": int(track.track_id), "det_index": int(track.idx), "score": float(track.score), "confirmed": bool(track.is_activated)})
