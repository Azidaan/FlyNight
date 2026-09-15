"""Изолированная обвязка BoT-SORT для эталонных экспериментов.

GT и правильные ID сюда не передаются. Вход вмешательства — только индексы
разрешённых кандидатов. Установленный пакет и штатные скрипты не меняются.
"""
from __future__ import annotations

import numpy as np

from ultralytics.trackers.basetrack import BaseTrack
from ultralytics.trackers.bot_sort import BOTSORT
from ultralytics.trackers.utils.stracks import multi_gmc

from tracking.offline_association import _DetSlice


class IndexedDetections(_DetSlice):
    def __init__(self, boxes, scores, classes, indices=None):
        super().__init__(boxes, scores, classes)
        self.indices = np.arange(len(scores)) if indices is None else np.asarray(indices)

    def __getitem__(self, mask):
        return IndexedDetections(self._xyxy[mask], self.conf[mask],
                                  self.cls[mask], self.indices[mask])


class AuditedBoTSORT(BOTSORT):
    def __init__(self, args, *, second_iou_only=False):
        super().__init__(args)
        if args.with_reid:
            raise ValueError("Этот диагностический протокол не включает ReID")
        self.second_iou_only = second_iou_only
        self.allowed = set()
        self.events = []
        self.stage = ""
        self.warp = None
        self.id_counter = 0

    def update(self, results, img=None, *, allowed=(), warp=None):
        self.allowed = set(map(int, allowed))
        self.events = []
        self.warp = warp
        # Несколько вариантов могут идти по одному кадру последовательно.
        BaseTrack._count = self.id_counter
        before = {t.track_id: (t.state, t.is_activated)
                  for t in self.tracked_stracks + self.lost_stracks}
        output = super().update(results, img)
        self.id_counter = BaseTrack._count
        now = {t.track_id: t for t in
               self.tracked_stracks + self.lost_stracks + self.removed_stracks}
        for tid, (state, activated) in before.items():
            if tid in now and now[tid].state != state:
                self.events.append({"event": "state_transition", "track_id": tid,
                                    "from": int(state), "to": int(now[tid].state)})
        consumed = {e["det_index"] for e in self.events if "det_index" in e}
        for idx, score in zip(results.indices, results.conf):
            if int(idx) not in consumed:
                reason = ("score_filter" if score <= self.args.track_low_thresh
                          and int(idx) not in self.allowed else "unmatched_or_not_initialized")
                self.events.append({"event": "rejected", "det_index": int(idx),
                                    "score": float(score), "reason": reason})
        return output

    def _split_detections(self, results):
        high = (results.conf >= self.args.track_high_thresh) | np.isin(results.indices, list(self.allowed))
        low = (results.conf > self.args.track_low_thresh) & ~high
        return results[high], results[low], high, low

    def init_track(self, results, img=None):
        tracks = super().init_track(results, img)
        for track, idx in zip(tracks, results.indices):
            track.idx = int(idx)
        return tracks

    def _pre_first_associate(self, pool, unconfirmed, img, results_high):
        if self.warp is None:
            return super()._pre_first_associate(pool, unconfirmed, img, results_high)
        multi_gmc(pool, self.warp)
        multi_gmc(unconfirmed, self.warp)

    def _first_association(self, *args):
        self.stage = "first"
        return super()._first_association(*args)

    def _second_association(self, *args):
        self.stage = "second"
        saved = self.args.fuse_score
        try:
            if self.second_iou_only:
                self.args.fuse_score = False
            return super()._second_association(*args)
        finally:
            self.args.fuse_score = saved

    def _apply_match(self, track, det, activated, refind):
        event = "continued" if track.is_activated and track.state == 1 else "reactivated"
        self.events.append({"event": event, "stage": self.stage,
                            "track_id": int(track.track_id), "det_index": int(det.idx),
                            "score": float(det.score)})
        return super()._apply_match(track, det, activated, refind)

    def _unconfirmed_association(self, unconfirmed, u_detection, detections, activated, removed):
        result = super()._unconfirmed_association(unconfirmed, u_detection, detections, activated, removed)
        for track in unconfirmed:
            if track.frame_id == self.frame_id and track.is_activated:
                self.events.append({"event": "confirmed", "track_id": int(track.track_id),
                                    "det_index": int(track.idx), "score": float(track.score)})
        return result

    def _init_new_tracks(self, u_detection, detections, activated, refind=None):
        # Отличие от штатного метода только в разрешении выбранного кандидата.
        for i in u_detection:
            track = detections[i]
            if track.score < self.args.new_track_thresh and int(track.idx) not in self.allowed:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated.append(track)
            self.events.append({"event": "created", "track_id": int(track.track_id),
                                "det_index": int(track.idx), "score": float(track.score),
                                "confirmed": bool(track.is_activated)})
