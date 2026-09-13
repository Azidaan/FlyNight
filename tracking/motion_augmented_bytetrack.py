"""ByteTrack с контролируемой поправкой прогноза внешним полем движения."""

from __future__ import annotations

import numpy as np
from ultralytics.trackers.basetrack import TrackState
from ultralytics.trackers.byte_tracker import BYTETracker

from tracking.flyvis_tracker import MotionField


class MotionAugmentedByteTrack(BYTETracker):
    """Смешать прогноз Калмана с наблюдаемым оптическим движением рамки.

    ``blend`` задаёт долю внешнего вектора. При ``use_confidence`` эта доля
    дополнительно умножается на пространственную согласованность поля.
    Режим ``lost`` изменяет только потерянные траектории и тем самым проверяет
    узкую гипотезу о восстановлении ID после частичной окклюзии.
    """

    def __init__(
        self,
        args,
        *,
        blend: float,
        apply_to: str = "all",
        use_confidence: bool = True,
        context_scale: float = 2.0,
        min_motion_samples: int = 3,
    ) -> None:
        super().__init__(args)
        if not 0.0 <= blend <= 1.0:
            raise ValueError("blend должен находиться в диапазоне [0, 1]")
        if apply_to not in {"all", "tracked", "lost"}:
            raise ValueError("apply_to должен быть all, tracked или lost")
        self.blend = float(blend)
        self.apply_to = apply_to
        self.use_confidence = bool(use_confidence)
        self.context_scale = float(context_scale)
        self.min_motion_samples = int(min_motion_samples)
        self.motion_field = MotionField.empty()
        self.motion_audit: list[dict] = []

    def set_motion_field(self, field: MotionField) -> None:
        self.motion_field = field

    def _eligible(self, track) -> bool:
        if self.apply_to == "all":
            return True
        if self.apply_to == "tracked":
            return track.state == TrackState.Tracked
        return track.state == TrackState.Lost

    def multi_predict(self, tracks) -> None:
        if not tracks:
            return
        centers_before = [track.mean[:2].copy() for track in tracks]
        boxes_before = [track.xyxy.copy() for track in tracks]
        super().multi_predict(tracks)
        for track, center_before, box_before in zip(
            tracks, centers_before, boxes_before
        ):
            if not self._eligible(track):
                continue
            estimate = self.motion_field.estimate_box(
                box_before,
                context_scale=self.context_scale,
                min_samples=self.min_motion_samples,
            )
            weight = self.blend
            if self.use_confidence:
                weight *= float(np.clip(estimate.confidence, 0.0, 1.0))
            kalman_displacement = track.mean[:2] - center_before
            blended = (
                (1.0 - weight) * kalman_displacement
                + weight * estimate.vector_xy
            )
            track.mean[:2] = center_before + blended
            self.motion_audit.append(
                {
                    "frame": int(self.frame_id),
                    "track_id": int(track.track_id),
                    "state": int(track.state),
                    "weight": float(weight),
                    "confidence": float(estimate.confidence),
                    "kalman_dx": float(kalman_displacement[0]),
                    "kalman_dy": float(kalman_displacement[1]),
                    "motion_dx": float(estimate.vector_xy[0]),
                    "motion_dy": float(estimate.vector_xy[1]),
                }
            )
