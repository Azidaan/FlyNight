"""ByteTrack с состоянием рамки ``центр, ширина, высота`` из BoT-SORT."""

from __future__ import annotations

from ultralytics.trackers.bot_sort import BOTrack, parse_bboxes
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.trackers.utils.kalman_filter import KalmanFilterXYWH


class XYWHByteTrack(BYTETracker):
    """Сохранить ассоциацию ByteTrack, заменив только модель состояния рамки.

    Обычный ByteTrack использует ``x, y, aspect, height`` (центр, отношение
    сторон, высота). Здесь используются независимые ``x, y, width, height``
    (центр, ширина, высота), как в BoT-SORT. Компенсация камеры и повторная
    идентификация по внешности отсутствуют.
    """

    def get_kalmanfilter(self) -> KalmanFilterXYWH:
        return KalmanFilterXYWH()

    def init_track(self, results, img=None) -> list[BOTrack]:
        if len(results) == 0:
            return []
        bboxes = parse_bboxes(results)
        return [
            BOTrack(xywh, score, class_id)
            for xywh, score, class_id in zip(bboxes, results.conf, results.cls)
        ]

    def multi_predict(self, tracks: list[BOTrack]) -> None:
        BOTrack.multi_predict(tracks)


class XYWHIoUSecondByteTrack(XYWHByteTrack):
    """XYWHByteTrack с чистым IoU на второй ассоциации.

    Первая ассоциация остаётся штатной: расстояние по IoU объединяется с
    confidence детекции, если ``fuse_score=True``. На второй ассоциации уже
    отобраны низкоуверенные детекции, поэтому повторное умножение сходства на
    confidence способно отвергнуть даже геометрически идеальное продолжение.
    Здесь только для этого этапа ``fuse_score`` временно отключается.
    """

    def _second_association(self, *args) -> None:
        saved_fuse_score = self.args.fuse_score
        try:
            self.args.fuse_score = False
            return super()._second_association(*args)
        finally:
            self.args.fuse_score = saved_fuse_score
