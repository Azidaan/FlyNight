"""LQA-guided ByteTrack: индивидуальный порог ассоциации по q_i (Этап 9, п. 9.3).

Формула плана (Часть 2.4):
    τ_i = τ_base − k · (1 − q_i),  k — коэффициент адаптации (default 0.2).

Механика ByteTrack (важно для текста диссертации, см. Часть 2.4 плана):
двухпроходная схема НЕ отбрасывает детекции ниже track_high_thresh — они
уходят во второй проход. Эффект адаптации: перераспределение детекций между
первым (надёжным) и вторым (слабым) проходами и порог инициации новых треков,
а не «спасение отброшенных детекций».

Реализация — сабкласс BYTETracker с двумя точками вмешательства:
  _split_detections: скалярный порог первого прохода заменяется вектором τ_i;
  _init_new_tracks:  порог инициации нового трека тоже индивидуальный
                     (q берётся по track.idx — индексу исходной детекции).

Путь 1 плана (модификация Ultralytics-трекера) выбран потому, что вся
инфраструктура оффлайн-ассоциации на кэше уже построена вокруг BYTETracker
(Этап 8) и эквивалентность live-пути доказана смоук-тестом.
"""
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics.trackers.byte_tracker import BYTETracker  # noqa: E402

from tracking.offline_association import _DetSlice  # noqa: E402


class _DetSliceQ(_DetSlice):
    """_DetSlice с вектором q_i на детекцию (для LQAGuidedByteTrack)."""

    def __init__(self, xyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray,
                 q: np.ndarray):
        super().__init__(xyxy, conf, cls)
        self.q = q.astype(np.float32)

    def __getitem__(self, mask) -> '_DetSliceQ':
        return _DetSliceQ(self._xyxy[mask], self.conf[mask], self.cls[mask],
                          self.q[mask])


class LQAGuidedByteTrack(BYTETracker):
    """ByteTrack с per-detection адаптацией порогов по карте качества Q.

    Параметры сверх BYTETracker: k — коэффициент адаптации (args.lqa_k).
    update() ожидает _DetSliceQ; без поля q работает как обычный ByteTrack.
    """

    def __init__(self, args):
        super().__init__(args)
        self.lqa_k = float(getattr(args, 'lqa_k', 0.2))
        self._frame_q: np.ndarray | None = None
        self._high_orig_idx: np.ndarray | None = None

    def update(self, results, img=None, feats=None, **kwargs):
        self._frame_q = getattr(results, 'q', None)
        self._high_orig_idx = None
        return super().update(results, img=img, feats=feats, **kwargs)

    def _tau(self, base: float, q: np.ndarray) -> np.ndarray:
        """Вектор индивидуальных порогов τ_i = base − k(1−q_i)."""
        return base - self.lqa_k * (1.0 - q)

    def _split_detections(self, results):
        """Первый проход: порог не скаляр, а вектор τ_i по q детекции.

        Дополнительно запоминает соответствие индексов: STrack.idx после
        init_track — это позиция в ОТФИЛЬТРОВАННОМ первом проходе
        (parse_bboxes нумерует arange-ом уже после маски), а _frame_q
        индексируется по исходному кадру.
        """
        if getattr(results, 'q', None) is None:
            return super()._split_detections(results)
        scores = results.conf
        tau_i = self._tau(self.args.track_high_thresh, results.q)
        remain_inds = scores >= tau_i
        inds_low = scores > self.args.track_low_thresh
        inds_below_high = ~remain_inds
        self._high_orig_idx = np.where(remain_inds)[0]
        return (results[remain_inds], results[inds_low & inds_below_high],
                remain_inds, inds_low & inds_below_high)

    def _init_new_tracks(self, u_detection, detections, activated,
                         refind=None):
        """Инициация новых треков: порог тоже индивидуальный по q_i."""
        if self._frame_q is None or self._high_orig_idx is None:
            return super()._init_new_tracks(u_detection, detections,
                                            activated, refind)
        for inew in u_detection:
            track = detections[inew]
            orig_idx = int(self._high_orig_idx[int(track.idx)])
            q_i = float(self._frame_q[orig_idx])
            thresh_i = self.args.new_track_thresh - self.lqa_k * (1.0 - q_i)
            if track.score < thresh_i:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated.append(track)
