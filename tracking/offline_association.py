"""Оффлайн-ассоциация ByteTrack на кэше детекций (Этап 8, подэтап 8.2).

Гоняет встроенный в Ultralytics BYTETracker по кэшу detection_cache.py —
без повторного прогона детектора. Одна конфигурация параметров на полном
val-поднаборе отрабатывает за секунды на CPU.

Выходной формат — scalabel JSON per video, идентичный
evaluation.tracking_metrics.run_tracker_on_videos: его понимает
evaluate_tracking_bdd без конвертации.

Соответствие Ultralytics model.track(): используется тот же класс
BYTETracker и та же двухпроходная логика update(); отличие только в
источнике детекций (кэш вместо живого predict). new_track_thresh
привязывается к track_high_thresh (как в конфигах Ultralytics, где
оба по умолчанию 0.25) — сетка плана варьирует track_high_thresh
и track_buffer.
"""
import json
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.bdd100k_loader import CLASS_NAMES  # noqa: E402
from tracking.detection_cache import iter_frames, load_video_cache  # noqa: E402


class _DetSlice:
    """Results-подобная обёртка кадра детекций для BYTETracker.update().

    BYTETracker ожидает объект с полями conf / cls / xywh, поддерживающий
    len() и индексацию булевой маской (см. _split_detections в
    ultralytics/trackers/byte_tracker.py).
    """

    def __init__(self, xyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray):
        self._xyxy = xyxy
        self.conf = conf
        self.cls = cls.astype(np.float32)

    @property
    def xywh(self) -> np.ndarray:
        xy = (self._xyxy[:, :2] + self._xyxy[:, 2:]) / 2
        wh = self._xyxy[:, 2:] - self._xyxy[:, :2]
        return np.concatenate([xy, wh], axis=1)

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, mask) -> '_DetSlice':
        return _DetSlice(self._xyxy[mask], self.conf[mask], self.cls[mask])


def make_tracker_args(
    track_high_thresh: float = 0.25,
    track_low_thresh: float = 0.1,
    track_buffer: int = 30,
    match_thresh: float = 0.8,
    fuse_score: bool = True,
    new_track_thresh: float | None = None,
) -> Namespace:
    """Namespace параметров BYTETracker (дефолты — bytetrack.yaml Ultralytics)."""
    return Namespace(
        track_high_thresh=track_high_thresh,
        track_low_thresh=track_low_thresh,
        new_track_thresh=(new_track_thresh if new_track_thresh is not None
                          else track_high_thresh),
        track_buffer=track_buffer,
        match_thresh=match_thresh,
        fuse_score=fuse_score,
    )


def associate_video(cache: dict, video_id: str, args: Namespace,
                    tracker_cls=None, q_key: str | None = None) -> list[dict]:
    """Ассоциация одного видео на кэше. Возвращает scalabel-кадры.

    tracker_cls — класс трекера (default BYTETracker; для Этапа 9 —
    LQAGuidedByteTrack). q_key — имя массива q в кэше (напр. 'q_mean');
    если задан, детекции кадра несут вектор q (нужен q-кэш inference_with_q).
    """
    if tracker_cls is None:
        from ultralytics.trackers.byte_tracker import BYTETracker
        tracker_cls = BYTETracker
    if q_key is not None and q_key not in cache:
        raise KeyError(f'В кэше нет {q_key!r} — нужен q-кэш (inference_with_q)')

    tracker = tracker_cls(args)
    fidx_all = cache['frame_idx']
    scalabel_frames = []
    for fidx, fname, xyxy, conf, cls in iter_frames(cache):
        if q_key is not None:
            from tracking.lqa_guided_tracker import _DetSliceQ
            dets = _DetSliceQ(xyxy, conf, cls, cache[q_key][fidx_all == fidx])
        else:
            dets = _DetSlice(xyxy, conf, cls)
        out = tracker.update(dets)  # (M, 8): x1,y1,x2,y2,id,score,cls,idx
        labels = []
        if len(out):
            for row in out:
                x1, y1, x2, y2, tid, score, c = row[:7]
                labels.append({
                    'id': str(int(tid)),
                    'category': CLASS_NAMES.get(int(c), f'class_{int(c)}'),
                    'score': float(score),
                    'box2d': {'x1': float(x1), 'y1': float(y1),
                              'x2': float(x2), 'y2': float(y2)},
                })
        scalabel_frames.append({
            'videoName': video_id, 'name': fname,
            'frameIndex': fidx, 'labels': labels,
        })
    return scalabel_frames


def run_association(
    cache_dir: str | Path,
    video_ids: list[str],
    out_dir: str | Path,
    split: str = 'val',
    per_video_args: dict[str, Namespace] | None = None,
    args: Namespace | None = None,
    tracker_cls=None,
    q_key: str | None = None,
) -> Path:
    """Ассоциация списка видео на кэше; пишет <out_dir>/<video>.json.

    Либо общий args на все видео, либо per_video_args (для адаптации
    по яркости или Control-Global, где у каждого видео свой порог).
    tracker_cls/q_key — для LQA-guided режима Этапа 9.
    """
    if (args is None) == (per_video_args is None):
        raise ValueError('Укажите ровно одно из: args | per_video_args')
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for vid in video_ids:
        out_json = out_dir / f'{vid}.json'
        if out_json.exists():
            continue
        cache = load_video_cache(cache_dir, split, vid)
        vid_args = per_video_args[vid] if per_video_args else args
        frames = associate_video(cache, vid, vid_args, tracker_cls, q_key)
        out_json.write_text(json.dumps(frames), encoding='utf-8')
    return out_dir
