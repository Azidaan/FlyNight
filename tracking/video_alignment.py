"""Сопоставление исходных видеокадров с прореженными кадрами BDD100K MOT."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class FrameMatch:
    reference_index: int
    reference_name: str
    video_frame_index: int
    mean_absolute_error: float
    second_best_error: float

    def to_dict(self) -> dict:
        return asdict(self)


def compact_frame(frame: np.ndarray, width: int = 160, height: int = 90) -> np.ndarray:
    """Получить малое цветное представление для устойчивого поиска кадра."""
    value = np.asarray(frame)
    if value.ndim == 2:
        value = cv2.cvtColor(value, cv2.COLOR_GRAY2BGR)
    if value.ndim != 3 or value.shape[2] != 3:
        raise ValueError("Ожидается серый или трёхканальный кадр")
    return cv2.resize(value, (width, height), interpolation=cv2.INTER_AREA)


def decode_compact_video(path: str | Path) -> tuple[list[np.ndarray], dict]:
    """Декодировать видео в малые кадры и вернуть параметры контейнера."""
    path = Path(path)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {path}")
    metadata = {
        "path": str(path.resolve()),
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": float(capture.get(cv2.CAP_PROP_FPS)),
        "container_frame_count": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(compact_frame(frame))
    capture.release()
    metadata["decoded_frame_count"] = len(frames)
    return frames, metadata


def load_compact_images(paths: list[Path]) -> list[np.ndarray]:
    frames = []
    for path in paths:
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f"Не удалось прочитать изображение: {path}")
        frames.append(compact_frame(frame))
    return frames


def align_reference_frames(
    video_frames: list[np.ndarray],
    reference_frames: list[np.ndarray],
    reference_names: list[str] | None = None,
    *,
    search_radius: int = 24,
) -> list[FrameMatch]:
    """Найти точный видеокадр для каждого прореженного JPEG по содержимому.

    Линейная позиция используется только как центр небольшого окна поиска.
    Итог выбирается по минимальной средней абсолютной разности пикселей и
    принимается лишь при строго возрастающей последовательности соответствий.
    """
    if not video_frames or not reference_frames:
        raise ValueError("Обе последовательности должны содержать кадры")
    if search_radius < 1:
        raise ValueError("search_radius должен быть положительным")
    if reference_names is None:
        reference_names = [str(index) for index in range(len(reference_frames))]
    if len(reference_names) != len(reference_frames):
        raise ValueError("Число имён не совпало с числом эталонных кадров")
    shape = video_frames[0].shape
    if any(frame.shape != shape for frame in video_frames + reference_frames):
        raise ValueError("Все малые кадры должны иметь одинаковую форму")

    matches = []
    video_last = len(video_frames) - 1
    reference_last = max(len(reference_frames) - 1, 1)
    for index, (name, reference) in enumerate(zip(reference_names, reference_frames)):
        expected = int(round(index * video_last / reference_last))
        lower = max(0, expected - search_radius)
        upper = min(len(video_frames), expected + search_radius + 1)
        costs = np.asarray(
            [
                float(cv2.absdiff(reference, video_frames[candidate]).mean())
                for candidate in range(lower, upper)
            ],
            dtype=np.float64,
        )
        order = np.argsort(costs)
        best_local = int(order[0])
        second_local = int(order[1]) if len(order) > 1 else best_local
        matches.append(
            FrameMatch(
                reference_index=index,
                reference_name=name,
                video_frame_index=lower + best_local,
                mean_absolute_error=float(costs[best_local]),
                second_best_error=float(costs[second_local]),
            )
        )

    indices = np.asarray([match.video_frame_index for match in matches], dtype=int)
    if len(indices) > 1 and not np.all(np.diff(indices) > 0):
        raise RuntimeError(
            "Содержательное сопоставление не монотонно; увеличивать окно без проверки нельзя"
        )
    return matches
