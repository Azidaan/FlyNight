"""Агрегация карты качества Q в скаляр q_i по рамке детекции (Этап 9, п. 9.2).

Формула плана (Часть 2.3): q_i — среднее значение Q^{l_i} внутри рамки
детекции, где l_i — уровень пирамиды, к которому привязана детекция.
Альтернативы для ablation: минимум и медиана.

Координаты: рамки приходят в пикселях ОРИГИНАЛЬНОГО изображения; карта Q
живёт в координатах letterbox-входа сети, уменьшенных в stride раз.
Проекция повторяет ultralytics LetterBox: gain = min(in/orig), паддинг
распределяется поровну (сверху/снизу, слева/справа).
"""
import numpy as np
import torch

AGGREGATIONS = ('mean', 'min', 'median')


def box_to_qmap_coords(
    box_xyxy: np.ndarray,
    orig_shape: tuple[int, int],
    input_shape: tuple[int, int],
    stride: float,
) -> tuple[int, int, int, int]:
    """Проецирует рамку из пикселей оригинала в ячейки карты Q.

    orig_shape, input_shape — (H, W) оригинала и letterbox-входа сети.
    Возвращает (x1, y1, x2, y2) в ячейках Q, гарантированно непустые
    (минимум 1x1) и обрезанные по границам карты.
    """
    oh, ow = orig_shape
    ih, iw = input_shape
    gain = min(ih / oh, iw / ow)
    pad_x = (iw - ow * gain) / 2
    pad_y = (ih - oh * gain) / 2

    x1 = (box_xyxy[0] * gain + pad_x) / stride
    y1 = (box_xyxy[1] * gain + pad_y) / stride
    x2 = (box_xyxy[2] * gain + pad_x) / stride
    y2 = (box_xyxy[3] * gain + pad_y) / stride

    qh, qw = int(round(ih / stride)), int(round(iw / stride))
    ix1 = max(0, min(qw - 1, int(np.floor(x1))))
    iy1 = max(0, min(qh - 1, int(np.floor(y1))))
    ix2 = max(ix1 + 1, min(qw, int(np.ceil(x2))))
    iy2 = max(iy1 + 1, min(qh, int(np.ceil(y2))))
    return ix1, iy1, ix2, iy2


def aggregate_q(q_map: torch.Tensor, cells: tuple[int, int, int, int],
                aggregation: str = 'mean') -> float:
    """Скаляр q_i из области карты Q (тензор (1,1,H,W) или (H,W))."""
    if q_map.dim() == 4:
        q_map = q_map[0, 0]
    x1, y1, x2, y2 = cells
    region = q_map[y1:y2, x1:x2]
    if region.numel() == 0:
        return 1.0  # пустая область — нейтральное качество, порог не трогаем
    if aggregation == 'mean':
        return float(region.mean())
    if aggregation == 'min':
        return float(region.min())
    if aggregation == 'median':
        return float(region.median())
    raise ValueError(f'aggregation: {AGGREGATIONS}, получено {aggregation!r}')


def compute_q_per_detection(
    q_maps_by_stride: dict[int, torch.Tensor],
    boxes_xyxy: np.ndarray,
    strides: np.ndarray,
    orig_shape: tuple[int, int],
    input_shape: tuple[int, int],
    aggregation: str = 'mean',
) -> np.ndarray:
    """q_i для каждой детекции кадра.

    q_maps_by_stride: {8: Q_P3, 16: Q_P4, 32: Q_P5} (тензоры (1,1,H,W));
    boxes_xyxy: (N,4) в пикселях оригинала; strides: (N,) страйд анкера
    каждой детекции (привязка к уровню — из inference_with_q).
    """
    out = np.ones(len(boxes_xyxy), dtype=np.float32)
    for i, (box, s) in enumerate(zip(boxes_xyxy, strides)):
        q_map = q_maps_by_stride.get(int(s))
        if q_map is None:
            continue  # модель без DRB на этом уровне — нейтральное q=1
        cells = box_to_qmap_coords(box, orig_shape, input_shape, float(s))
        out[i] = aggregate_q(q_map, cells, aggregation)
    return out
