"""Инференс с пробросом карты качества Q и привязкой детекций к уровням (Этап 9, п. 9.1).

Две технические проблемы плана и их решения:

1. «model.track()/predict() не возвращает Q». DRBWrappedLayer уже сохраняет
   last_q при каждом forward; здесь после predict() карты забираются через
   collect_q_maps и сопоставляются уровням по head.f/head.stride.

2. «Привязка детекции к уровню пирамиды теряется при декодировании».
   Голова YOLO26 (end2end) выбирает топ-K анкеров в Detect.get_topk_index;
   индексы анкеров там есть, но наружу не отдаются. Метод патчится обёрткой,
   сохраняющей их в head._last_anchor_idx. Страйд каждого анкера известен
   из head.strides (заполняется make_anchors) — уровень детекции получается
   ТОЧНО, без эвристик по размеру рамки.

   Предиктор Ultralytics для end2end-головы только фильтрует выход по conf,
   порядок сохраняется — соответствие детекций анкор-индексам
   восстанавливается той же маской score > conf.

Разрешение входа: predictor использует прямоугольный letterbox (например
384x640 для BDD 720x1280); маска FGA при этом билинейно интерполируется
(models/drb/fga.py). Реальный размер входа восстанавливается из формы
Q-карты, проекция координат рамок это учитывает.

Кэш: build_detection_cache_with_q пишет npz, совместимый с
tracking/detection_cache.py, плюс массивы q_<agg> и stride.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.bdd100k_loader import DATA_ROOT  # noqa: E402
from models.yolo_neck import collect_q_maps, has_drb  # noqa: E402
from tracking.detection_cache import CACHE_CONF, CACHE_IOU  # noqa: E402
from tracking.q_aggregation import AGGREGATIONS, compute_q_per_detection  # noqa: E402
from tracking.cache_integrity import (atomic_save_detection_arrays, ensure_cache_identity,
    frame_inventory, load_detection_arrays, validate_detection_arrays,
    verify_frame_inventory, verify_weights, weights_identity)

IMGSZ = 640


def _patch_topk_capture(head) -> None:
    """Обёртка get_topk_index: сохраняет анкор-индексы последнего вызова."""
    if getattr(head, '_topk_patched', False):
        return
    original = head.get_topk_index

    def wrapped(scores, max_det):
        s, c, idx = original(scores, max_det)
        head._last_anchor_idx = idx.detach()
        return s, c, idx

    head.get_topk_index = wrapped
    head._topk_patched = True


class QInference:
    """Инференс YOLO(+DRB): детекции + q_i + страйд уровня каждой детекции."""

    def __init__(self, model_path: str, conf: float = CACHE_CONF,
                 iou: float = CACHE_IOU, device: str | int = 0):
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.det_model = self.model.model
        self.head = self.det_model.model[-1]
        self.conf = conf
        self.iou = iou
        self.device = device
        self.with_drb = has_drb(self.det_model)
        _patch_topk_capture(self.head)
        # соответствие индекс слоя -> страйд (по порядку head.f и head.stride)
        self.layer_to_stride = {
            int(f): int(s) for f, s in zip(self.head.f, self.head.stride)}

    def q_maps_by_stride(self) -> dict[int, torch.Tensor]:
        """Карты Q последнего forward, ключ — страйд уровня."""
        return {self.layer_to_stride[idx]: q
                for idx, q in collect_q_maps(self.det_model).items()
                if idx in self.layer_to_stride}

    def predict_frame(self, img_path: str | Path,
                      aggregations: tuple[str, ...] = ('mean',)) -> dict:
        """Один кадр. Возвращает dict: xyxy, conf, cls, stride, q_<agg>...

        Рамки — в пикселях оригинала. stride — страйд уровня пирамиды
        каждой детекции. q_* присутствуют только если модель содержит DRB.
        """
        res = self.model.predict(source=str(img_path), conf=self.conf,
                                 iou=self.iou, imgsz=(IMGSZ, IMGSZ),
                                 device=self.device, verbose=False)[0]
        boxes = res.boxes
        n = len(boxes)
        out = {
            'xyxy': boxes.xyxy.cpu().numpy().astype(np.float32),
            'conf': boxes.conf.cpu().numpy().astype(np.float32),
            'cls': boxes.cls.cpu().numpy().astype(np.int16),
        }

        # Привязка к уровню: анкор-индексы топ-K -> conf-маска предиктора.
        # Предиктор end2end оставляет строки score > conf, сохраняя порядок
        # (topk отсортирован по убыванию score) — первые n строк топ-K
        # соответствуют детекциям Results один к одному.
        anchor_idx = self.head._last_anchor_idx  # (1, K, 1)
        anchor_strides = self.head.strides.flatten()  # страйд каждого анкера
        topk_strides = anchor_strides[anchor_idx[0, :, 0]].cpu().numpy()
        out['stride'] = topk_strides[:n].astype(np.int16)

        if self.with_drb and n:
            q_maps = self.q_maps_by_stride()
            orig_shape = tuple(res.orig_shape)          # (H, W) оригинала
            # Реальный размер letterbox-входа восстанавливается из формы
            # Q-карты: (H_q * stride, W_q * stride). Захардкоженный (640,640)
            # неверен: predictor использует прямоугольный letterbox
            # (напр. 384x640 для кадров BDD 720x1280).
            s0, q0 = next(iter(q_maps.items()))
            input_shape = (q0.shape[-2] * s0, q0.shape[-1] * s0)
            for agg in aggregations:
                out[f'q_{agg}'] = compute_q_per_detection(
                    q_maps, out['xyxy'], out['stride'], orig_shape,
                    input_shape, agg)
        elif self.with_drb:
            for agg in aggregations:
                out[f'q_{agg}'] = np.empty(0, np.float32)
        return out


def build_detection_cache_with_q(
    model_path: str,
    video_ids: list[str],
    split: str = 'val',
    cache_dir: str | Path = PROJECT_ROOT / 'runs' / 'det_cache_q',
    aggregations: tuple[str, ...] = tuple(AGGREGATIONS),
    device: str | int = 0,
) -> Path:
    """Кэш детекций с q_i — формат detection_cache.py + q_<agg> + stride.

    Совместим с tracking.offline_association (лишние массивы игнорируются)
    и используется LQA-guided трекером и режимом Control-Global (Этап 9).

    device='cpu' считает мимо видеокарты — медленнее, но не мешает идущему
    обучению. Обучения этап 9 не требует, детектор зафиксирован.
    """
    cache_dir = Path(cache_dir)
    out_dir = cache_dir / split
    out_dir.mkdir(parents=True, exist_ok=True)
    img_root = DATA_ROOT / 'images' / 'track' / split

    identity = weights_identity(model_path)
    meta = {**identity, 'cache_schema': 'detection_q_jpg_v2', 'conf': CACHE_CONF, 'iou': CACHE_IOU,
            'aggregations': list(aggregations), 'imgsz': IMGSZ,
            'data_root': str(DATA_ROOT.resolve()),
            'date': time.strftime('%Y-%m-%d %H:%M:%S')}
    ensure_cache_identity(cache_dir, meta)

    qi = None

    for i, vid in enumerate(video_ids, 1):
        out_npz = out_dir / f'{vid}.npz'
        frames = sorted((img_root / vid).glob('*.jpg'))
        if not frames:
            raise ValueError(f'No JPG frames found for {vid}')
        inventory = frame_inventory(frames)
        if out_npz.exists():
            cached = load_detection_arrays(out_npz, required=('stride',))
            verify_frame_inventory(cached, inventory)
            available = cached.get('quality_available')
            if available is None or available.shape != () or available.dtype != np.bool_:
                raise ValueError('Q cache does not declare whether the model provides quality maps')
            if bool(available):
                validate_detection_arrays(cached, required=tuple(f'q_{a}' for a in aggregations))
            continue
        if qi is None:
            qi = QInference(model_path, device=device)
            if not qi.with_drb:
                print('[WARN] модель без DRB — q_i не будут посчитаны (нейтральные 1.0)')
        per_frame: list[dict] = []
        t0 = time.time()
        for f in frames:
            per_frame.append(qi.predict_frame(f, aggregations))

        arrays: dict[str, np.ndarray] = {
            **inventory, 'quality_available': np.asarray(qi.with_drb, dtype=np.bool_)}
        keys = ['xyxy', 'conf', 'cls', 'stride'] + [
            f'q_{a}' for a in aggregations if qi.with_drb]
        fidx = [np.full(len(pf['conf']), k, dtype=np.int32)
                for k, pf in enumerate(per_frame)]
        arrays['frame_idx'] = (np.concatenate(fidx) if fidx
                               else np.zeros(0, np.int32))
        for key in keys:
            chunks = [pf[key] for pf in per_frame if key in pf]
            if chunks:
                arrays[key] = np.concatenate(chunks)
        verify_weights(model_path, identity)
        verify_frame_inventory(frame_inventory(frames), inventory)
        validate_detection_arrays(arrays, required=('stride', *(f'q_{a}' for a in aggregations if qi.with_drb)))
        atomic_save_detection_arrays(out_npz, arrays)
        n_det = len(arrays.get('conf', []))
        print(f'[INFO] q-кэш {i}/{len(video_ids)}: {vid} '
              f'({len(frames)} кадров, {n_det} детекций, {time.time()-t0:.1f}s)',
              flush=True)
    return out_dir
