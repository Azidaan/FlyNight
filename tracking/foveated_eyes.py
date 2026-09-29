"""Фовеация: глаз FlyVis на каждый трек (карточка flyvis_foveation_readout_v1).

У каждого подтверждённого трека своё рекуррентное состояние сети и своя
вырезка вокруг предсказанной рамки. Модуль не читает разметку и не меняет
состояние трекера: предсказание передаётся снаружи.
"""

from __future__ import annotations

import numpy as np

from tracking.flyvis_activity import OUTPUT_CELL_TYPES, type_column_indices


def crop_box(pred_xywh, frame_size: tuple[int, int], *, context: float = 2.0, min_side: int = 64) -> tuple[int, int, int, int]:
    """Квадрат со стороной max(context·max(w, h), min_side), сдвинутый внутрь кадра."""
    cx, cy, w, h = (float(v) for v in pred_xywh)
    width, height = frame_size
    side = int(round(max(context * max(w, h), float(min_side))))
    side = min(side, width, height) if side > min(width, height) else side
    x1 = int(round(cx - side / 2.0))
    y1 = int(round(cy - side / 2.0))
    x1 = min(max(x1, 0), max(width - side, 0))
    y1 = min(max(y1, 0), max(height - side, 0))
    return x1, y1, min(x1 + side, width), min(y1 + side, height)


def box_to_crop_frame(box_xyxy, crop: tuple[int, int, int, int], eye_size: int) -> tuple[float, float, float, float]:
    """Перевести рамку кадра в пиксели поля глаза (eye_size × eye_size) данной вырезки."""
    x1, y1, x2, y2 = crop
    sx = eye_size / max(x2 - x1, 1)
    sy = eye_size / max(y2 - y1, 1)
    bx1, by1, bx2, by2 = (float(v) for v in box_xyxy)
    return ((bx1 - x1) * sx, (by1 - y1) * sy, (bx2 - x1) * sx, (by2 - y1) * sy)


def original_detection_index(conf: np.ndarray, *, subset_index: int, score: float, high: float, low: float) -> int:
    """Индекс детекции в исходном кадре по индексу внутри подмножества ByteTrack и её баллу."""
    conf = np.asarray(conf)
    if score >= high:
        subset = np.flatnonzero(conf >= high)
    else:
        subset = np.flatnonzero((conf > low) & (conf < high))
    return int(subset[int(subset_index)])


class EyeBank:
    """Батч глаз FlyVis с переменным составом; состояния режутся и склеиваются по track_id."""

    def __init__(self, estimator, *, context: float = 2.0, min_side: int = 64, fade_in_seconds: float = 1.0, decode_flow: bool = False, max_batch: int = 0) -> None:
        self.max_batch = int(max_batch)  # 0 — все глаза одним батчем; иначе порциями (память сети ∝ числу глаз, результат тот же)
        self.estimator = estimator
        self.decode_flow = bool(decode_flow)
        self._baseline_flow: dict[int, "torch.Tensor"] = {}
        self._last_flow: dict[int, np.ndarray] = {}
        self.torch = estimator.torch
        self.network = estimator.network
        self.eye = estimator.eye
        self.eye_size = int(estimator.eye_height)
        assert int(estimator.eye_width) == self.eye_size
        self.context = float(context)
        self.min_side = int(min_side)
        self.fade_in_seconds = float(fade_in_seconds)
        self.dt = float(estimator.dt)
        self.steps = int(estimator.integration_steps)
        self.indices = type_column_indices(self.network.connectome.nodes.type[:], OUTPUT_CELL_TYPES)
        self._index_matrix = np.stack(list(self.indices.values()))  # (types, columns)
        centers = estimator._local_centers_yx
        self.points_local = np.column_stack([centers[:, 1] + self.eye_size // 2, centers[:, 0] + self.eye_size // 2]).astype(np.float32)
        self._activity: dict[int, "torch.Tensor"] = {}
        self._last_hex: dict[int, "torch.Tensor"] = {}
        self._baseline: dict[int, np.ndarray] = {}
        self.crops: dict[int, tuple[int, int, int, int]] = {}
        self.stats = {"births": 0, "steps": 0, "fade_in_seconds_total": 0.0}

    @property
    def track_ids(self) -> list[int]:
        return list(self._activity)

    def baseline(self, track_id: int) -> np.ndarray:
        return self._baseline[track_id]

    def last_flow(self, track_id: int) -> np.ndarray:
        """Остаточный декодированный поток глаза (hexals, 2) в единицах декодера, минус базовый после fade-in."""
        return self._last_flow[track_id]

    def _decode(self, activity_steps: "torch.Tensor") -> "torch.Tensor":
        """(B, steps, nodes) -> (B, 2, hexals): среднее декодированное поле по шагам."""
        with self.torch.inference_mode():
            decoded = self.estimator.decoder(activity_steps)  # (B, steps, 2, hexals)
        return decoded.mean(dim=1)

    def _render(self, gray: np.ndarray, crops: list[tuple[int, int, int, int]]):
        import cv2

        images = []
        for x1, y1, x2, y2 in crops:
            patch = gray[y1:y2, x1:x2]
            resized = cv2.resize(patch, (self.eye_size, self.eye_size), interpolation=cv2.INTER_AREA)
            images.append(resized.astype(np.float32) / 255.0)
        cartesian = self.torch.from_numpy(np.stack(images, axis=0)).unsqueeze(1)
        cartesian = cartesian.to(self.estimator.flyvis.device, dtype=self.torch.float32)
        return self.eye(cartesian)[:, 0:1]  # (B, 1, hexals)

    def _by_type(self, activity_steps: "torch.Tensor") -> np.ndarray:
        """(steps, nodes) -> (types, columns) float32, среднее по шагам."""
        mean = activity_steps.mean(dim=0).detach().cpu().numpy().astype(np.float32)
        return mean[self._index_matrix]

    def _gray(self, frame: np.ndarray) -> np.ndarray:
        import cv2

        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame

    def birth(self, frame: np.ndarray, boxes_xywh: dict[int, tuple]) -> None:
        for chunk in self._chunks(list(boxes_xywh)):
            self._birth({tid: boxes_xywh[tid] for tid in chunk}, frame)

    def _chunks(self, ids: list) -> list[list]:
        if self.max_batch <= 0 or len(ids) <= self.max_batch:
            return [ids] if ids else []
        return [ids[k : k + self.max_batch] for k in range(0, len(ids), self.max_batch)]

    def _fade_in(self, initial_frames: "torch.Tensor"):
        """Как `Network.fade_in_state`, но без списка состояний всех шагов разгона: хранится только текущее.

        Оригинал держит полное состояние сети (с синапсами) на каждом из int(fade_in/dt) шагов —
        память растёт как шаги × глаза × синапсы (до 9+ ГБ на населённых роликах). Результат тот же (тест).
        """
        torch = self.torch
        net = self.network
        steps = int(self.fade_in_seconds / self.dt)
        batch = initial_frames.shape[0]
        net.stimulus.zero(batch, steps)
        frames = torch.linspace(0, 1, steps)[None, :, None] * (initial_frames.repeat(1, steps, 1) - 0.5) + 0.5
        net.stimulus.add_input(frames[:, :, None])
        with torch.inference_mode():
            x = net.stimulus()
            net.clamp()
            params = net._param_api()
            state = net._initial_state(params, batch)
            for i in range(x.shape[1]):
                state = net._next_state(params, state, x[:, i], self.dt)
        return state

    def _birth(self, boxes_xywh: dict[int, tuple], frame: np.ndarray) -> None:
        """Родить глаза на кадре frame по текущим рамкам треков (fade-in на вырезке)."""
        if not boxes_xywh:
            return
        gray = self._gray(frame)
        ids = list(boxes_xywh)
        crops = [crop_box(boxes_xywh[tid], (gray.shape[1], gray.shape[0]), context=self.context, min_side=self.min_side) for tid in ids]
        hexes = self._render(gray, crops)
        with self.torch.inference_mode():
            state = self._fade_in(hexes[:, 0])
        activity = state.nodes.activity  # (B, nodes)
        base_flow = self._decode(activity.unsqueeze(1)) if self.decode_flow else None
        for row, tid in enumerate(ids):
            self._activity[tid] = activity[row].clone()
            self._last_hex[tid] = hexes[row].clone()
            self._baseline[tid] = self._by_type(activity[row : row + 1])
            self.crops[tid] = crops[row]
            if base_flow is not None:
                self._baseline_flow[tid] = base_flow[row].clone()
                self._last_flow[tid] = np.zeros((base_flow.shape[-1], 2), np.float32)
        self.stats["births"] += len(ids)
        self.stats["fade_in_seconds_total"] += self.fade_in_seconds * len(ids)

    def step(self, frame: np.ndarray, predictions_xywh: dict[int, tuple]) -> dict[int, np.ndarray]:
        ids = [tid for tid in predictions_xywh if tid in self._activity]
        result: dict[int, np.ndarray] = {}
        for chunk in self._chunks(ids):
            result.update(self._step(frame, {tid: predictions_xywh[tid] for tid in chunk}))
        return result

    def _step(self, frame: np.ndarray, predictions_xywh: dict[int, tuple]) -> dict[int, np.ndarray]:
        """Продвинуть существующие глаза на кадр frame к предсказанным рамкам; вернуть активность (types, columns)."""
        ids = [tid for tid in predictions_xywh if tid in self._activity]
        if not ids:
            return {}
        gray = self._gray(frame)
        crops = [crop_box(predictions_xywh[tid], (gray.shape[1], gray.shape[0]), context=self.context, min_side=self.min_side) for tid in ids]
        new_hex = self._render(gray, crops)  # (B, 1, hexals)
        last_hex = self.torch.stack([self._last_hex[tid] for tid in ids])  # (B, 1, hexals)
        alpha = self.torch.arange(1, self.steps + 1, device=new_hex.device, dtype=self.torch.float32).view(1, self.steps, 1, 1) / self.steps
        stimulus = last_hex[:, None, 0] * (1.0 - alpha) + new_hex[:, None, 0] * alpha  # (B, steps, 1, hexals)
        from flyvis.utils.tensor_utils import AutoDeref

        initial = self.network._state_api(AutoDeref(nodes=AutoDeref(activity=self.torch.stack([self._activity[tid] for tid in ids])), edges=AutoDeref()))
        with self.torch.inference_mode():
            states = self.network.simulate(stimulus, self.dt, initial_state=initial, as_states=True)
        activity = self.torch.stack([state.nodes.activity for state in states], dim=1)  # (B, steps, nodes)
        flow = self._decode(activity) if self.decode_flow else None
        result = {}
        for row, tid in enumerate(ids):
            self._activity[tid] = activity[row, -1].clone()
            self._last_hex[tid] = new_hex[row].clone()
            self.crops[tid] = crops[row]
            result[tid] = self._by_type(activity[row])
            if flow is not None:
                residual = (flow[row] - self._baseline_flow[tid]).detach().cpu().numpy().astype(np.float32)  # (2, hexals)
                self._last_flow[tid] = residual.T.copy()
        self.stats["steps"] += len(ids)
        return result

    def remove(self, track_ids) -> None:
        for tid in list(track_ids):
            self._activity.pop(tid, None)
            self._last_hex.pop(tid, None)
            self._baseline.pop(tid, None)
            self.crops.pop(tid, None)
            self._baseline_flow.pop(tid, None)
            self._last_flow.pop(tid, None)
