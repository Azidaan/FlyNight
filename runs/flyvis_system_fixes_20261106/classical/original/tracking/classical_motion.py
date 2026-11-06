"""Классические источники поля движения для общего каркаса трекера."""

from __future__ import annotations

import time

import cv2
import numpy as np

from tracking.flyvis_tracker import MotionField


class OpenCVMotionEstimator:
    """Поток Farneback или DIS с выдачей того же ``MotionField``, что FlyVis."""

    def __init__(
        self,
        method: str,
        *,
        processing_scale: float = 1.0,
        sample_step: int = 8,
    ) -> None:
        method = method.lower()
        if method not in {"farneback", "dis"}:
            raise ValueError("method должен быть farneback или dis")
        if not 0 < processing_scale <= 1.0:
            raise ValueError("processing_scale должен находиться в диапазоне (0, 1]")
        if sample_step < 1:
            raise ValueError("sample_step должен быть положительным")
        self.method = method
        self.processing_scale = float(processing_scale)
        self.sample_step = int(sample_step)
        self._dis = (
            cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
            if method == "dis"
            else None
        )
        self.stats = {"observations": 0, "flow_seconds": 0.0}
        self.reset()

    def reset(self) -> None:
        self._previous: np.ndarray | None = None

    def _gray(self, frame: np.ndarray) -> np.ndarray:
        frame = np.asarray(frame)
        if frame.ndim == 3 and frame.shape[2] == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        elif frame.ndim == 2:
            gray = frame
        else:
            raise ValueError("Кадр должен иметь форму HxW или HxWx3")
        if gray.dtype != np.uint8:
            if not np.isfinite(gray).all():
                raise ValueError("Кадр содержит NaN или Inf")
            gray = np.clip(gray, 0, 255).astype(np.uint8)
        if self.processing_scale != 1.0:
            gray = cv2.resize(
                gray,
                None,
                fx=self.processing_scale,
                fy=self.processing_scale,
                interpolation=cv2.INTER_AREA,
            )
        return gray

    def _sample(self, flow: np.ndarray, source_width: int, source_height: int) -> MotionField:
        height, width = flow.shape[:2]
        offset = self.sample_step // 2
        ys = np.arange(offset, height, self.sample_step, dtype=np.int32)
        xs = np.arange(offset, width, self.sample_step, dtype=np.int32)
        if not len(xs) or not len(ys):
            ys = np.array([height // 2], dtype=np.int32)
            xs = np.array([width // 2], dtype=np.int32)
        grid_x, grid_y = np.meshgrid(xs, ys)
        sampled = flow[grid_y, grid_x].reshape(-1, 2).astype(np.float32)
        scale_x = source_width / width
        scale_y = source_height / height
        sampled[:, 0] *= scale_x
        sampled[:, 1] *= scale_y
        points = np.column_stack(
            [grid_x.reshape(-1) * scale_x, grid_y.reshape(-1) * scale_y]
        ).astype(np.float32)
        return MotionField(points, sampled)

    def observe(self, frame: np.ndarray) -> MotionField:
        source_height, source_width = np.asarray(frame).shape[:2]
        current = self._gray(frame)
        self.stats["observations"] += 1
        if self._previous is None:
            self._previous = current
            zero = np.zeros((*current.shape, 2), dtype=np.float32)
            return self._sample(zero, source_width, source_height)

        started = time.perf_counter()
        if self.method == "farneback":
            flow = cv2.calcOpticalFlowFarneback(
                self._previous,
                current,
                None,
                pyr_scale=0.5,
                levels=4,
                winsize=21,
                iterations=3,
                poly_n=7,
                poly_sigma=1.5,
                flags=0,
            )
        else:
            assert self._dis is not None
            flow = self._dis.calc(self._previous, current, None)
        self.stats["flow_seconds"] += time.perf_counter() - started
        self._previous = current
        if not np.isfinite(flow).all():
            raise RuntimeError(f"{self.method} вернул NaN или Inf")
        return self._sample(flow, source_width, source_height)

