"""Потоковый источник поля движения на основе официального FlyVis.

Поддерживаются полный кадр и фиксированная сетка перекрывающихся фрагментов.
Сетка фиксируется по первому кадру видео; это важно, чтобы движение области
вырезания не создавало искусственный оптический поток.
"""

from __future__ import annotations

import time
import math
from dataclasses import dataclass

import cv2
import numpy as np

from tracking.flyvis_adapter import activate_flyvis
from tracking.flyvis_tracker import MotionField


# Оценено только на синтетических переносах 5--20 пикселей, без BDD100K.
# Это начальный физический масштаб, а не подобранный по метрикам трекера параметр.
SYNTHETIC_FLOW_GAIN = 102.59663505027247


@dataclass(frozen=True)
class TileLayout:
    rows: int = 2
    columns: int = 3
    overlap: float = 0.20

    def __post_init__(self) -> None:
        if self.rows < 1 or self.columns < 1:
            raise ValueError("Число строк и столбцов должно быть положительным")
        if not 0.0 <= self.overlap < 1.0:
            raise ValueError("Перекрытие должно находиться в диапазоне [0, 1)")


def integration_plan(source_fps: float, max_dt: float) -> tuple[int, float]:
    """Разбить один интервал видео на устойчивые шаги FlyVis.

    ``max_dt`` является верхней границей шага интегрирования, а не требованием
    к частоте исходного видео. Это позволяет обрабатывать настоящий поток
    29.97/30 Гц без увеличения шага выше рекомендованных FlyVis 0.02 с.
    Возвращаемый шаг всегда точно покрывает один интервал исходного кадра.
    """
    if source_fps <= 0 or max_dt <= 0:
        raise ValueError("source_fps и max_dt должны быть положительными")
    source_interval = 1.0 / float(source_fps)
    steps = max(1, int(math.ceil(source_interval / float(max_dt) - 1e-12)))
    return steps, source_interval / steps


def fixed_tile_bounds(
    width: int, height: int, layout: TileLayout
) -> list[tuple[int, int, int, int]]:
    """Получить покрывающую кадр фиксированную сетку ``x1,y1,x2,y2``."""
    tile_width = width / (layout.columns - (layout.columns - 1) * layout.overlap)
    tile_height = height / (layout.rows - (layout.rows - 1) * layout.overlap)
    stride_x = tile_width * (1.0 - layout.overlap)
    stride_y = tile_height * (1.0 - layout.overlap)
    bounds = []
    for row in range(layout.rows):
        for column in range(layout.columns):
            x1 = int(round(column * stride_x))
            y1 = int(round(row * stride_y))
            x2 = width if column == layout.columns - 1 else int(round(x1 + tile_width))
            y2 = height if row == layout.rows - 1 else int(round(y1 + tile_height))
            x1 = max(0, min(x1, width - 1))
            y1 = max(0, min(y1, height - 1))
            x2 = max(x1 + 1, min(x2, width))
            y2 = max(y1 + 1, min(y2, height))
            bounds.append((x1, y1, x2, y2))
    return bounds


class FlyVisMotionEstimator:
    """Сохранять внутреннее состояние FlyVis между кадрами одного видео."""

    def __init__(
        self,
        *,
        model: str = "flow/0000/000",
        mode: str = "full",
        layout: TileLayout | None = None,
        source_fps: float = 5.0,
        dt: float = 0.02,
        flow_gain: float = SYNTHETIC_FLOW_GAIN,
        fade_in_seconds: float = 1.0,
        temporal_resampling: str = "hold",
        flow_aggregation: str = "mean",
        subtract_initial_baseline: bool = True,
    ) -> None:
        if mode not in {"full", "tiled"}:
            raise ValueError("mode должен быть full или tiled")
        if source_fps <= 0 or dt <= 0 or flow_gain <= 0:
            raise ValueError("source_fps, dt и flow_gain должны быть положительными")
        if temporal_resampling not in {"hold", "linear"}:
            raise ValueError("temporal_resampling должен быть hold или linear")
        if flow_aggregation not in {"mean", "last", "sum"}:
            raise ValueError("flow_aggregation должен быть mean, last или sum")
        self.integration_steps, integration_dt = integration_plan(source_fps, dt)

        self.mode = mode
        self.layout = layout or TileLayout()
        self.source_fps = float(source_fps)
        self.requested_max_dt = float(dt)
        self.dt = float(integration_dt)
        self.flow_gain = float(flow_gain)
        self.fade_in_seconds = float(fade_in_seconds)
        self.temporal_resampling = temporal_resampling
        self.flow_aggregation = flow_aggregation
        self.subtract_initial_baseline = bool(subtract_initial_baseline)

        self.flyvis = activate_flyvis()
        self.torch = __import__("torch")
        view = self.flyvis.NetworkView(model)
        self.network = view.init_network()
        self.decoder = view.init_decoder()["flow"]
        self.network.eval().requires_grad_(False)
        self.decoder.eval().requires_grad_(False)

        extent = int(view.dir.config.network.connectome.extent)
        kernel_size = int(view.dir.config.task.dataset.boxfilter.kernel_size)
        self.kernel_area = kernel_size**2
        from flyvis.datasets.rendering import BoxEye

        self.eye = BoxEye(extent=extent, kernel_size=kernel_size)
        self.eye.conv = self.eye.conv.to(self.flyvis.device)
        self.eye_height, self.eye_width = (
            int(value) for value in self.eye.min_frame_size.tolist()
        )
        self._local_centers_yx = self.eye.receptor_centers.detach().cpu().numpy().astype(
            np.float32
        )
        self.stats = {
            "observations": 0,
            "render_seconds": 0.0,
            "network_seconds": 0.0,
            "decoder_seconds": 0.0,
        }
        self.reset()

    def reset(self) -> None:
        self._last_hex = None
        self._state = None
        self.last_decoded = None
        self.last_activity = None
        self.baseline_activity = None
        self._baseline_flow = None
        self._frame_size: tuple[int, int] | None = None
        self._bounds: list[tuple[int, int, int, int]] | None = None

    def _synchronize(self) -> None:
        if self.torch.cuda.is_available():
            self.torch.cuda.synchronize()

    def _timed(self, function):
        self._synchronize()
        started = time.perf_counter()
        result = function()
        self._synchronize()
        return result, time.perf_counter() - started

    def _prepare_layout(self, width: int, height: int) -> None:
        size = (width, height)
        if self._frame_size is not None and self._frame_size != size:
            raise ValueError(
                f"Размер кадра изменился внутри видео: {self._frame_size} -> {size}"
            )
        if self._frame_size is not None:
            return
        self._frame_size = size
        if self.mode == "full":
            self._bounds = [(0, 0, width, height)]
        else:
            self._bounds = fixed_tile_bounds(width, height, self.layout)

    def _render(self, frame: np.ndarray):
        frame = np.asarray(frame)
        if frame.ndim == 3:
            if frame.shape[2] != 3:
                raise ValueError("Ожидается BGR-кадр с тремя каналами")
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        elif frame.ndim == 2:
            gray = frame
        else:
            raise ValueError("Кадр должен иметь форму HxW или HxWx3")
        if not np.isfinite(gray).all():
            raise ValueError("Кадр содержит NaN или Inf")

        height, width = gray.shape
        self._prepare_layout(width, height)
        assert self._bounds is not None
        images = []
        for x1, y1, x2, y2 in self._bounds:
            crop = gray[y1:y2, x1:x2]
            resized = cv2.resize(
                crop, (self.eye_width, self.eye_height), interpolation=cv2.INTER_AREA
            )
            images.append(resized.astype(np.float32) / 255.0)
        cartesian = self.torch.from_numpy(np.stack(images, axis=0)).unsqueeze(1)
        cartesian = cartesian.to(self.flyvis.device, dtype=self.torch.float32)
        return self.eye(cartesian)

    def _points_and_vectors(self, flow) -> MotionField:
        assert self._bounds is not None
        flow_np = flow.detach().cpu().numpy().astype(np.float32)
        points = []
        vectors = []
        centers_y = self._local_centers_yx[:, 0] + self.eye_height // 2
        centers_x = self._local_centers_yx[:, 1] + self.eye_width // 2
        norm_x = centers_x / max(self.eye_width - 1, 1)
        norm_y = centers_y / max(self.eye_height - 1, 1)
        for tile_index, (x1, y1, x2, y2) in enumerate(self._bounds):
            tile_width = x2 - x1
            tile_height = y2 - y1
            points.append(
                np.column_stack(
                    [x1 + norm_x * tile_width, y1 + norm_y * tile_height]
                ).astype(np.float32)
            )
            tile_flow = flow_np[tile_index].T.copy()
            # Обучающие цели FlyVis имеют единицы pixel / image_height, после
            # чего BoxEye суммирует их по kernel_size**2 пикселям. Поэтому обе
            # компоненты возвращаются в пиксели через высоту текущего фрагмента.
            pixel_scale = self.flow_gain * tile_height / self.kernel_area
            tile_flow[:, 0] *= pixel_scale
            # У декодера FlyVis положительная вертикальная ось направлена вверх.
            tile_flow[:, 1] *= -pixel_scale
            vectors.append(tile_flow)
        return MotionField(np.concatenate(points), np.concatenate(vectors))

    def observe(self, frame: np.ndarray) -> MotionField:
        """Обработать очередной кадр и вернуть поле в исходных пикселях."""
        rendered, render_seconds = self._timed(lambda: self._render(frame))
        self.stats["observations"] += 1
        self.stats["render_seconds"] += render_seconds

        batch_size = rendered.shape[0]
        if self._last_hex is None:
            with self.torch.inference_mode():
                self._state, network_seconds = self._timed(
                    lambda: self.network.fade_in_state(
                        self.fade_in_seconds, self.dt, rendered[:, 0]
                    )
                )
                initial_activity = self._state.nodes.activity.unsqueeze(1)
                initial_decoded, decoder_seconds = self._timed(
                    lambda: self.decoder(initial_activity)
                )
            self.stats["network_seconds"] += network_seconds
            self.stats["decoder_seconds"] += decoder_seconds
            self._baseline_flow = initial_decoded[:, 0].detach()
            # Активность после fade-in: (tiles, nodes) — baseline для считывания.
            self.baseline_activity = initial_activity[:, 0].detach().cpu().numpy()
            self._last_hex = rendered[:, 0:1]
            self.last_decoded = None
            zero_flow = self.torch.zeros(
                (batch_size, 2, rendered.shape[-1]),
                device=self.flyvis.device,
                dtype=self.torch.float32,
            )
            return self._points_and_vectors(zero_flow)

        if self.temporal_resampling == "linear":
            alpha = self.torch.arange(
                1,
                self.integration_steps + 1,
                device=self.flyvis.device,
                dtype=self.torch.float32,
            ).view(1, self.integration_steps, 1, 1)
            alpha = alpha / self.integration_steps
            stimulus = self._last_hex * (1.0 - alpha) + rendered[:, 0:1] * alpha
        else:
            # Официальный обучающий конвейер FlyVis повторяет входные кадры
            # кусочно-постоянно при переводе Sintel с 24 на 50 Гц.
            stimulus = rendered[:, 0:1].repeat(1, self.integration_steps, 1, 1)
        with self.torch.inference_mode():
            states, network_seconds = self._timed(
                lambda: self.network.simulate(
                    stimulus, self.dt, initial_state=self._state, as_states=True
                )
            )
            activity = self.torch.stack(
                [state.nodes.activity for state in states], dim=1
            )
            decoded, decoder_seconds = self._timed(lambda: self.decoder(activity))
        self.stats["network_seconds"] += network_seconds
        self.stats["decoder_seconds"] += decoder_seconds
        if self.subtract_initial_baseline:
            assert self._baseline_flow is not None
            decoded = decoded - self._baseline_flow[:, None]
        self._state = states[-1]
        self._last_hex = rendered[:, 0:1]
        self.last_decoded = decoded.detach()
        # Активность внутренних шагов текущего кадра: (tiles, steps, nodes).
        self.last_activity = activity.detach().cpu().numpy()
        return self.field_from_last(self.flow_aggregation)

    def field_from_last(self, aggregation: str = "mean") -> MotionField:
        """Повторно агрегировать последний внутренний интервал без запуска сети."""
        if self.last_decoded is None:
            raise RuntimeError("Ещё не обработана пара кадров")
        if aggregation == "mean":
            interval_flow = self.last_decoded.mean(dim=1)
        elif aggregation == "sum":
            interval_flow = self.last_decoded.sum(dim=1)
        elif aggregation == "last":
            interval_flow = self.last_decoded[:, -1]
        else:
            raise ValueError("aggregation должен быть mean, last или sum")
        return self._points_and_vectors(interval_flow)
