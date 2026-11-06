"""Фронт-энд «фоторецептор + ламина» перед глазом FlyVis (реестр 1.7, Н9; записка «Модернизация мухи», М1).

FlyVis получает яркость `gray/255` без адаптации, а обучена при контрасте `exp(N(0, 0,2))`
и средней яркости ~0,45 (Sintel). Ночной кадр BDD100K (средняя яркость 0,05–0,12) вне
этого распределения. Модуль восстанавливает две ступени живой сетчатки, которых в
модели нет:

* фоторецептор R1–R6 — сжатие Нака–Раштона `I / (I + I_adapt)` с точкой полунасыщения,
  подстраивающейся под локальное среднее последних ~τ секунд (закон Вебера);
* ламина L1/L2 — локальный контраст: вычитание пространственного среднего и деление на
  локальный размах.

Заимствовано по идее из Brinkworth & O'Carroll, PLoS Comput Biol 2009 (адаптивный
фоторецептор + LMC перед детектором движения); реализация своя и упрощённая: без
адаптивной частоты среза и без динамической гаммы. Внутренние параметры мухи не меняются.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

MODES = ("none", "norm", "photoreceptor", "lamina")


@dataclass
class RetinaConfig:
    mode: str = "photoreceptor"
    tau_adapt: float = 0.3      # с: постоянная времени адаптации точки полунасыщения
    sigma_adapt: float = 24.0   # px: пространственное окно адаптации (размытие среднего)
    tau_filter: float = 0.0     # с: временной ФНЧ фоторецептора (0 — выключен)
    dark_floor: float = 0.05    # «темновой ток»: слагаемое в делителе, ограничивает усиление почти чёрных областей (шум сжатия)
    gain: float = 2.0           # усиление вокруг 0,5 после сжатия (приведение к размаху Sintel)
    sigma_lamina: float = 12.0  # px: окно центр-окружение ламины
    lamina_gain: float = 0.25   # масштаб локального контраста → доли от 0,5
    lamina_floor: float = 0.1   # порог размаха: контраст ниже него не усиливается (иначе делитель раздувает шум квантования)
    target_mean: float = 0.45   # средняя яркость обученного распределения
    target_std: float = 0.2     # размах для режима norm
    eps: float = 1e-3
    blur_downscale: int = 1     # >1: гладкие поля считаются на уменьшенном кадре (ускорение ~16× при 4)

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"mode должен быть одним из {MODES}, получено {self.mode!r}")


class Retina:
    """Состояние адаптации живёт между кадрами одного видео; `reset()` перед новым видео.

    `process(gray_uint8) -> float32 (H, W) в единицах 0..255` — так, чтобы дальнейший код
    (`EyeBank._render`: `/255`) не менялся.
    """

    def __init__(self, config: RetinaConfig, fps: float) -> None:
        self.config = config
        self.fps = float(fps)
        self._adapt: np.ndarray | None = None
        self._filtered: np.ndarray | None = None
        self.alpha_adapt = 1.0 - np.exp(-1.0 / (self.fps * config.tau_adapt)) if config.tau_adapt > 0 else 1.0
        self.alpha_filter = 1.0 - np.exp(-1.0 / (self.fps * config.tau_filter)) if config.tau_filter > 0 else 1.0

    def reset(self) -> None:
        self._adapt = None
        self._filtered = None

    def describe(self) -> dict:
        return {**asdict(self.config), "fps": self.fps, "alpha_adapt": float(self.alpha_adapt), "alpha_filter": float(self.alpha_filter)}

    def _blur(self, image: np.ndarray, sigma: float) -> np.ndarray:
        import cv2

        if sigma <= 0:
            return image
        factor = self.config.blur_downscale
        if factor > 1 and sigma >= 2.0 * factor:
            # Гладкое поле (σ ≫ пикселя): размытие на кадре, уменьшенном в factor раз, и билинейный возврат.
            # На полном кадре 1280×720 σ = 24 стоит ~60 мс на CPU, так — ~4 мс; поле низкочастотное.
            h, w = image.shape[:2]
            small = cv2.resize(image, (max(w // factor, 1), max(h // factor, 1)), interpolation=cv2.INTER_AREA)
            small = cv2.GaussianBlur(small, (0, 0), sigmaX=float(sigma) / factor, sigmaY=float(sigma) / factor, borderType=cv2.BORDER_REFLECT)
            return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
        return cv2.GaussianBlur(image, (0, 0), sigmaX=float(sigma), sigmaY=float(sigma), borderType=cv2.BORDER_REFLECT)

    def process(self, gray: np.ndarray) -> np.ndarray:
        cfg = self.config
        image = gray.astype(np.float32) / 255.0 if gray.dtype == np.uint8 else gray.astype(np.float32) / 255.0
        if cfg.mode == "none":
            return image * 255.0
        if cfg.mode == "norm":
            # Контроль без биологии: глобальная нормализация кадра к обученному распределению.
            z = (image - float(image.mean())) / (float(image.std()) + cfg.eps)
            return np.clip(cfg.target_mean + cfg.target_std * z, 0.0, 1.0) * 255.0
        # Фоторецептор: временной ФНЧ (опционально) и адаптация точки полунасыщения.
        if cfg.tau_filter > 0:
            self._filtered = image if self._filtered is None else self._filtered + self.alpha_filter * (image - self._filtered)
            image = self._filtered
        local_mean = self._blur(image, cfg.sigma_adapt)
        self._adapt = local_mean if self._adapt is None else self._adapt + self.alpha_adapt * (local_mean - self._adapt)
        denominator_floor = cfg.dark_floor + cfg.eps
        response = image / (image + self._adapt + denominator_floor)  # закон Вебера; dark_floor («темновой ток») гасит усиление шума в черноте
        reference = self._adapt / (2.0 * self._adapt + denominator_floor)  # ответ на локальный адаптированный уровень -> центр 0,5
        out = 0.5 + cfg.gain * (response - reference)
        if cfg.mode == "lamina":
            center = out
            surround = self._blur(center, cfg.sigma_lamina)
            spread = np.sqrt(np.maximum(self._blur((center - surround) ** 2, cfg.sigma_lamina), 0.0)) + cfg.lamina_floor
            out = 0.5 + cfg.lamina_gain * (center - surround) / spread
        return np.clip(out + (cfg.target_mean - 0.5), 0.0, 1.0) * 255.0


def make_retina(mode: str, fps: float, **overrides) -> Retina:
    return Retina(RetinaConfig(mode=mode, **overrides), fps)
