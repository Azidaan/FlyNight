"""Интеграция DRB в neck YOLO — wrapper-подход (Этап 4, подэтап 4.4).

Как устроено. Голова Detect в Ultralytics берёт входы из трёх слоёв neck
(индексы head.f, у yolo26s — [16, 19, 22]; читаются динамически). Каждый из
этих слоёв оборачивается в DRBWrappedLayer: слой считает свой выход, DRB
очищает его и отдаёт дальше; карта Q сохраняется в last_q для трекера.
Атрибуты .i/.f/.type копируются — конвейер DetectionModel их использует
для маршрутизации, обёртка для него прозрачна.

Почему через собственный Trainer. model.train() Ultralytics ПЕРЕСОБИРАЕТ
модель: trainer.get_model строит чистую DetectionModel из yaml и грузит веса —
обёртки, вставленные заранее, теряются. DRBTrainer переопределяет get_model
и вставляет DRB после сборки. Запуск: YOLO(w).train(trainer=DRBTrainer, ...).

Ограничения wrapper-подхода (для notes/drb_integration_notes.md):
- resume работает только через ветку "weights уже обёрнуты" в get_model;
- чекпойнты DRB-модели пикль-зависимы от кода проекта (models.drb должен
  быть импортируем при загрузке);
- перед финальной версией планируется чистая YAML-регистрация (Этап 5+).
"""
import sys
from pathlib import Path

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics.models.yolo.detect.train import DetectionTrainer  # noqa: E402

from models.drb import DRB  # noqa: E402


class DRBWrappedLayer(nn.Module):
    """Обёртка слоя neck: inner -> DRB. Прозрачна для конвейера Ultralytics."""

    def __init__(self, inner: nn.Module, drb: DRB) -> None:
        super().__init__()
        self.inner = inner
        self.drb = drb
        # служебные атрибуты маршрутизации DetectionModel
        self.i = getattr(inner, 'i', None)
        self.f = getattr(inner, 'f', None)
        self.type = getattr(inner, 'type', type(inner).__name__)
        self.np = getattr(inner, 'np', sum(p.numel() for p in inner.parameters()))
        self.last_q: torch.Tensor | None = None

    def forward(self, x):
        y = self.inner(x)
        y_out, q = self.drb(y)
        self.last_q = q.detach() if q is not None else None
        return y_out


def has_drb(det_model: nn.Module) -> bool:
    return any(isinstance(m, DRBWrappedLayer) for m in det_model.model)


def probe_pyramid_shapes(det_model: nn.Module, layer_indices: list[int],
                         imgsz: int = 640) -> dict[int, tuple[int, int, int]]:
    """(C, H, W) выходов указанных слоёв на входе imgsz — через hooks."""
    shapes: dict[int, tuple[int, int, int]] = {}
    handles = []
    for idx in layer_indices:
        def _hook(mod, inp, out, idx=idx):
            shapes[idx] = tuple(out.shape[1:])
        handles.append(det_model.model[idx].register_forward_hook(_hook))
    was_training = det_model.training
    det_model.eval()
    device = next(det_model.parameters()).device
    with torch.no_grad():
        det_model(torch.zeros(1, 3, imgsz, imgsz, device=device))
    for h in handles:
        h.remove()
    if was_training:
        det_model.train()
    return shapes


def insert_drb(det_model: nn.Module, imgsz: int = 640,
               lqa_kernel: int = 3, branches: str = 'both') -> list[int]:
    """Оборачивает DRB слои-входы головы Detect. Идемпотентно.

    branches: 'both' | 'lqa' | 'fga' — конфигурация ветвей (Этап 7).
    Возвращает индексы обёрнутых слоёв (уровни P3/P4/P5 по порядку head.f).
    """
    if has_drb(det_model):
        return [m.i for m in det_model.model if isinstance(m, DRBWrappedLayer)]

    head = det_model.model[-1]
    indices = list(head.f)                      # напр. [16, 19, 22]
    shapes = probe_pyramid_shapes(det_model, indices, imgsz)
    device = next(det_model.parameters()).device
    for idx in indices:
        c, h, w = shapes[idx]
        drb = DRB(c, h, w, lqa_kernel=lqa_kernel, branches=branches).to(device)
        det_model.model[idx] = DRBWrappedLayer(det_model.model[idx], drb)
    return indices


class DRBInputLayer(nn.Module):
    """DRB перед первым слоем: очистка на уровне RGB-пикселей (Серия II, Этап 7).

    В отличие от DRBWrappedLayer (DRB после выхода слоя), здесь DRB применяется
    ко ВХОДУ слоя 0 — то есть к изображению до backbone. Один экземпляр вместо
    трёх. Честная оговорка для раздела 2.5 диссертации: на входе C=3 против
    сотен каналов в neck — статистики LQA работают в совсем другом режиме.
    """

    def __init__(self, inner: nn.Module, drb: DRB) -> None:
        super().__init__()
        self.inner = inner
        self.drb = drb
        self.i = getattr(inner, 'i', None)
        self.f = getattr(inner, 'f', None)
        self.type = getattr(inner, 'type', type(inner).__name__)
        self.np = getattr(inner, 'np', sum(p.numel() for p in inner.parameters()))
        self.last_q: torch.Tensor | None = None

    def forward(self, x):
        x_clean, q = self.drb(x)
        self.last_q = q.detach() if q is not None else None
        return self.inner(x_clean)


def insert_drb_at_input(det_model: nn.Module, imgsz: int = 640,
                        lqa_kernel: int = 3) -> list[int]:
    """Вставляет один DRB на RGB-вход (до backbone). Идемпотентно."""
    if isinstance(det_model.model[0], DRBInputLayer):
        return [0]
    if has_drb(det_model):
        raise RuntimeError('Модель уже содержит DRB в neck — варианты несовместимы')
    device = next(det_model.parameters()).device
    drb = DRB(3, imgsz, imgsz, lqa_kernel=lqa_kernel).to(device)
    det_model.model[0] = DRBInputLayer(det_model.model[0], drb)
    return [0]


def iter_drb(det_model: nn.Module):
    """Пары (индекс слоя, модуль DRB) — для логирования α и доступа к маскам.

    Работает и с neck-вариантом (DRBWrappedLayer), и с input-вариантом.
    """
    for m in det_model.model:
        if isinstance(m, (DRBWrappedLayer, DRBInputLayer)):
            yield m.i, m.drb


def collect_q_maps(det_model: nn.Module) -> dict[int, torch.Tensor]:
    """Карты Q последнего forward: {индекс слоя: (B,1,H,W)}. Для трекера (Этап 9)."""
    return {m.i: m.last_q for m in det_model.model
            if isinstance(m, (DRBWrappedLayer, DRBInputLayer))
            and m.last_q is not None}


class DRBTrainer(DetectionTrainer):
    """DetectionTrainer, вставляющий DRB в пересобранную модель.

    Использование: YOLO(weights).train(trainer=DRBTrainer, data=..., ...)
    Вариант ветвей задаётся класс-атрибутом drb_variant (см. make_drb_trainer):
    'both' | 'lqa' | 'fga' — DRB в neck; 'input' — один DRB на RGB-входе.
    """

    drb_variant: str = 'both'

    def get_model(self, cfg=None, weights=None, verbose=True):
        if isinstance(weights, nn.Module) and (
                has_drb(weights) or isinstance(weights.model[0], DRBInputLayer)):
            return weights                      # resume: модель уже с DRB
        model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
        if self.drb_variant == 'input':
            wrapped = insert_drb_at_input(model)
            print(f'[DRB] вставлен на RGB-вход (слой {wrapped})')
        else:
            wrapped = insert_drb(model, branches=self.drb_variant)
            print(f'[DRB] обёрнуты слои {wrapped} '
                  f'(входы головы Detect, ветви: {self.drb_variant})')
        return model


def make_drb_trainer(variant: str = 'both') -> type[DRBTrainer]:
    """Фабрика Trainer-класса с зафиксированным вариантом DRB.

    Нужна потому, что model.train(trainer=...) принимает класс, а не
    экземпляр — параметризовать можно только через класс-атрибут.
    """
    if variant not in ('both', 'lqa', 'fga', 'input'):
        raise ValueError(f'variant: both|lqa|fga|input, получено {variant!r}')
    return type(f'DRBTrainer_{variant}', (DRBTrainer,), {'drb_variant': variant})
