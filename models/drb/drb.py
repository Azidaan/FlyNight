"""DRB (Degradation-Resistant Block) — сборка (Часть 2.1, Этап 4 плана).

    x' = x + α₁ · (Q ⊙ x) + α₂ · y_FGA,   α₁ = σ(w_LQA), α₂ = σ(w_FGA)

Инициализация w_LQA = w_FGA = 0 даёт α₁ = α₂ = 0.5 — обе ветви одинаково
активны на старте, дальше сеть перераспределяет веса сама. Веса независимые
(НЕ α и 1−α): жёсткая связка мешала бы усиливать обе ветви параллельно,
а обученные значения α интерпретируются на защите (Часть 2.1 плана).

forward возвращает КОРТЕЖ (x_out, Q): карта качества Q уходит наружу —
в трекер для адаптации порогов ассоциации (Этап 9).
"""
import torch
from torch import nn

from models.drb.fga import FGA
from models.drb.lqa import LQA


class DRB(nn.Module):
    """Полный блок: LQA + FGA + взвешенная сумма + остаточное соединение.

    branches — конфигурация ветвей для ablation Этапа 7:
      'both' (по умолчанию) — полный DRB;
      'lqa'  — только пространственная ветвь (FGA-ветвь не создаётся);
      'fga'  — только частотная ветвь (LQA не создаётся, Q = None).
    Отключение ветви эквивалентно фиксации её веса w = −∞ (σ = 0) из плана,
    но чище: неактивная ветвь не тратит вычисления (FFT) и параметры.
    """

    def __init__(self, in_channels: int, height: int, width: int,
                 lqa_kernel: int = 3, branches: str = 'both') -> None:
        super().__init__()
        if branches not in ('both', 'lqa', 'fga'):
            raise ValueError(f'branches: both|lqa|fga, получено {branches!r}')
        self.branches = branches
        # Вес ветви — ВЕКТОР по каналам, а не одно число на весь уровень.
        # Одним числом судьба всех каналов решалась разом, хотя одни несут
        # текстуру, другие контекст, и ветвь нужна им по-разному. Плюс
        # измеренное: единственный скаляр за 12 эпох сдвигался на 0.0036,
        # то есть ни на что; у вектора суммарный градиент во столько же раз
        # больше, сколько в нём каналов.
        if branches in ('both', 'lqa'):
            self.lqa = LQA(in_channels, kernel_size=lqa_kernel)
            self.w_lqa = nn.Parameter(torch.zeros(in_channels))
        if branches in ('both', 'fga'):
            self.fga = FGA(in_channels, height, width)
            self.w_fga = nn.Parameter(torch.zeros(in_channels))

    @property
    def alphas(self) -> tuple[float, float]:
        """Средние по каналам (α₁, α₂); у отключённой ветви — 0.0."""
        a1 = (torch.sigmoid(self.w_lqa).mean().item()
              if hasattr(self, 'w_lqa') else 0.0)
        a2 = (torch.sigmoid(self.w_fga).mean().item()
              if hasattr(self, 'w_fga') else 0.0)
        return (a1, a2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        q = None
        y = torch.zeros_like(x)
        if self.branches in ('both', 'lqa'):
            q = self.lqa(x)                               # (B,1,H,W), [0,1]
            a1 = torch.sigmoid(self.w_lqa).view(1, -1, 1, 1)
            y = y + a1 * (q * x)                          # broadcast Q по C
        if self.branches in ('both', 'fga'):
            a2 = torch.sigmoid(self.w_fga).view(1, -1, 1, 1)
            y = y + a2 * self.fga(x)
        return x + y, q
