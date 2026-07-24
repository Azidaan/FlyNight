"""LQA (Local Quality Assessment) — пространственная ветвь DRB (Часть 2.1 плана).

Три характеристики качества в скользящем окне k×k:
    D(x) = avgpool(x²) − [avgpool(x)]²          локальная дисперсия
    C(x) = maxpool(x) − avgpool(x)              пиковая активация
    G(x) = sqrt((Kx⊛x)² + (Ky⊛x)²)              градиент Собеля
Каждая карта считается поканально (B,C,H,W) и усредняется по C (решение
подэтапа 4.6 плана: среднее по каналам — простейшая интерпретируемая схема;
вариант Conv2d(3·C,1) отложен в ablation Этапа 7). Затем:
    Q = sigmoid(Conv1x1(concat[D̄, C̄, Ḡ]))  ∈ [0,1], форма (B,1,H,W).
"""
import torch
import torch.nn.functional as F
from torch import nn


class LQA(nn.Module):
    """Пространственная ветвь DRB: карта качества Q по трём характеристикам."""

    def __init__(self, in_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.k = kernel_size
        self.conv_merge = nn.Conv2d(3, 1, kernel_size=1)
        # Ядра Собеля — фиксированные буферы, не обучаются.
        kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
        ky = kx.t().contiguous()
        self.register_buffer('sobel', torch.stack([kx, ky]).unsqueeze(1))  # (2,1,3,3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        pad = self.k // 2

        avg = F.avg_pool2d(x, self.k, stride=1, padding=pad)
        avg_sq = F.avg_pool2d(x * x, self.k, stride=1, padding=pad)
        d = (avg_sq - avg * avg).clamp_min(0).mean(dim=1, keepdim=True)

        mx = F.max_pool2d(x, self.k, stride=1, padding=pad)
        peak = (mx - avg).mean(dim=1, keepdim=True)

        sobel = self.sobel.to(dtype=x.dtype).repeat(c, 1, 1, 1)      # (2C,1,3,3)
        g = F.conv2d(x, sobel, padding=1, groups=c)                   # (B,2C,H,W)
        # каналы каждой группы соседствуют: (2i — Kx⊛x_i, 2i+1 — Ky⊛x_i)
        g = g.view(b, c, 2, h, w)
        grad = torch.sqrt(g[:, :, 0] ** 2 + g[:, :, 1] ** 2 + 1e-12) \
            .mean(dim=1, keepdim=True)

        q = torch.sigmoid(self.conv_merge(torch.cat([d, peak, grad], dim=1)))
        return q
