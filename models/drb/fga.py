"""FGA (Frequency-Gated Attention) — частотная ветвь DRB (Часть 2.1 плана).

    y_FGA = IFFT2D(M ⊙ FFT2D(x))

Используются вещественные rfft2/irfft2 (эрмитова симметрия — хранится половина
спектра). Маска M — обучаемая, вещественная, поканальная (C, H, W//2+1),
инициализация единицами (нейтральный старт: FGA = тождественное преобразование).

Ограничение (фиксируется в разделе 2.1 диссертации): маска фиксированного
размера привязывает модель к разрешению обучения (вход 640×640). При другом
разрешении маска билинейно интерполируется — численно работает, но
оптимальность не гарантируется; честный путь — переобучение.

AMP: FFT выполняется в FP32 принудительно (numерическая стабильность под
autocast), результат приводится к dtype входа.
"""
import torch
import torch.nn.functional as F
from torch import nn


class FGA(nn.Module):
    """Частотная ветвь DRB: обучаемая маска в спектральной области."""

    def __init__(self, in_channels: int, height: int, width: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.height = height
        self.width = width
        self.mask = nn.Parameter(torch.ones(in_channels, height, width // 2 + 1))

    def _mask_for(self, h: int, w: int) -> torch.Tensor:
        wf = w // 2 + 1
        if (h, wf) == (self.height, self.width // 2 + 1):
            return self.mask
        # другой размер входа: интерполяция маски (см. ограничение в докстринге)
        m = self.mask.unsqueeze(0)                      # (1,C,H,Wf)
        m = F.interpolate(m, size=(h, wf), mode='bilinear', align_corners=False)
        return m.squeeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        with torch.autocast(device_type=x.device.type, enabled=False):
            spec = torch.fft.rfft2(x.float())            # (B,C,H,W//2+1) complex
            spec = spec * self._mask_for(h, w).to(spec.real.dtype)
            y = torch.fft.irfft2(spec, s=(h, w))
        return y.to(x.dtype)
