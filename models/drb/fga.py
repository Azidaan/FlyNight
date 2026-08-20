"""FGA (Frequency-Gated Attention) — частотная ветвь DRB (Часть 2.1 плана).

    y_FGA = IFFT2D(M ⊙ FFT2D(x))

Используются вещественные rfft2/irfft2 (эрмитова симметрия — хранится половина
спектра). Маска M — обучаемая, вещественная, поканальная, инициализация
единицами (нейтральный старт: FGA = тождественное преобразование). Хранится
она как отклонение от единицы, см. комментарий в __init__.

Маска РАДИАЛЬНАЯ: одно значение на канал и кольцо частот, а не на каждый
отсчёт спектра. Причина содержательная — «мелкие объекты против крупных» это
свойство модуля частоты, а не её направления, и различать |kx|=0.4 от
|ky|=0.4 задаче незачем. Причина практическая — измеренная: при значении на
каждый отсчёт параметр видел градиент только от своего одного отсчёта, за 12
эпох маска сдвигалась на 0.0004, а на высоких частотах в 107 раз слабее, чем
на низких, то есть ровно там, ради чего ветвь и заведена. Кольцо собирает
градиент с сотни с лишним отсчётов сразу.

Побочно снимается и ограничение по разрешению: принадлежность отсчёта кольцу
считается из его нормированной частоты, поэтому маска переносится на любой
размер входа без интерполяции.

AMP: FFT выполняется в FP32 принудительно (numерическая стабильность под
autocast), результат приводится к dtype входа.
"""
import torch
from torch import nn


class FGA(nn.Module):
    """Частотная ветвь DRB: обучаемая маска в спектральной области."""

    def __init__(self, in_channels: int, height: int, width: int,
                 bands: int = 24) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.height = height
        self.width = width
        self.bands = bands
        # Хранится ОТКЛОНЕНИЕ от нейтрали, а не сама маска. Нейтраль маски —
        # единица: спектр проходит без изменений. А штраф за величину весов
        # тянет любой параметр к нулю, и для маски ноль означает не «ничего не
        # менять», а «вырезать все частоты». Когда хранилась сама маска, её
        # равномерно гасило: 1.0 -> 0.71 за 12 эпох, причём одинаково при
        # вкладе ветви 50%, 12% и 2% — то есть это было затухание, а не
        # обучение. С отклонением ноль совпадает с нейтралью, тянуть некуда.
        #
        # Хранение ОДНОМЕРНОЕ: Ultralytics отправляет в ветвь Muon всё, у чего
        # ndim >= 2, а Muon нормирует шаг независимо от силы градиента, и
        # ровное тяготение штрафа перебивало слабый сигнал обучения.
        self.delta = nn.Parameter(torch.zeros(in_channels * bands))
        # Принадлежность отсчёта спектра кольцу — не параметр, а разметка;
        # persistent=False, чтобы не таскать её в чекпойнтах.
        self.register_buffer('band_idx', self._bands_for(height, width),
                             persistent=False)

    def _bands_for(self, h: int, w: int) -> torch.Tensor:
        """Номер радиального кольца для каждого отсчёта спектра, (H, W//2+1)."""
        fy = torch.fft.fftfreq(h).abs().view(h, 1)
        fx = torch.fft.rfftfreq(w).view(1, w // 2 + 1)
        r = (fy ** 2 + fx ** 2).sqrt() / (0.5 * 2 ** 0.5)      # 0..1
        return (r * self.bands).long().clamp_(max=self.bands - 1)

    @property
    def response(self) -> torch.Tensor:
        """Частотная характеристика (C, bands): нейтраль — единицы.

        То, что рисуется одной кривой от низких частот к высоким.
        """
        bands = getattr(self, 'bands', None)
        if bands is None or self.delta.numel() != self.in_channels * bands:
            raise RuntimeError(
                f'маска из чекпойнта старого формата: {self.delta.numel()} '
                f'значений на {self.in_channels} каналов. Такие прогоны '
                f'считались до перехода на радиальные кольца, и новым кодом '
                f'модель не читается — их надо пересчитать заново.')
        return 1.0 + self.delta.view(self.in_channels, bands)

    def _idx(self) -> torch.Tensor:
        """Разметка колец. Буфер непостоянный, и при распаковке сохранённой
        модели его может не оказаться — тогда считаем на месте."""
        idx = getattr(self, 'band_idx', None)
        if idx is None:
            idx = self._bands_for(self.height, self.width).to(self.delta.device)
            self.register_buffer('band_idx', idx, persistent=False)
        return idx

    @property
    def mask3d(self) -> torch.Tensor:
        """Маска в рабочем виде (C, H, W//2+1); нейтраль — единицы."""
        return self._mask_for(self.height, self.width)

    def _mask_for(self, h: int, w: int) -> torch.Tensor:
        idx = self._idx()
        if (h, w) != (self.height, self.width):
            # другой размер входа: кольца пересчитываются, маска переносится
            # без интерполяции — она задана в нормированных частотах
            idx = self._bands_for(h, w).to(self.delta.device)
        return self.response[:, idx]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        with torch.autocast(device_type=x.device.type, enabled=False):
            spec = torch.fft.rfft2(x.float())            # (B,C,H,W//2+1) complex
            spec = spec * self._mask_for(h, w).to(spec.real.dtype)
            y = torch.fft.irfft2(spec, s=(h, w))
        return y.to(x.dtype)
