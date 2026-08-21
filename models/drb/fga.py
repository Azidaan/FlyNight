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

    def _idx_for(self, h: int, w: int) -> torch.Tensor:
        """Разметка колец для КОНКРЕТНОГО размера карты признаков.

        Размер не всегда обучающий: проверка идёт прямоугольными пачками,
        и карты там другой формы. Разметка для обучающего размера кэшируется
        в буфер, для прочих считается на месте — она задана в нормированных
        частотах, поэтому переносится без интерполяции.
        """
        if (h, w) != (self.height, self.width):
            return self._bands_for(h, w).to(self.delta.device)
        idx = getattr(self, 'band_idx', None)
        if idx is None:                     # у распакованной модели буфера нет
            idx = self._bands_for(self.height, self.width).to(self.delta.device)
            self.register_buffer('band_idx', idx, persistent=False)
        return idx

    @property
    def mask3d(self) -> torch.Tensor:
        """Маска в рабочем виде (C, H, W//2+1); нейтраль — единицы."""
        return self._mask_for(self.height, self.width)

    def enable_adaptive(self, gain_max: float = 0.5) -> None:
        """Включает предсказание подъёма высоких частот по самому кадру.

        Зачем. Статическая маска, выученная по всему набору, сходится к
        нейтрали: усреднённая по дню и ночи, никакая фильтрация не лучше
        никакой. Замер 21.08 показал, почему — кадры разные. У ночных кадров
        BDD100K вдвое меньше высокочастотной энергии (доля 0.119 против
        0.140) и вдвое слабее градиент. Причём шума в них МЕНЬШЕ, а не
        больше (0.0030 против 0.0076): это видео с регистратора, камера сама
        давит шум и жмёт в JPEG. Значит ночью сигнал не тонет в шуме, а
        ослабевает — и нужна компенсирующая добавка, а не подавление.

        Предсказываются два числа: насколько поднять верх и с какой частоты
        начинать. Всего два, а не значение на каждое кольцо, — динамические
        фильтры с большой свободой схлопываются в константу, это известный
        вырожденный исход. Заодно выученную зависимость можно нарисовать
        одной кривой.

        Выходной слой обнулён, а смещение подобрано так, чтобы подъём на
        старте был около нуля: блок начинает точным тождеством.
        """
        self.gain_max = gain_max
        # Создаётся там же, где уже лежит блок: включение бывает и после
        # переноса модели на видеокарту, и тогда новый слой остался бы на
        # процессоре, а прямой проход упал бы на разных устройствах.
        self.pred = nn.Sequential(
            nn.Linear(3, 16), nn.SiLU(), nn.Linear(16, 2)
        ).to(self.delta.device, self.delta.dtype)
        with torch.no_grad():
            self.pred[2].weight.zero_()
            self.pred[2].bias.copy_(torch.tensor([-4.0, 0.0]))
        self.last_gain = 0.0
        self.last_knee = 0.0

    def _adaptive_gain(self, spec: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """Маска-подъём (B,1,H,Wf), предсказанная по спектру самого кадра.

        Разметка колец берётся под ФАКТИЧЕСКИЙ размер карты: на проверке
        пачки прямоугольные, и разметка обучающего размера туда не подходит.
        Без этого прогон падал на проверке после первой эпохи.
        """
        mag = spec.abs()                                   # (B,C,H,Wf)
        idx = self._idx_for(h, w)
        high = (idx >= (2 * self.bands) // 3)              # верхняя треть колец
        e_all = mag.sum(dim=(1, 2, 3)) + 1e-6
        e_hi = (mag * high).sum(dim=(1, 2, 3))
        desc = torch.stack([
            e_hi / e_all,                                  # доля верхней энергии
            (e_all.log1p() / 20.0).clamp(-5, 5),           # общий масштаб
            (mag.mean(dim=(1, 2, 3)).log1p() * 10.0).clamp(-5, 5),
        ], dim=1).to(self.pred[0].weight.dtype)

        out = self.pred(desc)
        gain = self.gain_max * torch.sigmoid(out[:, 0])    # насколько поднять
        knee = 0.20 + 0.50 * torch.sigmoid(out[:, 1])      # с какой частоты
        self.last_gain = float(gain.mean().detach())
        self.last_knee = float(knee.mean().detach())

        r = (idx.to(gain.dtype) + 0.5) / self.bands        # (H,Wf)
        shape = ((r[None] - knee[:, None, None])
                 / (1.0 - knee[:, None, None])).clamp(0.0, 1.0)
        return (1.0 + gain[:, None, None] * shape).unsqueeze(1)

    def _mask_for(self, h: int, w: int) -> torch.Tensor:
        return self.response[:, self._idx_for(h, w)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        with torch.autocast(device_type=x.device.type, enabled=False):
            spec = torch.fft.rfft2(x.float())            # (B,C,H,W//2+1) complex
            mask = self._mask_for(h, w).to(spec.real.dtype)
            if hasattr(self, 'pred'):
                # два множителя: выученный по набору и предсказанный по кадру
                mask = mask * self._adaptive_gain(spec, h, w)
            spec = spec * mask
            y = torch.fft.irfft2(spec, s=(h, w))
        return y.to(x.dtype)
