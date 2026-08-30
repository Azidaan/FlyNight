"""Фиксированный низкочастотный фильтр в операторы слияния шеи.

Зачем именно фиксированный. Обучаемая частотная маска сходилась к нейтрали
ЧЕТЫРЕЖДЫ из четырёх (по отсчётам спектра, по радиальным кольцам, предсказанная
по кадру, при десятикратном разгоне), а вес ветви `w_lqa` в режиме затвора не
получал градиента вовсе — все 896 каналов остались ровными нулями. Сеть, которой
дают выключатель, раз за разом им пользуется.

В литературе выигрывают ровно те, кто выключателя не даёт: HS-FPN получает
+2.9 AP по мелким при НУЛЕ обучаемых чисел в частотной части, FDA-DETR — +3.9
при четырёх числах на масштаб. У нашего блока их было 23 308 (а до переделки
747 520), и прирост около нуля.

Почему низкие частоты, а не высокие. Три независимых источника (разбор в хронике
от 21.08): ночной снимок деградирован шумом, который маскирует высокочастотные
текстуры, и высокочастотный слой несёт детали и шум одновременно. WTConv помогает
мелким объектам через НИЗКИЕ частоты и расширение охвата, подавляя высокочастотный
шум. Подъём высоких частот для ночи — вероятно, неверная операция.

Почему на апсемплинге. В шее yolo26s два оператора повышения разрешения (слои 11
и 14), оба `nearest` — они дают ступенчатые артефакты. Довод FreqFusion (TPAMI
2024): повышенный по разрешению путь несёт семантику и должен быть гладким, а
детали приходят из пропускной связи, с которой он склеивается. Сглаживание
именно там, а не «где-то в признаках».

Стоимость: одна depthwise-свёртка k×k. Ни преобразования Фурье, ни островка
полной точности внутри графа половинной точности — то есть прежние +17 % задержки
здесь не возникают в принципе.
"""
import torch
import torch.nn.functional as F
from torch import nn


def gaussian_kernel2d(kernel_size: int, sigma: float) -> torch.Tensor:
    """Радиально-симметричное ядро Гаусса, сумма равна единице."""
    if kernel_size % 2 == 0:
        raise ValueError(f'ядро должно быть нечётным, получено {kernel_size}')
    ax = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2
    g1 = torch.exp(-(ax ** 2) / (2 * sigma ** 2))
    k2 = torch.outer(g1, g1)
    return k2 / k2.sum()


class FixedLowPass(nn.Module):
    """Низкочастотная поправка с НУЛЁМ обучаемых чисел.

        y = x + s · (blur(x) − x),   s — константа конфига, не параметр.

    Форма с поправкой выбрана не ради безопасности старта (обучать здесь нечего),
    а ради двух вещей: при s = 0 получается точное тождество, что даёт побитовый
    тест на отключение; и сила вмешательства задаётся одним понятным числом,
    которое пишется в текст диссертации, а не подбирается сетью.

    Ядро — буфер, а не параметр: в state_dict попадает, в оптимизатор нет.
    Форма карты не фиксируется нигде, поэтому прямоугольные пачки валидации
    безопасны по построению (прежняя ловушка стоила пяти падений).
    """

    def __init__(self, channels: int, kernel_size: int = 5,
                 sigma: float = 1.0, strength: float = 1.0) -> None:
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.strength = float(strength)
        k = gaussian_kernel2d(kernel_size, sigma)
        self.register_buffer('kernel', k.expand(channels, 1, kernel_size, kernel_size).contiguous())

    def extra_repr(self) -> str:
        return (f'channels={self.channels}, k={self.kernel_size}, '
                f'sigma={self.sigma}, strength={self.strength}, обучаемых=0')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.strength == 0.0:
            return x
        pad = self.kernel_size // 2
        blur = F.conv2d(F.pad(x, (pad, pad, pad, pad), mode='replicate'),
                        self.kernel.to(dtype=x.dtype), groups=self.channels)
        return x + self.strength * (blur - x)


class SmoothedLayer(nn.Module):
    """Обёртка слоя шеи: inner → фильтр. Прозрачна для конвейера Ultralytics.

    Служебные атрибуты маршрутизации (`i`, `f`, `type`, `np`) переносятся с
    оборачиваемого слоя — без них DetectionModel не соберёт граф.
    """

    def __init__(self, inner: nn.Module, filt: FixedLowPass) -> None:
        super().__init__()
        self.inner = inner
        self.filt = filt
        self.i = getattr(inner, 'i', None)
        self.f = getattr(inner, 'f', None)
        self.type = getattr(inner, 'type', type(inner).__name__)
        self.np = getattr(inner, 'np', sum(p.numel() for p in inner.parameters()))

    def forward(self, x):
        return self.filt(self.inner(x))


def has_lowpass(det_model: nn.Module) -> bool:
    return any(isinstance(m, SmoothedLayer) for m in det_model.model)


def upsample_indices(det_model: nn.Module) -> list[int]:
    """Индексы операторов повышения разрешения в шее — читаются, не зашиваются."""
    return [i for i, m in enumerate(det_model.model)
            if isinstance(m, nn.Upsample)]


def _probe_out_channels(det_model: nn.Module, indices: list[int],
                        imgsz: int = 640) -> dict[int, int]:
    """Число каналов на выходе указанных слоёв — через hooks, без допущений."""
    ch: dict[int, int] = {}
    handles = []
    for idx in indices:
        def _hook(mod, inp, out, idx=idx):
            ch[idx] = int(out.shape[1])
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
    return ch


def insert_lowpass(det_model: nn.Module, placement: str = 'upsample',
                   kernel_size: int = 5, sigma: float = 1.0,
                   strength: float = 1.0, imgsz: int = 640) -> list[int]:
    """Ставит фиксированный фильтр в шею. Идемпотентно.

    placement:
        'upsample' — на операторы повышения разрешения (слои 11 и 14 у yolo26s);
                     основной вариант, довод FreqFusion;
        'levels'   — на выходы уровней, куда смотрит голова (16/19/22);
                     то же место, где стоял старый DRB, — прямое сравнение
                     «то же место, но фильтр фиксирован».

    Возвращает индексы обёрнутых слоёв.
    """
    if has_lowpass(det_model):
        return [m.i for m in det_model.model if isinstance(m, SmoothedLayer)]

    if placement == 'upsample':
        indices = upsample_indices(det_model)
        if not indices:
            raise RuntimeError('в модели не найдено ни одного nn.Upsample')
    elif placement == 'levels':
        indices = list(det_model.model[-1].f)
    else:
        raise ValueError(f"placement: upsample|levels, получено {placement!r}")

    ch = _probe_out_channels(det_model, indices, imgsz)
    device = next(det_model.parameters()).device
    for idx in indices:
        filt = FixedLowPass(ch[idx], kernel_size=kernel_size,
                            sigma=sigma, strength=strength).to(device)
        det_model.model[idx] = SmoothedLayer(det_model.model[idx], filt)
    return indices


def trainable_in_filters(det_model: nn.Module) -> int:
    """Сколько обучаемых чисел добавил фильтр. Должно быть РОВНО ноль."""
    return sum(p.numel() for m in det_model.model
               if isinstance(m, SmoothedLayer)
               for p in m.filt.parameters() if p.requires_grad)
