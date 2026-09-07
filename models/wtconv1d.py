"""WTConv1d и предсказатель движения на его основе.

Слой воспроизводит устройство WTConv из статьи «Wavelet Convolutions for Large
Receptive Fields» (Finder, Amoyal, Treister, Freifeld, ECCV 2024), перенесённое
на одномерный ряд: код авторов открыт под MIT
(github.com/BGU-CS-VIL/WTConv), здесь он написан заново по опубликованному
устройству, чтобы не тащить в проект внешнюю зависимость. Соответствие
проверяемо по трём признакам, которые и составляют метод:

    1. вейвлет-разложение выполняется как ГРУППОВАЯ СВЁРТКА с шагом 2
       фильтрами, построенными через pywt; сами фильтры НЕ обучаются;
    2. на каждом уровне и на каждой полосе стоит СВОЯ обучаемая свёртка
       (глубинная, ядро kernel_size) и свой обучаемый масштаб с малым
       начальным значением 0.1;
    3. параллельно идёт обычная глубинная свёртка по исходному ряду, и её
       выход СКЛАДЫВАЕТСЯ с восстановленным из полос — то есть при нулевом
       вкладе вейвлетной ветви слой вырождается в обычную свёртку.

Смысл конструкции: за L уровней разложения слой видит участок ряда длиной
порядка 2^L·kernel_size, а обучаемых параметров тратит как на маленькое окно.

Базис задаётся именем pywt (`wt_type`): 'db1' — это Хаар, значение авторов по
умолчанию. Для нашей задачи это главный варьируемый параметр: на изображениях
у авторов лучшим оказался Хаар, а на траекториях в гейте Б он был худшим.

Контрольная сеть PlainConvPredictor устроена идентично, но вейвлетный слой в
ней заменён обычной глубинной свёрткой того же размера ядра. Без неё нельзя
отличить вклад вейвлетной конструкции от вклада самого факта обучения.
"""
import sys
from pathlib import Path

import numpy as np
import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def create_1d_wavelet_filter(wave: str, channels: int, dtype=torch.float32):
    """Фильтры разложения (2C,1,L) и восстановления (2C,1,L) для C каналов."""
    w = pywt.Wavelet(wave)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=dtype)
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=dtype)
    rec_lo = torch.tensor(w.rec_lo[::-1], dtype=dtype).flip(-1)
    rec_hi = torch.tensor(w.rec_hi[::-1], dtype=dtype).flip(-1)
    dec = torch.stack([dec_lo, dec_hi], dim=0)          # (2, L)
    rec = torch.stack([rec_lo, rec_hi], dim=0)
    dec = dec.unsqueeze(1).repeat(channels, 1, 1)       # (2C, 1, L)
    rec = rec.unsqueeze(1).repeat(channels, 1, 1)
    return dec, rec


def wavelet_transform_1d(x: torch.Tensor, filt: torch.Tensor) -> torch.Tensor:
    """(B,C,N) -> (B,C,2,N/2): полосы [низкая, высокая] на каждом канале."""
    b, c, n = x.shape
    pad = filt.shape[-1] - 2
    y = F.conv1d(x, filt, stride=2, groups=c, padding=pad)
    return y.reshape(b, c, 2, y.shape[-1])


def inverse_wavelet_transform_1d(x: torch.Tensor, filt: torch.Tensor) -> torch.Tensor:
    """(B,C,2,N/2) -> (B,C,N)."""
    b, c, _, half = x.shape
    pad = filt.shape[-1] - 2
    y = x.reshape(b, c * 2, half)
    return F.conv_transpose1d(y, filt, stride=2, groups=c, padding=pad)


class _Scale(nn.Module):
    """Обучаемый масштаб на канал; малый старт — вклад ветви растёт постепенно."""

    def __init__(self, channels: int, init: float = 1.0):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1) * init)

    def forward(self, x):
        return self.weight * x


class WTConv1d(nn.Module):
    """Свёртка в вейвлет-области для одномерного ряда.

    Число входных и выходных каналов обязано совпадать — как и у авторов:
    слой заменяет ГЛУБИННУЮ свёртку, а не произвольную.
    """

    def __init__(self, channels: int, kernel_size: int = 5, wt_levels: int = 2,
                 wt_type: str = 'db1', bias: bool = True):
        super().__init__()
        self.channels = channels
        self.wt_levels = wt_levels
        self.wt_type = wt_type

        dec, rec = create_1d_wavelet_filter(wt_type, channels)
        self.wt_filter = nn.Parameter(dec, requires_grad=False)
        self.iwt_filter = nn.Parameter(rec, requires_grad=False)

        self.base_conv = nn.Conv1d(channels, channels, kernel_size,
                                   padding='same', groups=channels, bias=bias)
        self.base_scale = _Scale(channels, 1.0)

        # на каждом уровне — своя свёртка по обеим полосам сразу (2C каналов)
        self.wavelet_convs = nn.ModuleList([
            nn.Conv1d(channels * 2, channels * 2, kernel_size, padding='same',
                      groups=channels * 2, bias=False)
            for _ in range(wt_levels)])
        self.wavelet_scale = nn.ModuleList([
            _Scale(channels * 2, 0.1) for _ in range(wt_levels)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, n = x.shape
        x_lo_levels, x_hi_levels, shapes = [], [], []
        cur = x
        for i in range(self.wt_levels):
            shapes.append(cur.shape[-1])
            if cur.shape[-1] % 2:                      # нечётная длина — дополнить
                cur = F.pad(cur, (0, 1))
            bands = wavelet_transform_1d(cur, self.wt_filter)   # (B,C,2,N/2)
            half = bands.shape[-1]
            flat = bands.reshape(b, c * 2, half)
            flat = self.wavelet_scale[i](self.wavelet_convs[i](flat))
            bands = flat.reshape(b, c, 2, half)
            x_lo_levels.append(bands[:, :, 0])
            x_hi_levels.append(bands[:, :, 1])
            cur = bands[:, :, 0]

        next_lo = torch.zeros_like(x_lo_levels[-1])
        for i in range(self.wt_levels - 1, -1, -1):
            lo = x_lo_levels.pop() + next_lo
            hi = x_hi_levels.pop()
            merged = torch.stack([lo, hi], dim=2)
            next_lo = inverse_wavelet_transform_1d(merged, self.iwt_filter)
            next_lo = next_lo[:, :, :shapes[i]] if i else next_lo[:, :, :n]

        wave_out = next_lo[:, :, :n]
        if wave_out.shape[-1] < n:                     # страховка на краю
            wave_out = F.pad(wave_out, (0, n - wave_out.shape[-1]))
        return self.base_scale(self.base_conv(x)) + wave_out


class _Block(nn.Module):
    """Смешивание по времени (вейвлетное или обычное) плюс смешивание каналов."""

    def __init__(self, channels: int, kernel_size: int, wt_levels: int,
                 wt_type: str, mixer: str):
        super().__init__()
        if mixer == 'wtconv':
            self.mix = WTConv1d(channels, kernel_size, wt_levels, wt_type)
        elif mixer == 'plain':
            # контроль: та же глубинная свёртка, но без вейвлетной части
            self.mix = nn.Conv1d(channels, channels, kernel_size,
                                 padding='same', groups=channels, bias=True)
        else:
            raise ValueError(f'неизвестный смеситель: {mixer}')
        self.norm = nn.BatchNorm1d(channels)
        self.pw1 = nn.Conv1d(channels, channels * 2, 1)
        self.pw2 = nn.Conv1d(channels * 2, channels, 1)

    def forward(self, x):
        y = self.norm(self.mix(x))
        return x + self.pw2(F.gelu(self.pw1(y)))


class TrajPredictor(nn.Module):
    """Предсказатель положения объекта через k кадров по окну наблюдений.

    Вход: приращения координат по кадрам (B, 2, W) в единицах масштаба.
    Выход: смещение от последнего наблюдения до положения через каждый из
    разрывов gaps — (B, len(gaps), 2).

    Модель принципиально НЕ видит разметку: только прошлое самого трека, как и
    фильтр Калмана, с которым она сравнивается.
    """

    def __init__(self, gaps=(2, 3, 5), channels: int = 48, blocks: int = 3,
                 kernel_size: int = 5, wt_levels: int = 2, wt_type: str = 'db1',
                 mixer: str = 'wtconv'):
        super().__init__()
        self.gaps = tuple(gaps)
        self.stem = nn.Conv1d(2, channels, 1)
        self.blocks = nn.Sequential(*[
            _Block(channels, kernel_size, wt_levels, wt_type, mixer)
            for _ in range(blocks)])
        self.head = nn.Sequential(
            nn.Linear(channels * 2, channels * 2), nn.GELU(),
            nn.Linear(channels * 2, len(self.gaps) * 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.blocks(self.stem(x))
        feat = torch.cat([h[:, :, -1], h.mean(dim=-1)], dim=1)
        return self.head(feat).reshape(-1, len(self.gaps), 2)


def build_model(mixer: str = 'wtconv', wt_type: str = 'db1', **kw) -> TrajPredictor:
    return TrajPredictor(mixer=mixer, wt_type=wt_type, **kw)
