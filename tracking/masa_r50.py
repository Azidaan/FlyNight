"""MASA-R50 (Li et al., CVPR 2024, «Matching Anything by Segmenting Anything») без mmdetection: только путь эмбеддинга.

Веса — weights/masa/masa_r50.pth (HF dereksiyuanli/masa, Apache-2.0). Собрано по конфигу configs/masa-one/masa_r50_plug_and_play.py
и исходникам masa/models (копии для сверки — tools/masa_ref/):
  ResNet-50 (стиль caffe: шаг в первой свёртке 1×1, BN в режиме оценки) → FPN (256, 5 выходов, BN в боковых и выходных
  свёртках, пятый уровень — max-pool) → DeformFusion (3 блока DyHead: модулированные деформируемые свёртки
  torchvision.ops.deform_conv2d, смещения и маски из своего уровня) → RoIAlign 7×7 по рамке на уровне, выбранном по
  масштабу (как SingleRoIExtractor mmdet, finest_scale = 56) → QuasiDenseEmbedHead (4 свёртки 3×3 + GN(32) + ReLU,
  1 полносвязный 1024 + ReLU, выход 256). Вход — кадр BGR, приведение к длинной стороне 1024, нормировка ImageNet,
  дополнение до кратного 32. Выход — эмбеддинг на рамку (L2-норма — снаружи, как в трекере MASA — косинусная близость).
"""
from __future__ import annotations

import math
import pickle
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import deform_conv2d, roi_align

WEIGHTS = Path(__file__).resolve().parents[1] / "weights/masa/masa_r50.pth"


class _StubMeta(type):
    def __getattr__(cls, name):   # пустышка для любых атрибутов служебных классов mmengine
        return _Stub


class _Stub(metaclass=_StubMeta):
    def __init__(self, *a, **k):
        pass

    def __call__(self, *a, **k):
        return _Stub()

    def __setstate__(self, state):
        self.__dict__["_state"] = state


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except Exception:
            return _Stub


def load_state_dict(path=WEIGHTS) -> dict:
    mod = types.ModuleType("masa_pickle"); mod.Unpickler = _Unpickler; mod.load = pickle.load; mod.__name__ = "masa_pickle"
    ck = torch.load(str(path), map_location="cpu", pickle_module=mod, weights_only=False)
    return ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, stride=stride, bias=False); self.bn1 = nn.BatchNorm2d(planes)   # caffe: шаг здесь
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False); self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, 1, bias=False); self.bn3 = nn.BatchNorm2d(planes * 4)
        self.downsample = downsample

    def forward(self, x):
        idt = x if self.downsample is None else self.downsample(x)
        x = F.relu(self.bn1(self.conv1(x))); x = F.relu(self.bn2(self.conv2(x))); x = self.bn3(self.conv3(x))
        return F.relu(x + idt)


class ResNet50(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False); self.bn1 = nn.BatchNorm2d(64)
        self.inplanes = 64
        self.layer1 = self._make(64, 3, 1); self.layer2 = self._make(128, 4, 2); self.layer3 = self._make(256, 6, 2); self.layer4 = self._make(512, 3, 2)

    def _make(self, planes, blocks, stride):
        ds = None
        if stride != 1 or self.inplanes != planes * 4:
            ds = nn.Sequential(nn.Conv2d(self.inplanes, planes * 4, 1, stride=stride, bias=False), nn.BatchNorm2d(planes * 4))
        layers = [Bottleneck(self.inplanes, planes, stride, ds)]; self.inplanes = planes * 4
        layers += [Bottleneck(self.inplanes, planes) for _ in range(1, blocks)]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), 3, 2, 1)
        outs = []
        for l in (self.layer1, self.layer2, self.layer3, self.layer4):
            x = l(x); outs.append(x)
        return outs


class ConvBN(nn.Module):
    def __init__(self, cin, cout, k):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, padding=k // 2, bias=False); self.bn = nn.BatchNorm2d(cout)

    def forward(self, x):
        return self.bn(self.conv(x))


class FPN(nn.Module):
    def __init__(self):
        super().__init__()
        self.lateral_convs = nn.ModuleList([ConvBN(c, 256, 1) for c in (256, 512, 1024, 2048)])
        self.fpn_convs = nn.ModuleList([ConvBN(256, 256, 3) for _ in range(4)])

    def forward(self, feats):
        lat = [l(f) for l, f in zip(self.lateral_convs, feats)]
        for i in range(3, 0, -1):
            lat[i - 1] = lat[i - 1] + F.interpolate(lat[i], size=lat[i - 1].shape[-2:], mode="nearest")
        outs = [c(x) for c, x in zip(self.fpn_convs, lat)]
        outs.append(F.max_pool2d(outs[-1], 1, stride=2))
        return outs


class DyDCN(nn.Module):
    def __init__(self, c, stride=1):
        super().__init__()
        self.stride = stride
        self.conv = nn.Module(); self.conv.weight = nn.Parameter(torch.zeros(c, c, 3, 3))
        self.norm = nn.GroupNorm(16, c)

    def forward(self, x, offset, mask):
        ho, wo = (x.shape[-2] + 2 - 3) // self.stride + 1, (x.shape[-1] + 2 - 3) // self.stride + 1
        if offset.shape[-2:] != (ho, wo):
            # как в mmcv (ModulatedDeformConv2d не проверяет размер): ядро читает буфер смещений/маски как [C, ho, wo] —
            # для свёртки «сверху» DyHead (смещения с уровня крупнее) берутся первые C·ho·wo элементов буфера
            b = offset.shape[0]
            offset = offset.contiguous().reshape(b, -1)[:, : offset.shape[1] * ho * wo].reshape(b, offset.shape[1], ho, wo)
            mask = mask.contiguous().reshape(b, -1)[:, : mask.shape[1] * ho * wo].reshape(b, mask.shape[1], ho, wo)
        return self.norm(deform_conv2d(x.contiguous(), offset, self.conv.weight, None, stride=self.stride, padding=1, mask=mask))


class DyHeadBlock(nn.Module):
    def __init__(self, c=256):
        super().__init__()
        self.spatial_conv_offset = nn.Conv2d(c, 81, 3, padding=1)
        self.spatial_conv_high = DyDCN(c); self.spatial_conv_mid = DyDCN(c); self.spatial_conv_low = DyDCN(c, 2)

    def forward(self, x):
        outs = []
        for lvl in range(len(x)):
            om = self.spatial_conv_offset(x[lvl])
            offset, mask = om[:, :54], om[:, 54:].sigmoid()
            offs, masks = offset.split(18, 1), mask.split(9, 1)
            s = self.spatial_conv_mid(x[lvl], offs[0], masks[0]); n = 1
            if lvl > 0:
                s = s + self.spatial_conv_low(x[lvl - 1], offs[1], masks[1]); n += 1
            if lvl < len(x) - 1:
                s = s + F.interpolate(self.spatial_conv_high(x[lvl + 1], offs[2], masks[2]), size=x[lvl].shape[-2:], mode="bilinear", align_corners=True); n += 1
            outs.append(s / n)
        return outs


class EmbedHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.convs = nn.ModuleList()
        for _ in range(4):
            m = nn.Module(); m.conv = nn.Conv2d(256, 256, 3, padding=1, bias=False); m.gn = nn.GroupNorm(32, 256); self.convs.append(m)
        self.fcs = nn.ModuleList([nn.Linear(256 * 49, 1024)])
        self.fc_embed = nn.Linear(1024, 256)

    def forward(self, x):
        for m in self.convs:
            x = F.relu(m.gn(m.conv(x)))
        x = F.relu(self.fcs[0](x.flatten(1)))
        return self.fc_embed(x)


class MasaR50(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = ResNet50(); self.fpn = FPN(); self.fusion = nn.ModuleList([DyHeadBlock() for _ in range(3)]); self.embed = EmbedHead()

    def features(self, img):
        x = self.fpn(self.backbone(img))
        for b in self.fusion:
            x = b(x)
        return x

    def embed_boxes(self, feats, boxes, scale):
        """boxes — (N, 4) в пикселях исходного кадра; scale — множитель приведения кадра."""
        if not len(boxes):
            return torch.zeros((0, 256), device=feats[0].device)
        b = boxes * scale
        area = ((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])).clamp(min=1e-6).sqrt()
        lvl = torch.floor(torch.log2(area / 56 + 1e-6)).clamp(0, 3).long()
        out = torch.zeros((len(b), 256, 7, 7), device=b.device, dtype=feats[0].dtype)
        for i, s in enumerate((4, 8, 16, 32)):
            m = lvl == i
            if m.any():
                rois = torch.cat([torch.zeros((int(m.sum()), 1), device=b.device, dtype=b.dtype), b[m]], 1)
                out[m] = roi_align(feats[i], rois.to(feats[i].dtype), 7, spatial_scale=1.0 / s, sampling_ratio=0, aligned=True)
        return self.embed(out)


def _map_key(k: str) -> str | None:
    """Ключи mmdet → ключи этой сборки."""
    if k.startswith("backbone."):
        return k
    if k.startswith("masa_adapter.0."):
        k = k.replace("masa_adapter.0.", "fpn.")
        return k.replace(".conv.weight", ".conv.weight").replace(".bn.", ".bn.")
    if k.startswith("masa_adapter.1.dyhead_blocks."):
        return k.replace("masa_adapter.1.dyhead_blocks.", "fusion.")
    if k.startswith("track_head.embed_head."):
        return k.replace("track_head.embed_head.", "embed.")
    return None


def build(device="cuda") -> MasaR50:
    sd = load_state_dict()
    m = MasaR50()
    own = m.state_dict()
    mapped = {}
    for k, v in sd.items():
        nk = _map_key(k)
        if nk is not None and nk in own and tuple(own[nk].shape) == tuple(v.shape):
            mapped[nk] = v
    missing = [k for k in own if k not in mapped and not k.endswith("num_batches_tracked")]
    if missing:
        raise RuntimeError(f"не перенесены веса: {missing[:12]} … всего {len(missing)}")
    m.load_state_dict(mapped, strict=False)
    return m.eval().to(device)


def preprocess(frame_bgr: np.ndarray, device="cuda"):
    import cv2
    h, w = frame_bgr.shape[:2]
    s = 1024.0 / max(h, w)
    img = cv2.resize(frame_bgr, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_LINEAR)[:, :, ::-1].astype(np.float32)
    img = (img - np.array([123.675, 116.28, 103.53], np.float32)) / np.array([58.395, 57.12, 57.375], np.float32)
    H, W = img.shape[:2]; ph, pw = int(math.ceil(H / 32) * 32), int(math.ceil(W / 32) * 32)
    pad = np.zeros((ph, pw, 3), np.float32); pad[:H, :W] = img
    return torch.from_numpy(pad).permute(2, 0, 1)[None].to(device), s
