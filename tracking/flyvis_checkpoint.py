"""Строгая загрузка одной согласованной контрольной точки FlyVis.

Штатные recover_network/recover_decoder лишь предупреждают при отсутствующем
разделе и сохраняют случайную инициализацию. Здесь неполные, несовместимые или
нечисловые веса отвергаются до использования модели. Исходный словарь не
изменяется; vendor-код и файлы контрольных точек не переписываются.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import copy
from pathlib import Path
from typing import Any


def read_checkpoint(source: str | Path | Mapping[str, Any], *, required_sections=("network", "decoder")) -> Mapping[str, Any]:
    """Прочитать доверенный локальный файл либо проверить переданный словарь."""
    import torch

    if isinstance(source, (str, Path)):
        source = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(source, Mapping):
        raise ValueError("Контрольная точка FlyVis должна быть словарём")
    for key in required_sections:
        value = source.get(key)
        if not isinstance(value, Mapping) or not value:
            raise ValueError(f"Отсутствует непустой раздел весов FlyVis: {key}")
    return source


def validate_state(module, state: Mapping[str, Any], label: str) -> None:
    """Проверить все ключи, размеры и конечность до копирования параметров."""
    import torch

    if not isinstance(state, Mapping):
        raise ValueError(f"{label}: ожидается словарь параметров")
    expected = module.state_dict()
    converted_state = {}
    missing, extra = set(expected) - set(state), set(state) - set(expected)
    if missing or extra:
        raise ValueError(f"{label}: missing={sorted(missing)}, unexpected={sorted(extra)}")
    for key, value in state.items():
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"{label}.{key}: ожидается тензор")
        if value.shape != expected[key].shape:
            raise ValueError(f"{label}.{key}: размер {tuple(value.shape)} вместо {tuple(expected[key].shape)}")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{label}.{key}: веса содержат NaN или Inf")
        target = expected[key]
        if value.is_complex() != target.is_complex():
            raise ValueError(f"{label}.{key}: несовместимые комплексные/вещественные веса")
        # load_state_dict casts to the destination dtype. Finite float64 values
        # can overflow a float32/float16 model, so validate the actual cast too.
        converted = value.detach().to(device="cpu", dtype=target.dtype)
        if not bool(torch.isfinite(converted).all()):
            raise ValueError(f"{label}.{key}: переполнение при приведении к {target.dtype}")
        if not target.is_floating_point() and not target.is_complex():
            if not torch.equal(value.detach().cpu(), converted.to(dtype=value.dtype)):
                raise ValueError(f"{label}.{key}: потеря значения при приведении к {target.dtype}")
        converted_state[key] = converted
    validate_parameter_domains(module, converted_state, label)


def validate_parameter_domains(module, state: Mapping[str, Any], label: str) -> None:
    """Проверить ограничения FlyVis и результат преобразования raw-параметров.

    Проверка не вызывает Network.clamp: тот пропускает замороженные параметры.
    Копия распределения получает отдельный тензор, исходные веса не изменяются.
    Модули без API параметров FlyVis сохраняют обычный контракт state_dict.
    """
    import torch

    if not hasattr(module, "clamp_config"):
        return
    for route, attribute in (("nodes", "node_params"), ("edges", "edge_params")):
        for field, parameter in getattr(module, attribute).items():
            key = f"{route}_{field}"
            value = state[key].detach().cpu()
            mode = module.clamp_config[key]
            if mode is None:
                projected = value
            elif isinstance(mode, str) and mode == "non_negative":
                projected = value.clamp(min=0)
            elif isinstance(mode, (tuple, list)) and len(mode) == 2:
                projected = value.clamp(min=mode[0], max=mode[1])
            else:
                raise ValueError(f"{label}.{key}: неподдерживаемое ограничение {mode!r}")
            if not torch.equal(value, projected):
                raise ValueError(f"{label}.{key}: нарушено объявленное ограничение {mode!r}")
            distribution = copy(parameter.parameter)
            distribution.raw_values = value.clone()
            semantic = distribution.semantic_values
            if not isinstance(semantic, torch.Tensor) or not bool(torch.isfinite(semantic).all()):
                raise ValueError(f"{label}.{key}: semantic_values содержат NaN или Inf")


def load_checkpoint_modules(network, decoders, source):
    """Загрузить ядро и все требуемые декодеры; входной словарь сохраняется.

    Структурная проверка всех модулей предшествует первой записи весов. Это
    предотвращает частичную загрузку при ошибке в одном из разделов.
    """
    payload = read_checkpoint(source)
    if not decoders:
        raise ValueError("Требуется хотя бы один декодер FlyVis")
    validate_state(network, payload["network"], "network")
    for name, decoder in decoders.items():
        if name not in payload["decoder"]:
            raise ValueError(f"Отсутствует раздел decoder/{name}")
        validate_state(decoder, payload["decoder"][name], f"decoder/{name}")
    network.load_state_dict(payload["network"], strict=True)
    for name, decoder in decoders.items():
        decoder.load_state_dict(payload["decoder"][name], strict=True)
    return network, decoders


def load_network_and_decoder(view, checkpoint="best"):
    """Построить ядро и декодер по одному view и одному разрешённому файлу.

    Разрешение ``best`` выполняется ровно один раз: выбор файла не может
    различаться между ядром и декодером. Режим eval и заморозку задаёт вызывающий
    код, поскольку этот загрузчик применяется и перед обучением.
    """
    from flyvis.task.tasks import init_decoder

    path = view.get_checkpoint(checkpoint)
    payload = read_checkpoint(path)
    network = view.network_class(**view.dir.config.network.to_dict())
    decoders = init_decoder(view.dir.config.task.decoder, view.connectome)
    return load_checkpoint_modules(network, decoders, payload)


def load_network(view, checkpoint="best"):
    """Строго загрузить только ядро; режим eval как у NetworkView.init_network."""
    payload = read_checkpoint(view.get_checkpoint(checkpoint), required_sections=("network",))
    network = view.network_class(**view.dir.config.network.to_dict())
    validate_state(network, payload["network"], "network")
    network.load_state_dict(payload["network"], strict=True)
    return network.eval()
