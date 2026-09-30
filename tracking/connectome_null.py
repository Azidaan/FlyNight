"""Нулевые модели коннектома FlyVis: перемешивание связей с сохранением степеней и знаков (карточка fly_connectome_null_v1).

`rewire(network, mode, seed)` переставляет получателей связей (`network._target_indices`) внутри групп:

* ``type_sign`` — одинаковый знак веса связи и одинаковый тип нейрона-получателя. Сохраняются:
  число входов и выходов каждого нейрона, знак и обученная сила каждой связи, число входящих
  связей каждого знака у каждого типа. Разрушаются: какой нейрон какого типа и в какой колонке
  питает данного получателя (ретинотопия и схема «тип → тип»). Сильная нулевая модель топологии
  (как нулевая модель с сохранением степеней, Dhiman, arXiv 2604.04033, 2026).
* ``spatial`` — одинаковые тип источника, тип получателя и знак: разрушается только
  пространственная (ретинотопическая) структура, схема «тип → тип» сохраняется.

Активности источников собираются по `_source_indices` (не меняются), токи суммируются по
`_target_indices` — поэтому перестановка `_target_indices` и есть переподключение. Параметры
нейронов и связей не меняются.
"""

from __future__ import annotations

import numpy as np

MODES = ("none", "type_sign", "spatial")


def rewire(network, mode: str, seed: int = 0) -> dict:
    import torch

    if mode not in MODES:
        raise ValueError(f"mode должен быть одним из {MODES}")
    if mode == "none":
        return {"mode": mode}
    with torch.inference_mode():
        weight = network._param_api().edges.weight.detach().cpu().numpy()
    sign = np.sign(weight).astype(np.int8)
    types = np.asarray(network.connectome.nodes.type[:])
    _, type_id = np.unique(types, return_inverse=True)
    src = network._source_indices.cpu().numpy()
    tgt = network._target_indices.cpu().numpy().copy()
    keys = [sign, type_id[tgt]] + ([type_id[src]] if mode == "spatial" else [])
    group = np.unique(np.stack(keys, axis=1), axis=0, return_inverse=True)[1].reshape(-1)
    rng = np.random.default_rng(seed)
    new_tgt = tgt.copy()
    order = np.argsort(group, kind="stable")
    bounds = np.flatnonzero(np.diff(group[order])) + 1
    for block in np.split(order, bounds):
        new_tgt[block] = tgt[block][rng.permutation(len(block))]
    changed = float(np.mean(new_tgt != tgt))
    in_deg_same = bool(np.array_equal(np.bincount(tgt, minlength=len(types)), np.bincount(new_tgt, minlength=len(types))))
    network._target_indices = torch.as_tensor(new_tgt, dtype=network._target_indices.dtype, device=network._target_indices.device)
    return {"mode": mode, "seed": int(seed), "edges": int(len(tgt)), "groups": int(group.max() + 1), "fraction_changed": changed, "in_degree_preserved": in_deg_same}
