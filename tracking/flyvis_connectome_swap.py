"""Варианты коннектома FlyVis с числами синапсов из FlyWire (К16, ступень 1; карточка fly_flywire_swap_v1).

Штатная муха не меняется: варианты — отдельные JSON-файлы в формате FlyVis в data/flywire/flyvis_variants/.
  stock          — копия штатного fib25-fib19_v2.2.json (для сверки пути загрузки);
  flywire        — те же 65 типов и 605 рёбер; для рёбер между типами, присутствующими в каждой колонке (не Lawf1/Lawf2),
                   у которых оба типа найдены в FlyWire и там ≥ 0,5 синапса
                   на клетку-приёмник, распределение по смещениям колонок — из FlyWire (runs/flywire_flyvis/flywire_counts.json,
                   оси уже приведены к FlyVis; смещения с < 0,05 синапса и дальше радиуса решётки отброшены); остальные — FlyVis;
                   Общее число синапсов ребра сохраняется как в FlyVis («та же доза»), из FlyWire берётся профиль по смещениям;
                   число смещений ребра — как в FlyVis (k самых сильных из FlyWire: «та же разреженность»).
  flywire_shuf   — контроль: профили FlyWire перемешаны между заменёнными рёбрами одного знака (seed 0), доза — своя.
make_network(json, model) строит сеть FlyVis из файла и переносит обученные свободные параметры модели (nodes_bias,
nodes_time_const, edges_syn_strength — по парам типов, их набор не меняется); закреплённые edges_sign и edges_syn_count
берутся из нового файла.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FV_JSON = PROJECT_ROOT / "tools/flyvis-runtime/flyvis/connectome/fib25-fib19_v2.2.json"
FW_COUNTS = PROJECT_ROOT / "runs/flywire_flyvis/flywire_counts.json"
VARIANTS = PROJECT_ROOT / "data/flywire/flyvis_variants"
FREE = ("nodes_bias", "nodes_time_const", "edges_syn_strength")
EXTENT = 15
MIN_TOTAL, MIN_OFFSET = 0.5, 0.05


def hexdist(u, v):
    return max(abs(u), abs(v), abs(u + v))


def _write(name: str, spec: dict) -> Path:
    """Имя файла содержит хэш содержимого: FlyVis кэширует сборку связей по пути файла (каталог datamate),
    и правка файла с прежним именем молча брала бы старую сборку (так было 08.10)."""
    import hashlib
    text = json.dumps(spec, sort_keys=True)
    path = VARIANTS / f"{name}_{hashlib.md5(text.encode()).hexdigest()[:8]}.json"
    path.write_text(text, encoding="utf-8")
    return path


def build_variants(seed: int = 0) -> dict:
    VARIANTS.mkdir(parents=True, exist_ok=True)
    fv = json.loads(FV_JSON.read_text(encoding="utf-8"))
    # Lawf1 → Lawf1 (смещение [1, 0]) в штатной сети не реализуется (нет клеток Lawf1 со сдвигом при шаге [3, 2]), но в
    # собранных вариантах появлялось — убирается из всех файлов, включая штатную копию (сверка активности — без изменений)
    fv["edges"] = [e for e in fv["edges"] if not (e["src"] == "Lawf1" and e["tar"] == "Lawf1")]
    fw = json.loads(FW_COUNTS.read_text(encoding="utf-8"))
    paths = {"stock": _write("stock", fv)}
    new, replaced = json.loads(json.dumps(fv)), []
    sparse = {nd["name"] for nd in fv["nodes"] if nd["pattern"] != ["stride", [1, 1]]}   # Lawf1, Lawf2 — не в каждой колонке
    for e in new["edges"]:
        k = f"{e['src']}->{e['tar']}"
        if k not in fw or e["src"] in sparse or e["tar"] in sparse:
            continue   # у разрежённых типов чужие смещения могут не попадать на клетки — оставить FlyVis
        offs = [[list(map(int, o)), float(w)] for o, w in fw[k] if w >= MIN_OFFSET and hexdist(int(o[0]), int(o[1])) <= EXTENT]
        # та же разреженность: k самых сильных смещений FlyWire, k = числу смещений ребра в FlyVis (иначе сеть ×3 тяжелее —
        # 7408 чисел синапсов против 2355 — и 6 процессов сигнатур не помещались в 12 ГБ GPU, 08.10)
        offs = sorted(offs, key=lambda x: -x[1])[:len(e["offsets"])]
        tot_fw = sum(w for _, w in offs)
        if tot_fw >= MIN_TOTAL and offs:
            # «та же доза»: общее число синапсов ребра — как в FlyVis, из FlyWire — только профиль по смещениям колонок
            # (полные числа FlyWire при силах связей, обученных под FlyVis, дают взрыв динамики — проверено 08.10)
            tot_fv = sum(float(w) for _, w in e["offsets"])
            e["offsets"] = [[o, w * tot_fv / tot_fw] for o, w in offs]; replaced.append(k)
    paths["flywire"] = _write("flywire", new)
    shuf = json.loads(json.dumps(new)); rng = np.random.default_rng(seed)
    by = {}
    for i, e in enumerate(shuf["edges"]):
        if f"{e['src']}->{e['tar']}" in replaced:
            by.setdefault(e["alpha"], []).append(i)
    for sign, idx in by.items():
        prof = [new["edges"][i]["offsets"] for i in idx]
        tots = [sum(w for _, w in pr) for pr in prof]
        perm = rng.permutation(len(idx))
        for i, j, tot in zip(idx, perm, tots):   # чужой профиль, своя доза
            tj = sum(w for _, w in prof[j])
            shuf["edges"][i]["offsets"] = [[o, w * tot / tj] for o, w in prof[j]]
    paths["flywire_shuf"] = _write("flywire_shuf", shuf)
    info = {"edges_total": len(fv["edges"]), "edges_replaced": len(replaced), "replaced": replaced, "by_sign": {str(k): len(v) for k, v in by.items()},
            "min_total": MIN_TOTAL, "min_offset": MIN_OFFSET, "extent": EXTENT, "seed": seed,
            "files": {k: str(v.relative_to(PROJECT_ROOT)) for k, v in paths.items()}}
    (VARIANTS / "variants_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    return info


def transfer_free_parameters(destination, source) -> None:
    """Перенести свободные параметры по типам клеток/пар, не по позиции.

    Равенство форм не подтверждает соответствие биологических групп: порядок
    групп зависит от порядка узлов и рёбер в JSON. Все соответствия проверяются
    до первой записи, закреплённые знак и число синапсов не изменяются.
    """
    import torch

    copies = []
    for name in FREE:
        dst, src = getattr(destination, name), getattr(source, name)
        route, field = name.split("_", 1)
        attribute = "node_params" if route == "nodes" else "edge_params"
        dst_keys = list(getattr(destination, attribute)[field].keys)
        src_keys = list(getattr(source, attribute)[field].keys)
        if len(set(src_keys)) != len(src_keys) or len(set(dst_keys)) != len(dst_keys):
            raise ValueError(f"{name}: неоднозначные повторяющиеся ключи групп")
        if set(src_keys) != set(dst_keys):
            raise ValueError(f"{name}: набор типов клеток/пар отличается от штатного")
        if dst.shape != src.shape or dst.ndim != 1 or len(dst_keys) != len(dst) or len(src_keys) != len(src):
            raise ValueError(f"{name}: форма параметров не соответствует ключам групп")
        lookup = {key: i for i, key in enumerate(src_keys)}
        order = torch.as_tensor([lookup[key] for key in dst_keys], device=src.device)
        mapped = src.detach().index_select(0, order).to(device=dst.device, dtype=dst.dtype)
        if not bool(torch.isfinite(mapped).all()):
            raise ValueError(f"{name}: нечисловые параметры при переносе")
        copies.append((dst, mapped))
    with torch.no_grad():
        for dst, mapped in copies:
            dst.copy_(mapped)


def make_network(json_path, model: str = "flow/0000/000", *, device=None):
    """Сеть FlyVis из файла связей json_path с обученными свободными параметрами модели model."""
    from tracking.flyvis_adapter import activate_flyvis
    from tracking.flyvis_checkpoint import load_network
    flyvis = activate_flyvis(device=device)
    from flyvis.network.network import Network
    view = flyvis.NetworkView(model)
    ref = load_network(view)                      # штатная сеть с проверенными весами
    cfg = view.dir.config.network.to_dict()
    cfg["connectome"] = dict(cfg["connectome"]); cfg["connectome"]["file"] = str(Path(json_path).resolve())
    net = Network(**cfg)
    transfer_free_parameters(net, ref)
    net.eval().requires_grad_(False)
    return net, ref, view
