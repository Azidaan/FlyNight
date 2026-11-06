"""Строгая загрузка: отказ до изменения модели и сохранность входных весов."""
from copy import deepcopy
from types import SimpleNamespace, ModuleType
import sys

import pytest
import torch

from tracking.flyvis_checkpoint import read_checkpoint, load_checkpoint_modules, load_network_and_decoder


def modules():
    return torch.nn.Linear(2, 3, device="cpu"), {"flow": torch.nn.Linear(3, 2, device="cpu")}


def payload(network, decoders):
    return {"network": deepcopy(network.state_dict()),
            "decoder": {k: deepcopy(v.state_dict()) for k, v in decoders.items()}}


@pytest.mark.parametrize("bad", [None, [], {}, {"network": {}},
    {"network": {"x": 1}, "decoder": {}}, {"network": [], "decoder": {"flow": {}}}])
def test_required_sections_rejected(bad):
    with pytest.raises(ValueError):
        read_checkpoint(bad)


@pytest.mark.parametrize("fault", ["missing_flow", "missing_key", "extra_key", "wrong_shape", "nan", "inf", "not_tensor"])
def test_invalid_decoder_rejected_before_network_changes(fault):
    network, decoders = modules()
    before = deepcopy(network.state_dict())
    source = payload(network, decoders)
    source["network"]["weight"] += 7
    if fault == "missing_flow":
        source["decoder"] = {"another_task": source["decoder"]["flow"]}
    else:
        state = source["decoder"]["flow"]
        if fault == "missing_key": del state["bias"]
        elif fault == "extra_key": state["foreign"] = torch.ones(1, device="cpu")
        elif fault == "wrong_shape": state["weight"] = torch.ones(1, device="cpu")
        elif fault == "not_tensor": state["weight"] = [1, 2]
        else: state["weight"][0, 0] = float(fault)
    with pytest.raises(ValueError):
        load_checkpoint_modules(network, decoders, source)
    assert all(torch.equal(v, before[k]) for k, v in network.state_dict().items())


def test_valid_checkpoint_reusable_without_mutation(tmp_path):
    network, decoders = modules()
    source = payload(network, decoders)
    expected = deepcopy(source)
    path = tmp_path / "checkpoint.pt"
    torch.save(source, path)
    for data in (source, source, path):
        target, target_decoders = modules()
        load_checkpoint_modules(target, target_decoders, data)
        assert all(torch.equal(v, expected["network"][k]) for k, v in target.state_dict().items())
        assert all(torch.equal(v, expected["decoder"]["flow"][k]) for k, v in target_decoders["flow"].state_dict().items())
    assert source.keys() == expected.keys()
    assert source["decoder"].keys() == expected["decoder"].keys()
    assert all(torch.equal(v, expected["decoder"]["flow"][k]) for k, v in source["decoder"]["flow"].items())


def test_finite_high_precision_weight_cannot_overflow_loaded_model():
    network, decoders = modules()
    source = payload(network, decoders)
    source["network"]["weight"] = torch.full((3, 2), 1e300, device="cpu", dtype=torch.float64)
    before = deepcopy(network.state_dict())
    with pytest.raises(ValueError, match="переполнение"):
        load_checkpoint_modules(network, decoders, source)
    assert all(torch.equal(v, before[k]) for k, v in network.state_dict().items())


def test_view_resolves_checkpoint_only_once(monkeypatch):
    source_net, source_dec = modules()
    source = payload(source_net, source_dec)
    calls = []
    def resolve(key):
        calls.append(key)
        assert len(calls) == 1
        return source
    tasks = ModuleType("flyvis.task.tasks")
    tasks.init_decoder = lambda config, connectome: modules()[1]
    monkeypatch.setitem(sys.modules, tasks.__name__, tasks)
    view = SimpleNamespace(get_checkpoint=resolve, network_class=lambda **kw: modules()[0],
        dir=SimpleNamespace(config=SimpleNamespace(network=SimpleNamespace(to_dict=lambda: {}),
            task=SimpleNamespace(decoder={}))), connectome=None)
    network, decoder = load_network_and_decoder(view)
    assert calls == ["best"]
    assert torch.equal(network.weight, source_net.weight)
    assert torch.equal(decoder["flow"].weight, source_dec["flow"].weight)
@pytest.mark.parametrize('target_dtype,value', [
    (torch.int64, torch.tensor(1.5, device='cpu')),
    (torch.int8, torch.tensor(256, device='cpu')),
    (torch.bool, torch.tensor(2, device='cpu')),
])
def test_integer_and_boolean_casts_cannot_silently_change_values(target_dtype, value):
    from tracking.flyvis_checkpoint import validate_state
    module = torch.nn.Module()
    module.register_buffer('count', torch.tensor(0, dtype=target_dtype, device='cpu'))
    with pytest.raises(ValueError, match='потеря значения'):
        validate_state(module, {'count': value}, 'buffer')


def test_exact_integer_cast_is_allowed():
    from tracking.flyvis_checkpoint import validate_state
    module = torch.nn.Module()
    module.register_buffer('count', torch.tensor(0, dtype=torch.int64, device='cpu'))
    validate_state(module, {'count': torch.tensor(1.0, device='cpu')}, 'buffer')

