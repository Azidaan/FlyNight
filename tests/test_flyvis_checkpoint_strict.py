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


class _Distribution:
    def __init__(self, raw_values, *, exponential=False):
        self.raw_values = raw_values
        self.exponential = exponential

    @property
    def semantic_values(self):
        return self.raw_values.exp() if self.exponential else self.raw_values


def _domain_network():
    network = torch.nn.Module()
    network.node_params, network.edge_params, network.clamp_config = {}, {}, {}
    for name, values, mode, exponential in (
        ('nodes_bias', [1., 2.], None, False),
        ('nodes_time_const', [-1., 0.], None, False),
        ('edges_syn_strength', [.1, .2], 'non_negative', False),
        ('edges_syn_count', [-2., 1.], None, True),
    ):
        parameter = torch.nn.Parameter(torch.tensor(values, device='cpu'), requires_grad=False)
        network.register_parameter(name, parameter)
        route, field = name.split('_', 1)
        registry = network.node_params if route == 'nodes' else network.edge_params
        registry[field] = SimpleNamespace(parameter=_Distribution(parameter, exponential=exponential), keys=['a', 'b'])
        network.clamp_config[name] = mode
    return network


@pytest.mark.parametrize('key,value,match', [
    ('edges_syn_strength', -.1, 'ограничение'),
    ('edges_syn_count', 1000., 'semantic_values'),
])
def test_invalid_parameter_domain_rejected_before_any_write(key, value, match):
    network, decoders = _domain_network(), modules()[1]
    source = payload(network, decoders)
    source['network']['nodes_bias'] += 4
    source['network'][key][0] = value
    before, original = deepcopy(network.state_dict()), deepcopy(source)
    storages = {name: parameter.data_ptr() for name, parameter in network.named_parameters()}
    decoder_before = deepcopy(decoders['flow'].state_dict())
    with pytest.raises(ValueError, match=match):
        load_checkpoint_modules(network, decoders, source)
    assert all(torch.equal(value, before[name]) for name, value in network.state_dict().items())
    assert all(torch.equal(value, original['network'][name]) for name, value in source['network'].items())
    assert all(torch.equal(value, decoder_before[name]) for name, value in decoders['flow'].state_dict().items())
    for name, parameter in network.named_parameters():
        assert parameter.data_ptr() == storages[name]
        route, field = name.split('_', 1)
        registry = network.node_params if route == 'nodes' else network.edge_params
        assert registry[field].parameter.raw_values is parameter


def test_valid_domains_preserve_storage_and_allow_negative_logcount_and_tau():
    from tracking.flyvis_checkpoint import validate_state
    network = _domain_network()
    state, before = deepcopy(network.state_dict()), deepcopy(network.state_dict())
    pointers = {name: value.data_ptr() for name, value in network.state_dict().items()}
    for _ in range(2):
        validate_state(network, state, 'network')
    assert all(torch.equal(value, before[name]) for name, value in state.items())
    assert all(torch.equal(value, before[name]) and value.data_ptr() == pointers[name]
               for name, value in network.state_dict().items())


@pytest.mark.parametrize('mode,values,accepted', [
    ((0., 1.), [0., 1.], True),
    ([None, 1.], [-2., 1.], True),
    ([0., None], [0., 2.], True),
    ((0., 1.), [-.1, 1.], False),
    ('unknown', [0., 1.], False),
])
def test_only_declared_supported_bounds_are_applied(mode, values, accepted):
    from tracking.flyvis_checkpoint import validate_state
    network = _domain_network()
    network.clamp_config['nodes_bias'] = mode
    state = deepcopy(network.state_dict())
    state['nodes_bias'] = torch.tensor(values, device='cpu')
    if accepted:
        validate_state(network, state, 'network')
    else:
        with pytest.raises(ValueError):
            validate_state(network, state, 'network')

