"""Перенос весов сохраняет смысл групп при перестановке коннектома."""
from types import SimpleNamespace
from copy import deepcopy
import pytest
import torch
from tracking.flyvis_connectome_swap import transfer_free_parameters, FREE
from tracking.flyvis_checkpoint import load_network


def network(node_keys=("A", "B"), edge_keys=(("A", "B"), ("B", "A")), values=(1., 2.)):
    return SimpleNamespace(nodes_bias=torch.nn.Parameter(torch.tensor(values, device="cpu")),
        nodes_time_const=torch.nn.Parameter(torch.tensor(values, device="cpu")),
        edges_syn_strength=torch.nn.Parameter(torch.tensor(values, device="cpu")),
        edges_sign=torch.tensor([-1., 1.], device="cpu"), edges_syn_count=torch.tensor([3., 4.], device="cpu"),
        node_params={k:SimpleNamespace(keys=list(node_keys)) for k in ("bias", "time_const")},
        edge_params={"syn_strength":SimpleNamespace(keys=list(edge_keys))})


def test_reordered_types_map_values_and_keep_fixed_connectome():
    source=network();before=deepcopy(source)
    target=network(node_keys=("B", "A"), edge_keys=(("B", "A"), ("A", "B")), values=(0.,0.))
    transfer_free_parameters(target, source)
    for name in FREE:
        assert torch.equal(getattr(target,name),torch.tensor([2.,1.]))
        assert torch.equal(getattr(source,name),getattr(before,name))
    assert torch.equal(target.edges_sign,source.edges_sign)
    assert torch.equal(target.edges_syn_count,source.edges_syn_count)


@pytest.mark.parametrize("keys", [(("A","B"),("A","B")), (("A","B"),("A","C"))])
def test_same_shape_wrong_or_duplicate_groups_fail_before_any_copy(keys):
    source=network();target=network(edge_keys=keys,values=(0.,0.));before=deepcopy(target)
    with pytest.raises(ValueError): transfer_free_parameters(target,source)
    assert all(torch.equal(getattr(target,n),getattr(before,n)) for n in FREE)


def test_identical_order_is_exact():
    source=network();target=network(values=(0.,0.))
    transfer_free_parameters(target,source)
    assert all(torch.equal(getattr(source,n),getattr(target,n)) for n in FREE)


def test_network_only_loader_requires_network_but_not_unused_decoder():
    original=torch.nn.Linear(2,2,device="cpu")
    view=SimpleNamespace(get_checkpoint=lambda key:{"network":original.state_dict()},
        network_class=lambda **kw:torch.nn.Linear(2,2,device="cpu"),
        dir=SimpleNamespace(config=SimpleNamespace(network=SimpleNamespace(to_dict=lambda:{}))))
    loaded=load_network(view)
    assert not loaded.training and torch.equal(loaded.weight,original.weight)
    view.get_checkpoint=lambda key:{"decoder":{"flow":{}}}
    with pytest.raises(ValueError,match="network"):load_network(view)
