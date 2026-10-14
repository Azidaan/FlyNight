"""Separate inference-only compiler adapter; source models and math stay fixed."""
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/z13_triton_hota_v1_20261014'


def setup():
    sys.path.insert(0, str(RUN / 'packages'))
    os.environ.setdefault('TRITON_CACHE_DIR', str(RUN / 'triton_cache'))
    os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR', str(RUN / 'inductor_cache'))
    os.environ.setdefault('TRITON_HOME', str(RUN / 'triton_home'))
    os.environ['CC'] = str(RUN / 'packages/triton/runtime/tcc/tcc.exe')
    os.environ['OMP_NUM_THREADS'] = '2'
    os.environ['MKL_NUM_THREADS'] = '2'


def enable(localizer):
    """Compile tensor arithmetic, keeping tracker, crops and readout unchanged."""
    import torch
    if localizer.arm == 'B':
        original = localizer.bb.step
        localizer.bb.step = torch.compile(original, backend='inductor', fullgraph=True,
                                           dynamic=True)
        return {'arm': 'B', 'compiled': 'HexGRU.step'}
    net = localizer.bb.net
    from flyvis.utils.tensor_utils import AutoDeref
    assert not net._state_hooks, 'State hooks are outside the fixed adapter contract'
    assert isinstance(net.dynamics.activation, torch.nn.ReLU), 'Only the actual ReLU model is supported'
    assert all(not p.requires_grad for p in net.parameters()), 'Inference only'
    source, target = net._source_indices, net._target_indices
    n_nodes = net.n_nodes

    def tensor_step(activity, stimulus, bias, time_const, weight, dt):
        edge = weight * torch.relu(activity.index_select(-1, source))
        incoming = torch.zeros_like(activity)
        incoming.scatter_add_(-1, target.expand_as(edge), edge)
        velocity = (1 / torch.maximum(time_const, activity.new_full((), dt))) * (
            -activity + bias + incoming + stimulus)
        return activity + velocity * dt

    compiled = torch.compile(tensor_step, backend='inductor', fullgraph=True, dynamic=True)

    def next_state(params, state, x_t, dt):
        assert tuple(state.nodes.keys()) == ('activity',) and not state.edges
        activity = compiled(state.nodes.activity, x_t, params.nodes.bias,
                            params.nodes.time_const, params.edges.weight, float(dt))
        return net._state_api(AutoDeref(nodes=AutoDeref(activity=activity), edges=AutoDeref()))

    net._next_state = next_state
    return {'arm': localizer.arm, 'compiled': 'FlyVis Euler step + edge gather/scatter',
            'nodes': n_nodes, 'edges': len(source)}


def install():
    from tracking.z13_spatial_localizer import SpatialLocalizer
    original = SpatialLocalizer.__init__

    def init(self, *args, **kwargs):
        original(self, *args, **kwargs)
        self.compiler_info = enable(self)

    SpatialLocalizer.__init__ = init
