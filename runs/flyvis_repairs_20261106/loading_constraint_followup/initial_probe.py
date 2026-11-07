"""Bounded CPU checkpoint-domain probe; never edit production or stored weights."""
from pathlib import Path
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
torch.set_num_threads(2)
torch.set_default_device('cpu')
assert not torch.cuda.is_initialized()
from tracking.flyvis_adapter import activate_flyvis
from tracking.flyvis_checkpoint import load_network_and_decoder, load_checkpoint_modules, read_checkpoint
from tracking.flyvis_connectome_swap import transfer_free_parameters

flyvis = activate_flyvis(device='cpu')
import yaml
view = flyvis.NetworkView('flow/0000/000')
selected = Path(view.get_checkpoint('best'))
stock_paths = sorted((ROOT/'runs/flyvis/results/flow/0000').glob('*/best_chkpt'))
metadata_paths = [path.parent/'_meta.yaml' for path in stock_paths]
protected = list(dict.fromkeys([selected, *stock_paths, *metadata_paths]))
before = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in protected}
payload = read_checkpoint(selected)
net, decoders = load_network_and_decoder(view)
assert {p.device.type for p in net.parameters()} == {'cpu'}
net.eval()
source_digest = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in (
    'tracking/flyvis_checkpoint.py', 'tracking/flyvis_motion.py', 'tracking/flyvis_connectome_swap.py',
    'tools/flyvis-runtime/flyvis/network/network.py', 'tools/flyvis-runtime/flyvis/network/initialization.py',
    'tools/flyvis-runtime/flyvis/network/dynamics.py')}
stock = []
for path, config_path in zip(stock_paths, metadata_paths):
    state = read_checkpoint(path)['network']
    config = yaml.safe_load(config_path.read_text(encoding='utf-8'))['config']['network']
    stock.append(dict(path=path.relative_to(ROOT).as_posix(),
        strength_min=float(state['edges_syn_strength'].min()),
        tau_min=float(state['nodes_time_const'].min()),
        raw_count_min=float(state['edges_syn_count'].min()),
        raw_count_max=float(state['edges_syn_count'].max()),
        semantic_count_finite=bool(torch.isfinite(state['edges_syn_count'].exp()).all()),
        strength_clamp=config['edge_config']['syn_strength'].get('clamp'),
        tau_clamp=config['node_config']['time_const'].get('clamp'),
        count_initial_distribution=config['edge_config']['syn_count']['initial_dist']))

stimulus = torch.full((1, 3, net.n_nodes), .05, device='cpu', dtype=torch.float32)
outputs = {}
cases = []
strength_index = int(torch.nonzero(payload['network']['edges_syn_strength'] > 0)[0].item())
count_index = int(torch.argmax(payload['network']['edges_syn_count']).item())
tau_index = 0
dt = .02


def run(name, source, *, strength_trainable=False):
    original = deepcopy(source)
    net.requires_grad_(False)
    loaded = False
    error = None
    try:
        load_checkpoint_modules(net, decoders, source)
        loaded = True
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
    row = dict(name=name, accepted=loaded, error=error, strength_trainable=strength_trainable)
    if loaded:
        net.edges_syn_strength.requires_grad_(strength_trainable)
        before_strength = net.edges_syn_strength.detach().clone()
        with torch.no_grad():
            activity = net(stimulus, dt=dt)
            params = net._param_api()
            effective = params.edges.weight
        row.update(raw_strength_min_before=float(before_strength.min()),
            raw_strength_min_after=float(net.edges_syn_strength.min()),
            raw_strength_mutated_by_forward=not torch.equal(before_strength, net.edges_syn_strength.detach()),
            semantic_count_finite=bool(torch.isfinite(net.edge_params['syn_count'].semantic_values).all()),
            effective_weight_finite=bool(torch.isfinite(effective).all()),
            effective_weight_negative_count=int((effective < 0).sum()),
            activity_finite=bool(torch.isfinite(activity).all()),
            activity_nonfinite_count=int((~torch.isfinite(activity)).sum()),
            raw_tau_min=float(net.nodes_time_const.min()))
        outputs[name] = activity.detach().cpu().numpy().copy()
    row['input_mapping_unchanged'] = all(torch.equal(value, original['network'][key])
        for key, value in source['network'].items()) and all(
        torch.equal(value, original['decoder'][part][key])
        for part, state in source['decoder'].items() for key, value in state.items())
    cases.append(row)
    return row


run('stock_frozen', deepcopy(payload))
run('stock_frozen_repeat', deepcopy(payload))
negative = deepcopy(payload)
negative['network']['edges_syn_strength'][strength_index] = -abs(negative['network']['edges_syn_strength'][strength_index])
run('negative_strength_frozen', negative)
run('negative_strength_trainable_official_clamp', negative, strength_trainable=True)
projected = deepcopy(negative)
projected['network']['edges_syn_strength'][strength_index] = 0
run('zero_strength_frozen_control', projected)

overflow = deepcopy(payload)
overflow['network']['edges_syn_count'][count_index] = 1000.
run('finite_log_count_semantic_overflow_frozen', overflow)

for value, label in [(-1., 'negative'), (0., 'zero'), (dt, 'dt')]:
    temporal = deepcopy(payload)
    temporal['network']['nodes_time_const'][tau_index] = value
    run(label+'_tau_frozen', temporal)

destination = view.network_class(**view.dir.config.network.to_dict())
destination.eval().requires_grad_(False)
load_checkpoint_modules(net, decoders, deepcopy(payload))
transfer_free_parameters(destination, net)
with torch.no_grad():
    outputs['stock_transfer_control'] = destination(stimulus, dt=dt).cpu().numpy().copy()
load_checkpoint_modules(net, decoders, negative)
transfer_free_parameters(destination, net)
with torch.no_grad():
    outputs['negative_strength_transfer_frozen'] = destination(stimulus, dt=dt).cpu().numpy().copy()

comparisons = dict(
    stock_repeat_exact=np.array_equal(outputs['stock_frozen'], outputs['stock_frozen_repeat']),
    stock_transfer_exact=np.array_equal(outputs['stock_frozen'], outputs['stock_transfer_control']),
    negative_frozen_equals_official_clamped=np.array_equal(outputs['negative_strength_frozen'], outputs['negative_strength_trainable_official_clamp']),
    negative_frozen_vs_clamped_max_abs=float(np.max(np.abs(outputs['negative_strength_frozen']-outputs['negative_strength_trainable_official_clamp']))),
    official_clamped_equals_zero_control=np.array_equal(outputs['negative_strength_trainable_official_clamp'], outputs['zero_strength_frozen_control']),
    negative_transfer_equals_negative_loaded=np.array_equal(outputs['negative_strength_transfer_frozen'], outputs['negative_strength_frozen']),
    negative_tau_equals_zero=np.array_equal(outputs['negative_tau_frozen'], outputs['zero_tau_frozen']),
    negative_tau_equals_dt=np.array_equal(outputs['negative_tau_frozen'], outputs['dt_tau_frozen']))
after = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in protected}
assert before == after
assert comparisons['stock_repeat_exact'] and comparisons['stock_transfer_exact']
assert all(case['input_mapping_unchanged'] for case in cases)
assert not torch.cuda.is_initialized()
np.savez_compressed(OUT/'outputs.npz', **outputs)
report = dict(at=datetime.now(timezone.utc).isoformat(), cuda_initialized=False,
    device=str(flyvis.device), source_sha256=source_digest,
    selected_checkpoint=selected.relative_to(ROOT).as_posix(),
    selected_checkpoint_sha256=before[str(selected.relative_to(ROOT))],
    actual_config=view.dir.config.network.to_dict(),
    stock_count=len(stock), stock=stock, cases=cases, comparisons=comparisons,
    intervention_indices=dict(strength=strength_index, log_count=count_index, tau=tau_index),
    transfer_negative_strength=float(destination.edges_syn_strength[strength_index]),
    protected_files_unchanged=before == after, protected_file_sha256=before,
    output_sha256=hashlib.sha256((OUT/'outputs.npz').read_bytes()).hexdigest(),
    probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    scope='Synthetic finite parameter interventions on isolated CPU instances; three integration steps per case, no training or HOTA. Existing weight/config files are read only. No production changes.')
(OUT/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
print(json.dumps(dict(stock_count=len(stock), cases=cases, comparisons=comparisons, weights_unchanged=True, cuda_initialized=False), indent=2, allow_nan=False))
