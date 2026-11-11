"""Tap the missing first state of the original unchanged causal pass."""
from common import *
start(cpu=False)
import numpy as np
import torch
from tracking.gpu_lock import GpuLock
from tracking.flyvis_adapter import activate_flyvis
from scripts.z13_target_model import build_arm, receptor_features_rich

verify()
phase = read(OLD / 'execution_protocol.json'); edges = read(OLD / 'protocol_v2.json')['edges']
norm = {n: read(ROOT/f'runs/z13_target_center_dev_20261018/{n}/normalization.json') for n in ['C', 'R1']}
cm, cs = [np.array(norm['C'][n])[:34] for n in ['mean', 'sd']]
rm, rs = [np.array(norm['R1'][n])[:34] for n in ['mean', 'sd']]
torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
assert torch.cuda.is_available()
receipts = []; bb = None; captures = {}; capture_step = False
try:
    with GpuLock('flyvis_fovea_gpu'), torch.no_grad():
        for i, e in enumerate(edges):
            sequence = e['sequence']; targets = {3: key(e['video'], sequence[:4]), len(sequence)-1: key(e['video'], sequence)}
            pending = {k for k in targets.values() if not (OUT/'maps'/(k+'.json')).exists()}
            if not pending:
                for k in targets.values():
                    meta = read(OUT/'maps'/(k+'.json')); assert sha(OUT/'maps'/(k+'.npz')) == meta['sha256']; receipts.append(meta)
                continue
            k = key(e['video'], sequence)
            hx = np.load(ROOT/phase['inputs'][k]['path'], allow_pickle=False)
            refk = targets[3]
            np.testing.assert_array_equal(hx[:4], np.load(ROOT/phase['inputs'][refk]['path'], allow_pickle=False))
            rich = receptor_features_rich(hx[None].astype(float))
            if bb is None:
                activate_flyvis(device='cuda'); _, _, bb = build_arm('C', torch, None)
                assert bb.dev.type == 'cuda' and not bb.net.training and bb.steps == 2 and abs(bb.dt-1/60) < 1e-12
                weights = {n: p.detach().clone() for n, p in bb.net.named_parameters()}
                forward = bb.net.forward

                def tapped(*args, **kwargs):
                    states = forward(*args, **kwargs)
                    assert kwargs.get('as_states') is True and len(states) == 2
                    if capture_step:
                        before = states[-1].nodes.activity.detach().clone()
                        captures['first'] = states[0].nodes.activity[:, bb.index].detach().clone()
                        assert torch.equal(before, states[-1].nodes.activity)
                    return states

                bb.net.forward = tapped
            x = torch.from_numpy(hx[None]).to(bb.dev); state = bb.init(x[:, :1])
            for t in range(1, len(sequence)):
                capture_step = t in targets and targets[t] in pending
                state, feature = bb.step(state, x[:, t-1], x[:, t])
                if not capture_step: continue
                k = targets[t]; old = arrays(k)
                last = ((feature[0].cpu().numpy().T.astype(float)-cm)/cs).astype(np.float32)
                error = float(abs(last-old['C34']).max())
                assert error <= 1e-4, (k, error)
                ordinary = ((rich[t][0].T-rm)/rs).astype(np.float32)
                np.testing.assert_array_equal(ordinary, old['R1'])
                mid = ((captures.pop('first')[0].cpu().numpy().T.astype(float)-cm)/cs).astype(np.float32)
                rmid = (((rich[t-1][0].T+rich[t][0].T)/2-rm)/rs).astype(np.float32)
                assert mid.shape == rmid.shape == old['C34'].shape == (721, 34)
                assert np.isfinite(mid).all() and np.isfinite(rmid).all()
                path = OUT/'maps'/(k+'.npz'); path.parent.mkdir(exist_ok=True)
                with path.open('xb') as f: np.savez_compressed(f, C1=mid, R1_mid=rmid)
                meta = dict(key=k, sha256=sha(path), video=e['video'], sequence=sequence[:t+1],
                    final_C34_max_error=error, final_R1_exact=True, first_index=0, last_index=1,
                    dt=bb.dt, steps=bb.steps, tap_did_not_mutate_terminal_state=True,
                    first_vs_last_relative_l2=float(np.linalg.norm(mid-last)/(np.linalg.norm(last)+1e-30)))
                write('maps/'+k+'.json', meta); receipts.append(meta)
            del x, state, feature
            status('recovering', completed=i+1, total=len(edges), snapshots=len(receipts))
        assert len(receipts) == 52 and len({r['key'] for r in receipts}) == 52
        if bb is not None: assert all(torch.equal(p, weights[n]) for n, p in bb.net.named_parameters())
finally:
    if bb is not None and 'forward' in globals(): bb.net.forward = forward
verify()
write('maps_complete.json', dict(passed=True, snapshots=len(receipts), weights_unchanged=True,
    final_parity_max_error=max(r['final_C34_max_error'] for r in receipts), final_R1_exact=True,
    records=receipts, no_old_scores_recomputed=True, model_clock_not_video_clock=True))
status('maps_complete', completed=52, total=52)
print('MIDPOINT_MAPS_READY',len(receipts))
