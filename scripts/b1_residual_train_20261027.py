"""Одна рабочая ступень Б1: обучение поправки на fit84, оценка на fit20.

Все варианты используют сохранённую базу и голову одинакового устройства.
Оценка действия по сохранённым запросам не называется HOTA полного трекера.
"""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BASE = ROOT / 'runs/b1_real_inputs_20261026'
RUN = ROOT / 'runs/b123_20261027/b1_training'
MSK = timezone(timedelta(hours=3))
ARMS = ('C', 'D', 'R1', 'B')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.next')
    temp.write_bytes((json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))
    temp.replace(path)


def status(stage, **values):
    write(RUN / 'status.json', dict(state='running', stage=stage, updated=datetime.now(MSK).isoformat(), **values))


def prepare():
    assert not RUN.exists(), 'Do not overwrite a frozen training stage'
    auth = read(ROOT / 'runs/b123_20261027/authorization.json')
    assert read(BASE / 'independent_audit.json')['passed']
    old = read(BASE / 'protocol.json'); RUN.mkdir()
    files = ['scripts/b1_residual_train_20261027.py', 'tracking/b1_learning.py',
             'tracking/b1_real_inputs.py', 'tracking/flyvis_temporal_residual.py',
             'tracking/z13_readout.py', 'scripts/z13_target_model.py',
             'scripts/z13_identity_gate_readout_20261026.py', 'scripts/z13_identity_gate_tagged_20261026.py',
             'runs/b123_20261027/authorization.json']
    files += [f'runs/b1_real_inputs_20261026/{name}' for name in
              ('protocol.json', 'report.json', 'base_inputs.npz', 'base_predictions.npz', 'supervision.npz', 'gate.joblib')]
    for video in old['fit84'] + old['fit20']:
        files += [f'runs/b1_real_inputs_20261026/inputs/{video}.json', f'runs/z13_data/{video}.npz']
    now = datetime.now(MSK)
    p = dict(id='b1_temporal_residual_training_v1', created=now.isoformat(), timezone='Europe/Moscow',
             deadline=min(now + timedelta(hours=2), datetime.fromisoformat(auth['save_at'])).isoformat(),
             author_hard_stop=auth['hard_stop'], save_at=auth['save_at'], device='cuda:0', seed=26,
             fit84=old['fit84'], fit20=old['fit20'], evaluate26=[], arms=list(ARMS), epochs=1,
             train='One full pass over all fit84 queries with observable history and known real-region identity labels; per-video/per-query deterministic shuffle shared by arms. No sampling by helpful outcome.',
             loss='Per-query BCE of updated identity probabilities over known real regions, plus 0.01 mean squared residual; NULL/unknown identity excluded; no neutral-action loss. D adds core attraction penalty weight 1.',
             head='TemporalResidualAssociation(channels=34, hidden=32), zero output init, same seed and architecture all arms; Adam lr0.001, gradient norm clipping1.',
             core='C stock frozen; D identical stock initialization, FREE and CORE_LR from z13_readout, clamp after update; R1 identical history 17EMA+17transient channels; B existing spatial HexGRU16, lr0.001. Parameter counts/timings reported, B is not parameter-matched.',
             fixed_base='Saved out-of-fold fit84/full-fit fit20 identity probabilities; same saved action gate and threshold0.5; no refit or calibration on fit20 in this stage.',
             checkpoint='Atomic checkpoint each100updates and end of video; model+optimizer+next query cursor+history; reload exact verified on first optimizer steps.',
             implementation_acceptance=['All four paths complete forward/backward/update/save/reload with finite loss', 'Zero output preserves identity probabilities exactly', 'D receives nonzero core gradients after first head step and core changes after update; C core unchanged'],
             screening='Full fit20 saved-action utility and identity BCE/AUC with paired-video intervals. Record harmful/beneficial changes and coverage. No tuning after results. No final scientific acceptance on these metrics.',
             continuation='After all arms complete, assess working full-tracker integration and HOTA; if it is not ready, preserve this result and exact remaining implementation. B2/B3 preparation proceeds in the authorized window. No success based on offline utility.',
             limitations='Reused development104, transductive unlabeled PCA recipe from base. One seed/epoch is pilot only. Not HOTA, no train26 or independent final test.',
             sha256={name: sha(ROOT / name) for name in sorted(set(files))})
    write(RUN / 'protocol.json', p)
    write(RUN / 'status.json', dict(state='ready', stage='registered before training', updated=now.isoformat()))


class Dataset:
    def __init__(self, p):
        self.p = p
        with np.load(BASE / 'base_inputs.npz') as z:
            self.data = {k: z[k] for k in ('uid', 'name', 'geo', 'mask')}
        with np.load(BASE / 'supervision.npz') as z:
            assert np.array_equal(z['uid'], self.data['uid'])
            self.data.update({k: z[k] for k in ('ident', 'good', 'harm', 'kind')})
        with np.load(BASE / 'base_predictions.npz') as z:
            assert np.array_equal(z['uid'], self.data['uid'])
            self.probability = z['identity_probability']
        self.index = {int(u): i for i, u in enumerate(self.data['uid'])}
        self.rows = {v: read(BASE / 'inputs' / (v + '.json'))['rows'] for v in p['fit84'] + p['fit20']}
        self.last_video = None; self.hexes = None

    def tensor(self, torch, row, device):
        if self.last_video != row['video']:
            with np.load(ROOT / row['retinal_source']) as z:
                self.hexes = z['fr_hex'].astype(np.float32)
            self.last_video = row['video']
        value = self.hexes[row['prefix_start']:row['current_array_index'] + 1][None]
        return torch.as_tensor(value, device=device)

    def train_rows(self):
        rng = np.random.default_rng(self.p['seed'])
        videos = rng.permutation(self.p['fit84']).tolist(); out = []
        for video in videos:
            rows = self.rows[video]
            for n in rng.permutation(len(rows)):
                row = rows[n]; i = self.index[row['uid']]
                known = self.data['mask'][i] & (self.data['ident'][i] >= 0)
                known[0] = False
                if row['has_history'] and known.any(): out.append(row)
        return out


def build(torch, arm):
    from tracking.b1_learning import ReceptorHistory
    from tracking.flyvis_temporal_residual import TemporalResidualAssociation
    from tracking.z13_readout import FlyBackbone, make_modules
    from scripts.z13_target_model import build_arm
    # Seed head identically after any backend creation (including GRU randomness).
    if arm in ('C', 'D', 'B'):
        _, geom, stock = build_arm('C', torch, None)
        if arm == 'B':
            _, cls = make_modules(torch)
            torch.manual_seed(26); torch.cuda.manual_seed_all(26)
            bb = cls(geom.neighbors).cuda(); bb.dev = torch.device('cuda')
        else: bb = FlyBackbone(stock.est, trainable=arm == 'D')
    else: bb = ReceptorHistory('cuda')
    torch.manual_seed(26); torch.cuda.manual_seed_all(26)
    head = TemporalResidualAssociation().cuda()
    groups = [dict(params=list(head.parameters()), lr=.001)]
    if arm == 'D': groups += bb.param_groups()
    if arm == 'B': groups += [dict(params=list(bb.parameters()), lr=.001)]
    return bb, head, torch.optim.Adam(groups)


def backbone_state(bb, arm):
    if arm in ('C', 'D'): return bb.net.state_dict()
    if arm == 'B': return bb.state_dict()
    return {}


def checkpoint(torch, path, bb, head, opt, arm, cursor, history, protocol_sha):
    state = dict(arm=arm, cursor=cursor, head=head.state_dict(), backbone=backbone_state(bb, arm),
                 optimizer=opt.state_dict(), history=history, protocol_sha256=protocol_sha)
    temp = path.with_suffix('.next'); torch.save(state, temp); temp.replace(path)


def load_checkpoint(torch, path, bb, head, opt, arm, protocol_sha):
    state = torch.load(path, map_location='cuda', weights_only=False)
    assert state['arm'] == arm and state['protocol_sha256'] == protocol_sha
    head.load_state_dict(state['head']); opt.load_state_dict(state['optimizer'])
    if arm in ('C', 'D'): bb.net.load_state_dict(state['backbone'])
    if arm == 'B': bb.load_state_dict(state['backbone'])
    return state['cursor'], state['history']


def one(torch, bb, head, row, dataset, arm):
    from tracking.b1_learning import tensors_for_row, forward_residual
    i = dataset.index[row['uid']]; hx = dataset.tensor(torch, row, bb.dev)
    inputs = tensors_for_row(row, hx, dataset.probability[i], bb.dev)
    updated, delta = forward_residual(bb, head, hx, row, inputs, arm in ('D', 'B'))
    return updated, delta, inputs, i


def run_arm(torch, arm, dataset, p, protocol_sha):
    from tracking.b1_learning import identity_objective
    target = RUN / arm; target.mkdir(exist_ok=True)
    if (target / 'report.json').exists():
        r = read(target / 'report.json')
        assert r['complete'] and r['protocol_sha256'] == protocol_sha
        assert sha(target / 'probability.npz') == r['probability_sha256']
        return r
    bb, head, opt = build(torch, arm)
    rows = dataset.train_rows(); original = [x.detach().clone() for x in getattr(bb, 'params', [])]
    original_stock = {n: x.detach().clone() for n, x in bb.net.named_parameters()} if arm == 'C' else {}
    cursor, history = 0, []
    cp = target / 'checkpoint.pt'
    if cp.exists(): cursor, history = load_checkpoint(torch, cp, bb, head, opt, arm, protocol_sha)
    implementation = read(target / 'implementation.json') if (target / 'implementation.json').exists() else {}
    began = time.monotonic()
    core_nonzero = False
    for step in range(cursor, len(rows)):
        if (RUN / 'stop.request').exists() or datetime.now(MSK) >= datetime.fromisoformat(p['deadline']) - timedelta(seconds=30):
            checkpoint(torch, cp, bb, head, opt, arm, step, history, protocol_sha)
            write(target / 'interrupted.json', dict(cursor=step, total=len(rows), updated=datetime.now(MSK).isoformat()))
            raise TimeoutError('Saved before stage deadline')
        row = rows[step]; opt.zero_grad(set_to_none=True)
        updated, delta, inputs, i = one(torch, bb, head, row, dataset, arm)
        if step == 0:
            assert torch.equal(updated[inputs['real']], inputs['base'][inputs['real']])
            implementation['zero_start_exact'] = True
        labels = torch.as_tensor(dataset.data['ident'][i, :updated.shape[1]][None], device=bb.dev)
        loss = identity_objective(updated, delta, labels, inputs['real'])
        assert loss is not None
        if arm == 'D': loss = loss + bb.penalty()
        assert torch.isfinite(loss)
        loss.backward()
        if arm == 'D':
            norms = [float(x.grad.norm()) if x.grad is not None else 0 for x in bb.params]
            core_nonzero |= all(np.isfinite(norms)) and all(x > 0 for x in norms)
        parameters = [x for group in opt.param_groups for x in group['params']]
        torch.nn.utils.clip_grad_norm_(parameters, 1.)
        opt.step()
        if arm == 'D': bb.clamp()
        history.append(float(loss.detach()))
        if step == 2:
            # Verify actual optimizer checkpoint restoration before the long tail.
            checkpoint(torch, cp, bb, head, opt, arm, step + 1, history, protocol_sha)
            with torch.no_grad(): before = one(torch, bb, head, row, dataset, arm)[0]
            saved = {n: x.detach().clone() for n, x in head.named_parameters()}
            saved_backbone = {n: x.detach().clone() for n, x in backbone_state(bb, arm).items()}
            with torch.no_grad():
                next(head.parameters()).add_(.123)
            restored, _ = load_checkpoint(torch, cp, bb, head, opt, arm, protocol_sha)
            with torch.no_grad(): after = one(torch, bb, head, row, dataset, arm)[0]
            assert restored == step + 1
            assert all(torch.equal(saved[n], x) for n, x in head.named_parameters())
            assert all(torch.equal(saved_backbone[n], x) for n, x in backbone_state(bb, arm).items())
            implementation['checkpoint_restore_exact'] = True
            implementation['repeated_forward_max_difference'] = float((before[inputs['real']] - after[inputs['real']]).abs().max())
            implementation['first_three_losses_finite'] = all(np.isfinite(history[:3]))
            if arm == 'D':
                assert core_nonzero, 'Core gradient did not open after output updates'
                implementation['core_gradient_nonzero'] = True
            write(target / 'implementation.json', implementation)
        if (step + 1) % 100 == 0 or step + 1 == len(rows) or rows[step + 1]['video'] != row['video']:
            checkpoint(torch, cp, bb, head, opt, arm, step + 1, history, protocol_sha)
            status('training residual', arm=arm, done=step + 1, total=len(rows), epochs=1,
                   loss=float(loss.detach()), seconds=time.monotonic() - began,
                   seconds_left=(time.monotonic() - began) / max(1, step + 1 - cursor) * (len(rows) - step - 1))
    probability = dataset.probability.copy(); evaluated = 0
    with torch.no_grad():
        for video in p['fit20']:
            if datetime.now(MSK) >= datetime.fromisoformat(p['deadline']) - timedelta(seconds=20):
                raise TimeoutError('Evaluation interrupted; training checkpoint retained')
            for row in dataset.rows[video]:
                if row['has_history'] and len(row['choices']) > 1:
                    updated, _, _, i = one(torch, bb, head, row, dataset, arm)
                    probability[i, :updated.shape[1]] = updated[0].cpu().numpy()
                evaluated += 1
            status('evaluating frozen residual', arm=arm, video=video, evaluated_queries=evaluated)
    if arm == 'C': assert all(torch.equal(original_stock[n], x) for n, x in bb.net.named_parameters())
    if arm == 'D': assert any(not torch.equal(q, x.detach()) for q, x in zip(original, bb.params))
    np.savez_compressed(target / 'probability.npz', uid=dataset.data['uid'], probability=probability,
                        evaluated=np.isin(dataset.data['name'], p['fit20']))
    result = dict(complete=True, arm=arm, protocol_sha256=protocol_sha, train_queries=len(rows), optimizer_steps=len(rows),
                  evaluated_queries=evaluated, seconds=time.monotonic() - began, implementation=implementation,
                  head_parameters=sum(x.numel() for x in head.parameters()),
                  backend_trainable_parameters=sum(x.numel() for group in opt.param_groups[1:] for x in group['params']),
                  initial_loss_mean=float(np.mean(history[:min(100, len(history))])),
                  final_loss_mean=float(np.mean(history[-100:])), checkpoint_sha256=sha(cp),
                  probability_sha256=sha(target / 'probability.npz'), core_updated=arm == 'D', hota_measured=False)
    write(target / 'report.json', result)
    del bb, head, opt
    torch.cuda.empty_cache()
    return result


def evaluate_actions(dataset, p, results):
    import joblib
    from sklearn.metrics import roc_auc_score
    from scripts.z13_identity_gate_readout_20261026 import gate_features
    from scripts.z13_identity_gate_tagged_20261026 import policy_net
    from scripts.z13_continue import paired
    gate = joblib.load(BASE / 'gate.joblib'); d = dataset.data
    known = d['kind'] >= 0
    alt = d['mask'] & (d['geo'][..., -1] < .5)
    inside = np.isin(d['name'], p['fit20']); scored = inside[:, None] & alt
    per, details = {}, {}
    for arm in ('base', *ARMS):
        probability = dataset.probability if arm == 'base' else np.load(RUN / arm / 'probability.npz')['probability']
        features = gate_features(d, probability)
        scores = np.full(d['mask'].shape, np.nan); scores[scored] = gate.predict_proba(features[scored])[:, 1]
        per[arm], counts = policy_net(d, scores, p['fit20'], .5)
        real = d['mask'].copy(); real[:, 0] = False
        labeled = inside[:, None] & real & (d['ident'] >= 0)
        y, pred = d['ident'][labeled], probability[labeled]
        details[arm] = dict(counts=counts, identity_auc=float(roc_auc_score(y, pred)),
                            identity_regions=int(labeled.sum()), mean_net=float(np.mean(list(per[arm].values()))))
    contrasts = {f'D-{arm}': paired(per['D'], per[arm]) for arm in ('base', 'C', 'R1', 'B')}
    write(RUN / 'report.json', dict(complete=True, stage='one-epoch residual pilot, saved queries only',
          arms=results, per_video=per, summary=details, paired=contrasts, threshold=.5,
          scientific_hypothesis_accepted=False, hota_measured=False, train26_evaluations=0,
          full_tracker_integration_ready=False, limitation=p['limitations']))


def run():
    p = read(RUN / 'protocol.json')
    for name, expected in p['sha256'].items(): assert sha(ROOT / name) == expected, name
    import torch
    torch.set_num_threads(2)
    assert torch.cuda.is_available()
    from tracking.gpu_lock import GpuLock
    dataset = Dataset(p); results = {}
    with GpuLock('flyvis_fovea_gpu'):
        for arm in ARMS:
            status('building arm', arm=arm)
            results[arm] = run_arm(torch, arm, dataset, p, sha(RUN / 'protocol.json'))
    # Explicit device boundary; sklearn gets numpy arrays, no Torch tensor creation.
    evaluate_actions(dataset, p, results)
    write(RUN / 'status.json', dict(state='completed', stage='saved-query pilot completed; HOTA pending', updated=datetime.now(MSK).isoformat()))


if __name__ == '__main__':
    try:
        prepare() if '--prepare' in sys.argv else run()
    except Exception as error:
        if RUN.exists(): write(RUN / ('failure_' + datetime.now(MSK).strftime('%H%M%S') + '.json'), dict(error=repr(error), updated=datetime.now(MSK).isoformat()))
        raise
