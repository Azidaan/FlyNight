"""FlyVis core adaptation for target identity among same-class neighbours (single GPU).

The actual D readout path is kept: exemplar frame -> normalised 68 features -> observed ridge
target filter (closed form, differentiable) -> target response on the query frame -> mean response
inside each real detection region. The 734 free core parameters (plus one logit scale) learn that
our target's region responds more than the other real regions of the same class (cross-entropy over
regions; offline identity labels only as training targets). Training: fit84 only. Stop gate on fit20
(never seen by the core): top-1 target choice of new D versus initial D and stock C with the same
readout. Features for the two-stage readout are regenerated only if the gate passes (separate step).
"""
from pathlib import Path
from datetime import datetime
import os
import sys
import json
import time
import copy
import hashlib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.z13_candidate_reserve_20261021 import read, write, sha, now
from scripts.z13_continue import paired

IDENT = 'z13_core_identity_v1'
RUN = ROOT / 'runs/z13_core_identity_20261026'
BASE = ROOT / 'runs/z13_region_assignment_20261025'
SUPERVISION = ROOT / 'runs/z13_temporal_pair_readiness_20261025/supervision'
CHECKPOINT_D = ROOT / 'runs/z13_spatial_v2_20261013/D/full/model.pt'
DEADLINE = datetime.fromisoformat('2026-10-27T07:30:00+03:00')
STEPS = 300
EPISODES_PER_STEP = 8
GATE_MIN = 0.03


def status(stage, **kw):
    if (RUN / 'stop.request').exists() or now() >= DEADLINE: raise TimeoutError('Stop request or author weekday deadline 07:30 MSK')
    write(RUN / 'status.json', dict(state='running', stage=stage, updated=now().isoformat(), **kw))


def prepare():
    assert not (RUN / 'protocol.json').exists(); RUN.mkdir(parents=True, exist_ok=True)
    p = read(BASE / 'recovery2/protocol.json')
    sources = [Path(__file__), ROOT / 'tracking/z13_readout.py', ROOT / 'tracking/z13_region_assignment.py', ROOT / 'tracking/z13_target_center.py',
               ROOT / 'scripts/z13_target_model.py', ROOT / 'scripts/z13_train.py', CHECKPOINT_D,
               ROOT / 'runs/z13_target_center_dev_20261018/D/normalization.json', ROOT / 'runs/z13_target_center_dev_20261018/C/normalization.json',
               ROOT / 'runs/z13_target_center_20261018/D/choice.json', ROOT / 'runs/z13_target_center_20261018/C/choice.json']
    write(RUN / 'protocol.json', dict(
        id=IDENT, created=now().isoformat(), timezone='Europe/Moscow', device='cuda', deadline=DEADLINE.isoformat(),
        authorization='Автор 26.10.2026 ~18:30 MSK: «Давай пробовать» (дообучение представления FlyVis после ночного итога); окно до 27.10 07:40 MSK.',
        fit84=p['fit84'], fit20=p['fit20'], evaluate26=p['evaluate26'],
        training=dict(data='fit84 only', eligible='queries with known target identity present among >=2 real regions whose identities are all known',
                      loss='cross-entropy over real regions of tau * mean target response inside region; tau learnable (init 10, lr 1e-2)',
                      core='734 free parameters (nodes_bias, nodes_time_const, edges_syn_strength), CORE_LR from tracking/z13_readout.py, attraction penalty weight 1, clamp after step',
                      steps=STEPS, episodes_per_step=EPISODES_PER_STEP, seed=0, start='D checkpoint runs/z13_spatial_v2_20261013/D/full/model.pt', normalization='fixed D mean/sd; lambda 0.1'),
        technical='Before training: descriptor and pooled region features of initial D reproduce cached spatial-set features (pooled max |diff| < 1e-3, response ranking identical).',
        gate=dict(data='fit20 eligible queries (never used for core training)', metric='per-video top-1 accuracy of choosing the target region by mean target response',
                  criterion=f'new D - initial D mean >= +{GATE_MIN} and lower 95% bound > 0, AND new D - stock C lower 95% bound > 0', pass_next='separate registered step: regenerate features for 130 videos and run the two-stage readout final evaluation (fit84 train, fit20 calibration, train26 once) with D, new D, C, R1, no-fly, true identity',
                  fail='record negative, keep checkpoint, no feature regeneration'),
        cannot_claim='Top-1 identity on fit20 is not assignment benefit nor HOTA. Earlier core adaptations (z13_aligned_core_v1, z13_core_continue_v1) did not improve localisation.',
        sha256={q.relative_to(ROOT).as_posix(): sha(q) for q in sources}))
    write(RUN / 'status.json', dict(state='ready', stage='protocol', updated=now().isoformat()))


def build_queries(split, videos):
    """Eligible queries grouped by split-local episode: crop, prediction, region masks and target region index."""
    from scripts.z13_region_assignment_recovery2_20261025 import namespace
    from tracking.z13_region_assignment import regions
    ns = namespace(); queries, _ = ns['queries_and_crops'](); geom = ns['geometry']()
    labelrows = {}
    for v in videos:
        labelrows.update({r['uid']: r for r in read(BASE / 'labels/videos' / (v + '.json'))['rows']})
    byep = {}; counts = dict(queries=0, eligible=0)
    for v in videos:
        sup = {r['uid']: r for r in read(SUPERVISION / (v + '.json'))['rows']}
        with np.load(ROOT / f'runs/train_cache/{v}/detections/raw_video.npz') as z:
            det = {k: z[k] for k in ('xyxy', 'conf', 'cls', 'frame_idx')}
        for q in queries[v]:
            counts['queries'] += 1; s = sup.get(q['uid']); l = q['label']
            if s is None or not s['target_identity'] or len(s['current_identity']) < 2 or not all(s['current_identity']): continue
            if s['target_identity'] not in s['current_identity']: continue
            ix = (det['frame_idx'] == q['frame']) & (det['conf'] >= .25); row = labelrows[q['uid']]
            choices, masks, _ = regions(geom, l['crop'], l['pred'], det['xyxy'][ix], det['conf'][ix], det['cls'][ix], row['regions']['track_class'])
            assert choices == row['regions']['choices'][1:] and len(masks) - 1 == len(s['current_identity'])
            byep.setdefault(l['episode'], []).append(dict(uid=q['uid'], video=v, pos=int(l['pos']), masks=masks[1:], target=s['current_identity'].index(s['target_identity'])))
            counts['eligible'] += 1
    return byep, counts, geom


def features(torch, bb, hx, nb, need):
    """Normalisation-free (B,68,721) features at needed frames, with gradient when enabled."""
    state = bb.init(hx[:, :1]); last = hx[:, 0]; out = {}
    for i in range(1, hx.shape[1]):
        state, f = bb.step(state, last, hx[:, i]); last = hx[:, i]
        if i in need: out[i] = torch.cat([f, f[:, :, nb].mean(-1)], 1).double()
    return out


def query_logits(torch, x_ex, ex_mask, x_q, masks, geom, lam, mu, sd):
    from tracking.z13_target_center import observed_center_target
    xe = (x_ex.T - mu) / sd; xq = (x_q.T - mu) / sd
    y = torch.as_tensor(observed_center_target(ex_mask, geom.points, geom.spacing), dtype=xe.dtype, device=xe.device)
    m = xe.mean(0); xc = xe - m; ym = y.mean()
    eye = torch.eye(xe.shape[1], dtype=xe.dtype, device=xe.device)
    w = torch.linalg.solve(xc.T @ xc / len(xe) + lam * eye, xc.T @ (y - ym) / len(xe))
    score = (xq - m) @ w + ym
    M = torch.as_tensor(np.asarray(masks, np.float64), device=xe.device)
    return (M @ score) / M.sum(1), w


def evaluate(torch, bb, byep, data, geom, lam, mu, sd, label):
    nb = torch.as_tensor(geom.neighbors[:, 1:], device=bb.dev); per = {}; hits = []
    with torch.no_grad():
        for n, (e, qs) in enumerate(sorted(byep.items())):
            if n % 25 == 0: status('evaluate', arm=label, done=n, total=len(byep))
            hx, ex_mask = data.batch(torch, bb.dev, [e], [max(q['pos'] for q in qs)])
            F = features(torch, bb, hx, nb, {3} | {4 + q['pos'] for q in qs})
            for q in qs:
                s, _ = query_logits(torch, F[3][0], ex_mask[0].cpu().numpy(), F[4 + q['pos']][0], q['masks'], geom, lam, mu, sd)
                ok = int(int(s.argmax()) == q['target']); hits.append((q['video'], ok))
    for v in sorted({h[0] for h in hits}): per[v] = float(np.mean([h[1] for h in hits if h[0] == v]))
    return per


def run():
    import torch
    from scripts import z13_target_model as old
    from scripts.z13_train import Episodes
    from tracking.z13_readout import FlyBackbone
    from tracking.gpu_lock import GpuLock
    p = read(RUN / 'protocol.json')
    for k, h in p['sha256'].items(): assert sha(ROOT / k) == h, k
    assert torch.cuda.is_available(), 'GPU required; no CPU fallback'
    torch.manual_seed(0); torch.cuda.manual_seed_all(0); rng = np.random.default_rng(0); began = time.monotonic()
    status('queries')
    train_q, train_counts, geom = build_queries('fit84', p['fit84']); gate_q, gate_counts, _ = build_queries('fit20', p['fit20'])
    write(RUN / 'queries.json', dict(fit84=train_counts, fit20=gate_counts, fit84_episodes=len(train_q), fit20_episodes=len(gate_q)))
    norm = {a: read(ROOT / f'runs/z13_target_center_dev_20261018/{a}/normalization.json') for a in ('D', 'C')}
    lam = {a: read(ROOT / f'runs/z13_target_center_20261018/{a}/choice.json')['center']['lambda_value'] for a in ('D', 'C')}
    with GpuLock('flyvis_fovea_gpu'):
        ck = torch.load(CHECKPOINT_D, map_location='cpu', weights_only=False)
        bank, _, frozen = old.build_arm('D', torch, ck); bb = FlyBackbone(frozen.est, trainable=True); dev = bb.dev
        mu = {a: torch.as_tensor(np.asarray(norm[a]['mean']), device=dev) for a in norm}; sd = {a: torch.as_tensor(np.asarray(norm[a]['sd']), device=dev) for a in norm}
        data84, data20 = Episodes(p['fit84']), Episodes(p['fit20'])
        result = {}
        if (RUN / 'gate_initial.json').exists(): result = read(RUN / 'gate_initial.json')
        else:
            result['D_initial'] = evaluate(torch, bb, gate_q, data20, geom, lam['D'], mu['D'], sd['D'], 'D_initial')
            _, _, cbb = old.build_arm('C', torch, ck)
            result['C'] = evaluate(torch, cbb, gate_q, data20, geom, lam['C'], mu['C'], sd['C'], 'C'); del cbb; torch.cuda.empty_cache()
            write(RUN / 'gate_initial.json', result)
        tau = torch.nn.Parameter(torch.tensor(10., dtype=torch.float64, device=dev))
        opt = torch.optim.Adam(bb.param_groups() + [{'params': [tau], 'lr': 1e-2}])
        nb = torch.as_tensor(geom.neighbors[:, 1:], device=dev); episodes = sorted(train_q); history = []; t0 = time.monotonic(); step = 0
        cp = RUN / 'checkpoint.pt'
        if cp.exists():
            c = torch.load(cp, map_location='cpu', weights_only=False)
            bb.net.load_state_dict({k: v.to(dev) for k, v in c['net'].items()}); opt.load_state_dict(c['opt']); tau.data.fill_(c['tau'])
            rng.bit_generator.state = c['rng']; step = c['step']; history = c['history']
        for step in range(step, STEPS):
            picked = rng.choice(len(episodes), size=EPISODES_PER_STEP, replace=False); opt.zero_grad(set_to_none=True)
            nq = sum(len(train_q[episodes[i]]) for i in picked); objective = 0.; correct = 0
            for i in picked:
                e = episodes[i]; qs = train_q[e]
                hx, ex_mask = data84.batch(torch, dev, [e], [max(q['pos'] for q in qs)])
                F = features(torch, bb, hx, nb, {3} | {4 + q['pos'] for q in qs}); loss = 0.
                for q in qs:
                    s, _ = query_logits(torch, F[3][0], ex_mask[0].cpu().numpy(), F[4 + q['pos']][0], q['masks'], geom, lam['D'], mu['D'], sd['D'])
                    loss = loss + torch.nn.functional.cross_entropy((tau * s)[None], torch.tensor([q['target']], device=dev))
                    correct += int(int(s.argmax()) == q['target'])
                (loss / nq).backward(); objective += float(loss.detach()) / nq
            penalty = bb.penalty(); penalty.backward()
            if any(q.grad is not None and not bool(torch.isfinite(q.grad).all()) for q in bb.params): raise RuntimeError('Nonfinite core gradient')
            opt.step(); bb.clamp()
            history.append(dict(step=step + 1, loss=objective, train_top1=correct / nq, penalty=float(penalty.detach()), tau=float(tau.detach()),
                                batch_sha256=hashlib.sha256(np.asarray(picked, np.int64).tobytes()).hexdigest()))
            rate = (time.monotonic() - t0) / (len(history) - (step + 1 - len(history)) if False else 1)
            if (step + 1) % 10 == 0 or step == 0:
                torch.save(dict(net={k: v.detach().cpu() for k, v in bb.net.state_dict().items()}, opt=opt.state_dict(), tau=float(tau.detach()),
                                rng=rng.bit_generator.state, step=step + 1, history=history), cp)
                write(RUN / 'history.json', history)
            el = time.monotonic() - t0
            status('training', done=step + 1, total=STEPS, loss=objective, train_top1=correct / nq, seconds=el, seconds_left=el / (len(history)) * (STEPS - step - 1) if history else None)
        torch.save(dict(net={k: v.detach().cpu() for k, v in bb.net.state_dict().items()}, tau=float(tau.detach()), step=STEPS, history=history), RUN / 'model.pt')
        result['D_identity'] = evaluate(torch, bb, gate_q, data20, geom, lam['D'], mu['D'], sd['D'], 'D_identity')
    comp = {'Did-Dinitial': paired(result['D_identity'], result['D_initial']), 'Did-C': paired(result['D_identity'], result['C']), 'Dinitial-C': paired(result['D_initial'], result['C'])}
    passed = comp['Did-Dinitial']['mean_per_video'] >= GATE_MIN and comp['Did-Dinitial']['ci95'][0] > 0 and comp['Did-C']['ci95'][0] > 0
    write(RUN / 'report.json', dict(complete=True, id=IDENT, gate_pass=bool(passed), paired=comp, per_video=result,
                                   means={k: float(np.mean(list(v.values()))) for k, v in result.items()}, steps=STEPS, seconds=time.monotonic() - began, hota_measured=False))
    write(RUN / 'status.json', dict(state='completed', stage='done', gate_pass=bool(passed), updated=now().isoformat()))


if __name__ == '__main__':
    if sys.argv[1] == '--prepare': prepare()
    elif sys.argv[1] == '--run':
        try: run()
        except Exception as error:
            write(RUN / 'failure.json', dict(at=now().isoformat(), error=repr(error))); raise
