"""Paired D adaptation through the actual ridge target readout; single GPU."""
import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.z13_parallel_next import read, write, sha
RUN = ROOT / 'runs/z13_aligned_core_20261018'
PROTOCOL = None
DEADLINE = None
START = time.monotonic()


def verify():
    for rel, expected in PROTOCOL['sha256'].items():
        if sha(ROOT / rel) != expected:
            raise RuntimeError('Frozen source changed: ' + rel)


def check():
    from tracking.z13_checkpoint import TrainingPaused
    if (RUN / 'stop.request').exists() or time.time() >= DEADLINE:
        raise TrainingPaused('Day budget ended; completed work preserved')


def status(**values):
    state = values.pop('state', 'running')
    write(RUN / 'status.json', dict(state=state, updated=datetime.now(timezone.utc).isoformat(),
          elapsed_seconds=time.monotonic()-START, **values))


def features_with_grad(torch, bb, hx, neighbors):
    from tracking.z13_readout import N_EXEMPLAR
    nb = torch.as_tensor(neighbors[:, 1:], device=bb.dev)
    state = bb.init(hx[:, :1]); last = hx[:, 0]; exemplar = None
    for i in range(1, hx.shape[1]):
        check()
        state, f = bb.step(state, last, hx[:, i]); last = hx[:, i]
        f = torch.cat([f, f[:, :, nb].mean(-1)], 1).double()
        if i == N_EXEMPLAR - 1:
            exemplar = f[0].T
    return exemplar, f[0].T[None]


def evaluate(torch, bb, bank, geom, stats, folder, split):
    from scripts import z13_train as training
    from scripts.z13_target_model import episode_features, center_and_presence
    from tracking.z13_readout import N_EXEMPLAR, to_frame
    from tracking.z13_target_center import observed_center_target, ridge_scores
    from scripts.z13_continue import per_video_useful, coverage_table
    old_data = training.DATA
    data_dir = ROOT / ('runs/z13_data' if split == 'train26' else 'runs/z13_target_center_dev_20261018/data')
    training.DATA = data_dir
    all_rows = []; offset = 0; vids = PROTOCOL[split]
    try:
        for vi, video in enumerate(vids):
            check(); output = folder / (video + '.json')
            if output.exists():
                rows = read(output); all_rows.extend(rows)
                with np.load(data_dir / (video + '.npz')) as z:
                    offset += len(z['lab_kind'])
                continue
            data = training.Episodes([video]); episodes = {}
            for i, lab in enumerate(data.labels):
                if not lab[8]:
                    episodes.setdefault(lab[0], []).append(i)
            eps = sorted(episodes); rows = []
            with torch.no_grad():
                for ei, e in enumerate(eps):
                    check()
                    ids = episodes[e]
                    hx, masks = data.batch(torch, bb.dev, [e], [max(data.labels[i][1] for i in ids)])
                    need = {N_EXEMPLAR-1} | {N_EXEMPLAR + data.labels[i][1] for i in ids}
                    f = episode_features(torch, 'D', bb, hx, geom.neighbors, need)
                    xe = (f[N_EXEMPLAR-1][0].T-stats[0])/stats[1]
                    xs = np.stack([(f[N_EXEMPLAR+data.labels[i][1]][0].T-stats[0])/stats[1] for i in ids])
                    target = observed_center_target(masks[0].cpu().numpy(), geom.points, geom.spacing)
                    scores = ridge_scores(xe, target, xs, [PROTOCOL['lambda_value']])[0]
                    for j, i in enumerate(ids):
                        lab = data.labels[i]
                        center_eye, peak = center_and_presence(scores[j], geom)
                        center = np.asarray(to_frame(center_eye, lab[7], bank.eye_size))
                        label = offset + i if split == 'train26' else f'{video}:{i}'
                        row = dict(video=video, label=label, pos=int(lab[1]), kind=int(lab[2]),
                                   center=center.tolist(), pred=np.asarray(lab[6]).tolist(),
                                   p_present=peak, useful=False)
                        if lab[2] == 1:
                            row.update(err=float(np.linalg.norm(center-lab[4])),
                                       err_pred=float(np.linalg.norm(np.asarray(lab[6])-lab[4])),
                                       scale=float(max(lab[5][2]-lab[5][0], lab[5][3]-lab[5][1])))
                            row['useful'] = bool(training.success(row))
                        rows.append(row)
                    if ei % 16 == 0:
                        status(stage='evaluation '+folder.name, variant='D', video=video,
                               done=ei+1, total=len(eps), videos_done=vi, videos_total=len(vids))
            write(output, rows); all_rows.extend(rows); offset += len(data.labels)
    finally:
        training.DATA = old_data
    positive = [r for r in all_rows if r['kind'] == 1]
    result = dict(rows=len(all_rows), positives=len(positive), useful=sum(r['useful'] for r in positive),
                  useful_mean_video=float(np.mean(list(per_video_useful(all_rows).values()))),
                  coverage=coverage_table(all_rows))
    write(folder / 'rows.json', all_rows); write(folder / 'summary.json', result)
    return all_rows, result


def compare(rows, split):
    from scripts.z13_continue import paired, per_video_useful
    base = ROOT / ('runs/z13_target_center_20261018' if split == 'train26' else 'runs/z13_target_center_dev_20261018/aggregate')
    references = {arm: read(base / (f'{arm}_center_rows.json' if split != 'train26' else f'{arm}/center_rows.json')) for arm in ('D','C','R1')}
    keys = lambda rr: {(r['video'], str(r['label']), r['kind']) for r in rr}
    assert all(keys(rr) == keys(rows) for rr in references.values())
    current = per_video_useful(rows)
    pairs = {'D_adapted-'+arm+'_center': paired(current, per_video_useful(rr)) for arm, rr in references.items()}
    result = dict(paired=pairs, primary_pass=(pairs['D_adapted-D_center']['ci95'][0] > 0 and
                 pairs['D_adapted-D_center']['mean_per_video'] >= .02 and pairs['D_adapted-C_center']['ci95'][0] > 0))
    return result


def main():
    global PROTOCOL, DEADLINE
    ap=argparse.ArgumentParser(); ap.add_argument('--deadline', type=float, required=True); ap.add_argument('--resume', action='store_true')
    args=ap.parse_args(); DEADLINE=args.deadline; PROTOCOL=read(RUN/'protocol.json'); verify()
    gate=read(ROOT/'runs/z13_parallel_next_20261018/differentiable/report.json')
    gate_end=read(ROOT/'runs/z13_parallel_next_20261018/differentiable/controller_result.json')
    assert gate['primary_pass'] and gate_end['exit_code']==0 and gate_end['worker_ended']
    if (RUN/'report.json').exists():
        raise RuntimeError('Already completed')
    import torch
    from scripts import z13_target_model as old
    from scripts.z13_train import Episodes
    from tracking.z13_readout import FlyBackbone, to_eye
    from tracking.z13_target_center import observed_center_target
    from tracking.z13_target_center_torch import ridge_scores_torch
    from tracking.z13_aligned_loss import query_target, query_loss
    from tracking.z13_checkpoint import save_checkpoint, restore_checkpoint, TrainingPaused
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required for this stage; no implicit CPU fallback')
    torch.set_num_threads(2); torch.manual_seed(0); torch.cuda.manual_seed_all(0); random.seed(0); np.random.seed(0)
    checkpoint=torch.load(ROOT/PROTOCOL['initial_model'],map_location='cpu',weights_only=False)
    bank,geom,frozen=old.build_arm('D',torch,checkpoint)
    bb=FlyBackbone(frozen.est,trainable=True)
    assert str(bb.dev).startswith('cuda')
    device_report=dict(device=str(bb.dev),name=torch.cuda.get_device_name(),free_total_bytes=list(torch.cuda.mem_get_info()))
    write(RUN/'device.json',device_report)
    start_weights={k:v.detach().cpu().clone() for k,v in bb.net.state_dict().items()}
    status(stage='fixed initial feature normalization',variant='D',done=0,total=200)
    if (RUN/'stats.npz').exists():
        with np.load(RUN/'stats.npz') as z:
            stats=(z['mu'],z['sd'])
    else:
        fit20=Episodes(PROTOCOL['fit'][:20]); stats=old.feature_stats(torch,'D',bb,geom,fit20); del fit20
        np.savez(RUN/'stats.npz',mu=stats[0],sd=stats[1])
    identity=PROTOCOL
    ro=torch.nn.Identity().to(bb.dev)
    opt=torch.optim.Adam(bb.param_groups()); rng=np.random.default_rng(0)
    start=0; history=[]
    cp=RUN/'checkpoint.pt'
    if cp.exists():
        if not args.resume:
            raise RuntimeError('Checkpoint exists; explicit resume required')
        start,history=restore_checkpoint(torch,cp,'D',bb,ro,opt,rng,identity)
    if not (RUN/'reproduction.json').exists():
        rows,_=evaluate(torch,bb,bank,geom,stats,RUN/'baseline','train26')
        reference={(r['video'],str(r['label'])):r for r in read(ROOT/'runs/z13_target_center_20261018/D/center_rows.json')}
        assert len(rows)==len(reference)
        dc=max(float(np.max(np.abs(np.asarray(r['center'])-reference[r['video'],str(r['label'])]['center']))) for r in rows)
        dp=max(abs(r['p_present']-reference[r['video'],str(r['label'])]['p_present']) for r in rows)
        passed=dc<.5 and dp<.001
        write(RUN/'reproduction.json',dict(max_center_px=dc,max_score_difference=dp,passed=passed,lambda_value=PROTOCOL['lambda_value']))
        if not passed:
            raise RuntimeError('Initial D not reproduced; tolerance cannot be relaxed')
    else:
        assert read(RUN/'reproduction.json')['passed']
    data=Episodes(PROTOCOL['fit']); usable=[i for i,l in enumerate(data.labels) if l[2]>=0]
    mu=torch.as_tensor(stats[0],device=bb.dev); sd=torch.as_tensor(stats[1],device=bb.dev)
    if not cp.exists():
        save_checkpoint(torch,cp,'D',bb,ro,opt,rng,start,history,identity)
    # A clean save/restore roundtrip precedes optimization.
    start,history=restore_checkpoint(torch,cp,'D',bb,ro,opt,rng,identity)
    current=start; previous_rng=copy.deepcopy(rng.bit_generator.state); inflight=False
    stages={}
    for milestone in (250,1000):
        if (RUN/f'stage{milestone}'/'comparison.json').exists():
            stages[str(milestone)]=read(RUN/f'stage{milestone}'/'comparison.json')
            continue
        t0=time.monotonic(); start_phase=current
        try:
            for step in range(current,milestone):
                check(); previous_rng=copy.deepcopy(rng.bit_generator.state)
                inflight=True
                picked=rng.choice(usable,size=16,replace=False)
                opt.zero_grad(set_to_none=True); objective=0.
                for j,index in enumerate(picked):
                    check(); lab=data.labels[int(index)]
                    hx,masks=data.batch(torch,bb.dev,[lab[0]],[lab[1]])
                    xe,xs=features_with_grad(torch,bb,hx,geom.neighbors)
                    mask=masks[0].cpu().numpy()
                    target=torch.as_tensor(observed_center_target(mask,geom.points,geom.spacing),device=bb.dev)
                    scores=ridge_scores_torch((xe-mu)/sd,target,(xs-mu)/sd,[PROTOCOL['lambda_value']])[0,0]
                    center=to_eye(lab[4],lab[7],bank.eye_size) if lab[2]==1 else None
                    query=torch.as_tensor(query_target(mask,geom.points,geom.spacing,center),device=bb.dev)
                    loss=query_loss(scores,query)
                    if not bool(torch.isfinite(loss)):
                        raise RuntimeError('Nonfinite query loss')
                    (loss/16).backward(); objective+=float(loss.detach())/16
                    status(stage=f'core adaptation to step{milestone}',variant='D',done=step,total=milestone,
                           micro_done=j+1,micro_total=16,last_objective=objective,
                           seconds_per_step=((time.monotonic()-t0)/(step-start_phase+1)))
                object_grad=float(sum(p.grad.detach().square().sum() for p in bb.params if p.grad is not None).sqrt())
                penalty=bb.penalty(); penalty.backward()
                if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in bb.params):
                    raise RuntimeError('Nonfinite core gradient')
                opt.step(); bb.clamp(); current=step+1; inflight=False
                history.append(dict(it=current,object=objective,penalty=float(penalty.detach()),core_object_gradient=object_grad,
                                    batch_sha256=__import__('hashlib').sha256(np.asarray(picked,np.int64).tobytes()).hexdigest()))
                if current%10==0 or current==milestone or current==1:
                    save_checkpoint(torch,cp,'D',bb,ro,opt,rng,current,history,identity)
                    write(RUN/'history.json',history)
                    status(stage=f'core adaptation to step{milestone}',variant='D',done=current,total=milestone,
                           checkpoint_step=current,seconds_per_step=(time.monotonic()-t0)/(current-start_phase),
                           remaining_seconds=(time.monotonic()-t0)/(current-start_phase)*(milestone-current))
        except BaseException:
            # An incomplete microbatch has not updated parameters; replay only that unfinished batch on resume.
            if inflight:
                rng.bit_generator.state=previous_rng
            save_checkpoint(torch,cp,'D',bb,ro,opt,rng,current,history,identity)
            raise
        save_checkpoint(torch,RUN/f'checkpoint_{milestone}.pt','D',bb,ro,opt,rng,current,history,identity)
        folder=RUN/f'stage{milestone}'
        rows,summary=evaluate(torch,bb,bank,geom,stats,folder,'train26')
        comparison=compare(rows,'train26'); comparison['summary']=summary; stages[str(milestone)]=comparison
        write(folder/'comparison.json',comparison)
        if milestone==250:
            delta=comparison['paired']['D_adapted-D_center']
            if not (delta['mean_per_video']>0 and delta['ci95'][1]>=.02):
                verify(); write(RUN/'report.json',dict(stages=stages,decision='stop_after_predeclared_pilot',
                    steps=current,dev_evaluated=False,hota_measured=False,limitations=PROTOCOL['limitations']))
                status(state='completed',stage='pilot criterion did not justify continuation',done=current,total=250)
                return
    full=stages['1000']; dev=None
    if full['primary_pass']:
        check()
        rows,summary=evaluate(torch,bb,bank,geom,stats,RUN/'dev1000','dev35')
        dev=compare(rows,'dev35'); dev['summary']=summary; write(RUN/'dev1000'/'comparison.json',dev)
    verify()
    write(RUN/'report.json',dict(stages=stages,steps=current,dev_evaluated=dev is not None,dev=dev,hota_measured=False,
        decision='candidate_needs_utility_and_tracker_test' if full['primary_pass'] else 'aligned_core_improvement_not_confirmed',
        limitations=PROTOCOL['limitations']))
    status(state='completed',stage='final report saved',done=current,total=1000)


if __name__=='__main__':
    from tracking.gpu_lock import GpuLock
    try:
        with GpuLock('flyvis_fovea_gpu'):
            main()
    except BaseException as exc:
        from tracking.z13_checkpoint import TrainingPaused
        status(state='paused' if isinstance(exc,TrainingPaused) else 'failed',stage=str(exc),done=0,total=0)
        raise
