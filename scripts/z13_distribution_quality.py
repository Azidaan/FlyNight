"""Prepare and run train-only OOF quality fitting for the fixed spatial pilot.

Preparation is CPU-only. Execution requires an explicit bounded --hours window.
No dev evaluation, new localization settings, or mutation of source runs.
"""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import argparse
import hashlib
import json
import os
import random
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SRC = ROOT / 'runs/z13_spatial_v2_20261013'
PILOT = ROOT / 'runs/z13_distribution_pilot_20261015'
OUT = ROOT / 'runs/z13_distribution_quality_20261015'
ID = 'z13_distribution_quality_v1'

def read(path): return json.loads(path.read_text(encoding='utf-8'))
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def now(): return datetime.now(timezone(timedelta(hours=3)))

def validate_jobs(p):
    fit, calibration = set(p['fit']), set(p['tau'])
    assert len(fit) == 104 and len(calibration) == 26 and not fit & calibration
    assert len(p['jobs']) == 6
    for arm in 'BCD':
        jobs = [j for j in p['jobs'] if j['arm'] == arm]
        assert {j['name'] for j in jobs} == {'fold0', 'fold1'}
        seen = set()
        for j in jobs:
            a, b = set(j['fit']), set(j['predict'])
            assert len(a) == len(b) == 52 and not a & b and a | b == fit
            assert not seen & b
            seen |= b
        assert seen == fit

def prepare():
    import torch
    from tracking.z13_checkpoint import atomic_json
    from scripts.z13_spatial_train import persist_torch
    assert not (OUT / 'protocol.json').exists(), 'Do not rewrite a frozen protocol'
    p = read(SRC / 'protocol.json')
    p.update(id=ID, created=now().isoformat(), steps=500,
        jobs=[j for j in p['jobs'] if j['name'] != 'full'],
        scope='Preparation authorized by user continuation. GPU requires a separately authorized bounded window. Train only; no dev/HOTA/heldout/downloads.',
        continuation='Spatial target, same 500 steps/lr/batches as pilot; each fold starts from its own prior fold model. All backbones frozen.',
        policy='Same pre-existing top 1 percent on all natural train26 scores; ties retained; labels not used for percentile. Original 90 percent / 50 rule reported separately, never relabelled as passed.',
        fraction=.01,
        stop='Any provenance, leakage, nonfinite or resume failure stops. No automatic HOTA and no parameter search.',
        limitations=p['limitations'] + ['Pilot target chosen using these train26: all diagnostics exploratory.',
            'Fold feature extractors differ from full models; unseen-video predictions prevent direct fitting leakage but do not remove distribution shift.'])
    validate_jobs(p)
    frozen = read(PILOT / 'protocol.json')['source_sha256']
    for name, digest in frozen.items(): assert sha(ROOT / name) == digest, name
    sources = [Path(__file__), ROOT/'tracking/z13_distribution_target.py', ROOT/'scripts/z13_spatial_train.py',
        ROOT/'scripts/z13_train.py', ROOT/'tracking/z13_checkpoint.py', ROOT/'tracking/z13_spatial.py',
        ROOT/'tracking/z13_readout.py', ROOT/'tracking/z13_spatial_localizer.py', ROOT/'scripts/z13_threshold_diagnostic.py',
        PILOT/'protocol.json', SRC/'protocol.json']
    for job in p['jobs']:
        folder = SRC / job['arm'] / job['name']
        ident = read(folder/'identity.json')
        assert ident['config']['job'] == job
        for name, digest in ident['files_sha256'].items(): assert sha(ROOT/name) == digest, name
        sources.extend([folder/'model.pt', folder/'identity.json'])
    for arm in 'BCD':
        sources.extend([PILOT/arm/'spatial/checkpoint.pt', PILOT/arm/'spatial/predictions.json', SRC/arm/'full/model.pt'])
    p['source_sha256'] = {str(x.relative_to(ROOT)): sha(x) for x in sources}
    p['data_sha256'] = {k:v for k,v in frozen.items() if k.endswith('.npz')}
    OUT.mkdir(exist_ok=True)
    atomic_json(OUT/'protocol.json', p)
    exported = {}
    for arm in 'BCD':
        base = torch.load(SRC/arm/'full/model.pt', map_location='cpu', weights_only=False)
        ck = torch.load(PILOT/arm/'spatial/checkpoint.pt', map_location='cpu', weights_only=False)
        assert ck['step'] == 500 and ck['identity']['arm'] == arm and ck['identity']['loss'] == 'spatial'
        assert all(torch.equal(v, ck['backbone'][k]) for k,v in base['backbone'].items())
        folder = OUT/arm/'full'; folder.mkdir(parents=True)
        base.update(readout=ck['readout'], tau=None, identity={'source_pilot':str(PILOT.relative_to(ROOT)), 'source_sha256':sha(PILOT/arm/'spatial/checkpoint.pt')})
        base.pop('quality', None)
        persist_torch(torch, folder/'model.pt', base)
        pred = PILOT/arm/'spatial/predictions.json'
        (folder/'predictions.json').write_bytes(pred.read_bytes())
        exported[arm] = dict(rows=len(read(pred)), predictions_exact=sha(pred)==sha(folder/'predictions.json'), step=500)
    atomic_json(OUT/'preflight.json', dict(passed=True, exported=exported, jobs=6, fit=104, calibration=26,
        no_gpu=True, gpu_started=False, new_full_localizer_training=False))
    print(json.dumps(read(OUT/'preflight.json')))

def worker(deadline):
    import numpy as np
    import torch
    from tracking.gpu_lock import GpuLock
    from tracking.z13_checkpoint import RunControl, atomic_json, save_checkpoint, restore_checkpoint
    from scripts.z13_spatial_train import build, predict, quality, persist_torch
    from scripts.z13_train import Episodes, success
    from tracking.z13_readout import run_sequence, to_eye
    from tracking.z13_distribution_target import spatial_targets, distribution_loss
    from tracking.z13_spatial_localizer import percentile_threshold
    p = read(OUT/'protocol.json'); validate_jobs(p)
    for name, digest in {**p['source_sha256'], **p['data_sha256']}.items(): assert sha(ROOT/name)==digest,name
    control = RunControl(OUT, OUT/'stop.request', deadline, resume=True)
    torch.set_num_threads(2)
    with GpuLock('flyvis_fovea_gpu'):
        for job in p['jobs']:
            control.check()
            arm = job['arm']; folder = OUT/arm/job['name']; folder.mkdir(parents=True, exist_ok=True)
            ident = {'protocol_sha256':sha(OUT/'protocol.json'), 'config':{'job':job}}
            if (folder/'completed.json').exists():
                assert read(folder/'identity.json') == ident
                assert sha(folder/'predictions.json') == read(folder/'completed.json')['prediction_sha256']
                continue
            atomic_json(folder/'identity.json', ident)
            torch.manual_seed(0); np.random.seed(0); random.seed(0)
            est, bank, geom, bb, ro, dev = build(arm, torch)
            ck = torch.load(SRC/arm/job['name']/'model.pt', map_location='cpu', weights_only=False)
            core = bb if arm=='B' else bb.net
            core.load_state_dict(ck['backbone']); ro.load_state_dict(ck['readout'])
            for param in core.parameters(): param.requires_grad_(False)
            opt = torch.optim.Adam(ro.parameters(), lr=p['readout_lr']); rng=np.random.default_rng(0)
            checkpoint=folder/'checkpoint.pt'; hist=[]; start=0
            if checkpoint.exists(): start,hist=restore_checkpoint(torch,checkpoint,arm,bb,ro,opt,rng,ident)
            else: save_checkpoint(torch,checkpoint,arm,bb,ro,opt,rng,0,hist,ident)
            data=Episodes(job['fit']); usable=[i for i,r in enumerate(data.labels) if r[2]>=0]
            points=torch.as_tensor(geom.points,device=dev); ro.train()
            for step in range(start,p['steps']):
                control.check(); picked=rng.choice(usable,p['batch'],replace=False); opt.zero_grad(set_to_none=True); value=0.
                for left in range(0,p['batch'],p['micro_batch']):
                    labels=[data.labels[i] for i in picked[left:left+p['micro_batch']]]
                    hx,mask=data.batch(torch,dev,[r[0] for r in labels],[r[1] for r in labels])
                    logits=run_sequence(torch,bb,ro,hx,mask)
                    selected=logits[torch.arange(len(labels),device=dev),torch.as_tensor([r[1] for r in labels],device=dev)]
                    centers=[to_eye(r[4],r[7],bank.eye_size) if r[2]==1 else (0.,0.) for r in labels]
                    target,known=spatial_targets(torch,[r[2] for r in labels],centers,points,geom.spacing)
                    loss=distribution_loss(torch,selected,target,known)
                    assert bool(torch.isfinite(loss))
                    (loss*len(labels)/p['batch']).backward(); value+=float(loss.detach())*len(labels)/p['batch']
                assert all(q.grad is None or bool(torch.isfinite(q.grad).all()) for q in ro.parameters())
                opt.step(); hist.append(dict(step=step+1,loss=value))
                if (step+1)%10==0 or step+1==p['steps'] or control.stopped():
                    save_checkpoint(torch,checkpoint,arm,bb,ro,opt,rng,step+1,hist,ident)
                    control.status(state='running',stage='oof_training',arm=arm,fold=job['name'],step=step+1,total=p['steps'],checkpoint_verified=True)
                control.check()
            assert all(torch.equal(v.cpu(),ck['backbone'][k].cpu()) for k,v in core.state_dict().items())
            del data
            rows=predict(torch,Episodes(job['predict']),bb,ro,dev,geom,bank.eye_size,control)
            atomic_json(folder/'predictions.json',rows)
            atomic_json(folder/'completed.json',dict(step=500,rows=len(rows),prediction_sha256=sha(folder/'predictions.json'),backbone_unchanged=True))
            del bb,core,ro,bank,est,opt,ck; torch.cuda.empty_cache()
        result={}
        for arm in 'BCD':
            control.check()
            # CPU quality fit is short; finish it atomically, then check the deadline again.
            quality(torch,p,arm,OUT)
            control.check()
            rows=read(OUT/arm/'threshold_predictions.json')
            selection=percentile_threshold([r['p_present'] for r in rows],p['fraction'])
            selected=[r for r in rows if r['p_present']>=selection['tau']]
            known=[r for r in selected if r['kind']>=0]
            result[arm]=dict(policy=selection,selected_known=len(known),selected_unknown=len(selected)-len(known),
                useful_known=sum(success(r) for r in known),legacy_90_50=read(OUT/arm/'tau.json')['tau'])
            ck=torch.load(OUT/arm/'model.pt',map_location='cpu',weights_only=False)
            ck.update(format='z13-spatial-hota-policy-v1',tau={'tau':selection['tau']},policy=selection)
            persist_torch(torch,OUT/arm/'policy_model.pt',ck)
            atomic_json(OUT/arm/'policy.json',result[arm])
        atomic_json(OUT/'result.json',dict(finished=now().isoformat(),arms=result,hota_measured=False,scope=p['scope']))
        control.status(state='completed',stage='quality_and_policy_saved')

def controller(hours):
    from tracking.z13_checkpoint import atomic_json
    import msvcrt
    assert 0 < hours <= 4
    lock=(OUT/'controller.lock').open('a+b'); lock.write(b'0'); lock.flush(); lock.seek(0)
    msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    assert not (OUT/'stop.request').exists(), 'Do not silently clear stop requests'
    hard=now()+timedelta(hours=hours); soft=hard-timedelta(minutes=5)
    flags=subprocess.CREATE_NO_WINDOW|subprocess.IDLE_PRIORITY_CLASS
    with (OUT/'stdout.log').open('a',encoding='utf-8') as log, (OUT/'stderr.log').open('a',encoding='utf-8') as err:
        proc=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--worker','--deadline',soft.isoformat()],cwd=ROOT,
            creationflags=flags,stdout=log,stderr=err,env={**os.environ,'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2'})
        atomic_json(OUT/'controller.json',dict(pid=os.getpid(),worker_pid=proc.pid,started=now().isoformat(),deadline=hard.isoformat(),save_deadline=soft.isoformat()))
        try: code=proc.wait(timeout=max(1,(hard-now()).total_seconds()))
        except subprocess.TimeoutExpired:
            (OUT/'stop.request').write_text('Window limit',encoding='utf-8')
            subprocess.run(['taskkill','/PID',str(proc.pid),'/T','/F'],creationflags=flags,check=False)
            code=proc.wait(timeout=30)
        atomic_json(OUT/'controller_result.json',dict(exit_code=code,ended=now().isoformat(),own_worker_ended=proc.poll() is not None))
    lock.close()

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--prepare',action='store_true'); parser.add_argument('--worker',action='store_true')
    parser.add_argument('--deadline'); parser.add_argument('--hours',type=float)
    args=parser.parse_args()
    if args.prepare: prepare()
    elif args.worker:
        assert args.deadline
        from tracking.z13_checkpoint import TrainingPaused
        try: worker(args.deadline)
        except TrainingPaused: raise SystemExit(75)
    elif args.hours: controller(args.hours)
    else: parser.error('Choose --prepare or authorized --hours window')
