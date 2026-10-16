"""Three bounded functional module probes. No dev, tracker run, HOTA or downloads."""
from pathlib import Path
from datetime import datetime, timedelta, timezone
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
import hashlib
import json
import random
import subprocess
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'runs/z13_candidates_20261016'
PREVIOUS = ROOT / 'runs/z13_distribution_quality_20261015'
PILOT = ROOT / 'runs/z13_distribution_pilot_20261015'
SRC = ROOT / 'runs/z13_spatial_v2_20261013'
HARD = '2026-10-16T07:40:00+03:00'
SOFT = '2026-10-16T07:35:00+03:00'

def read(p): return json.loads(p.read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def now(): return datetime.now(timezone(timedelta(hours=3)))
def write(p, data):
    from tracking.z13_checkpoint import atomic_json
    atomic_json(p, data)


def prepare():
    assert not (OUT/'protocol.json').exists(), 'Frozen protocol already exists'
    p = read(PILOT/'protocol.json')
    sources = [Path(__file__), ROOT/'tracking/z13_candidate_modules.py',
               ROOT/'scripts/z13_candidates_analyze.py', ROOT/'scripts/z13_candidates_record.py',
               ROOT/'scripts/z13_train.py', ROOT/'scripts/z13_spatial_train.py',
               ROOT/'tracking/z13_spatial.py', ROOT/'tracking/z13_readout.py',
               ROOT/'tracking/z13_checkpoint.py', ROOT/'tracking/z13_distribution_target.py',
               ROOT/'tracking/gpu_lock.py', PILOT/'protocol.json',
               ROOT/'tests/test_z13_candidate_modules.py',
               ROOT/'research_flyvis_china_2026-10-15/candidate_data_check.json']
    for arm in 'BCD':
        sources += [PILOT/arm/'spatial/checkpoint.pt', SRC/arm/'full/model.pt']
    # Source pilot already binds its dependencies and input arrays.
    hashes = dict(p['source_sha256'])
    hashes.update({str(f.relative_to(ROOT)): sha(f) for f in sources})
    for v in p['fit'] + p['evaluation']:
        f = ROOT/'runs/fly_signatures'/v/'signatures.npz'
        hashes[str(f.relative_to(ROOT))] = sha(f)
    jobs = [{'arm': a, 'mode': m} for m in ['base', 'foreground', 'size', 'fixed', 'adaptive'] for a in 'BCD']
    protocol = dict(id='z13_candidates_20261016', created=now().isoformat(),
        authorization='Author: test all three proposals; absolute limit 07:40 MSK 16.10.2026',
        fit=p['fit'], evaluation=p['evaluation'], jobs=jobs, steps=500, batch=16, micro=4, lr=.003, seed=0,
        scope='Exploratory functional probes on train104/26 only. No HOTA/dev/heldout/downloads.',
        initialization='Immutable spatial-pilot full checkpoint in every arm; all backbones frozen.',
        foreground='Context contrast outside dilated observed exemplar box; gain clamped [0,2], initially zero; same spatial distribution loss.',
        size='Frozen parent localizer; train only 64->32->2 head, log-size residual in [-log2,log2] against last OBSERVED tracker size, SmoothL1. Same predicted centers for both sizes.',
        temporal='Nearest-column transport by known crop geometry; unmatched columns reset to current features. Fixed alpha=.5; adaptive sigmoid(b+s*mean absolute standardized change), b=s=0 initially.',
        sampling='Same seed and known rows for all localizer continuations. Size uses same rows, loss only on known positives.',
        selection='Top 1 percent natural train26 presence scores, ties retained, including unknowns; no tuning. Unknown outcomes reported separately.',
        criteria={
            'foreground':['D foreground - D base useful-center fraction CI lower >0', 'D foreground - B foreground useful-center fraction CI lower >0', 'known-bad fraction in selected rows <= D base'],
            'size':['D size mean per-video IoU(new size - last-observed size), SAME predicted centers: CI lower >0', 'D size - B size new-size IoU CI lower >0', 'Parent centers unchanged within 1e-5 numerical tolerance'],
            'adaptive':['D adaptive - D fixed useful-center fraction CI lower >0', 'D adaptive - B adaptive useful-center fraction CI lower >0', 'known-bad fraction in selected rows <= D fixed']},
        confidence='Paired video bootstrap, 2000 replicates seed20261016, 95% intervals; exploratory reused train26 and one seed, no multiplicity-adjusted discovery claim.',
        limitations=['Last observed size is not current Kalman predicted size; size test does not establish tracker benefit.',
                    'B and D size comparisons may have different centers; only within-arm size delta holds centers fixed.',
                    'Background context can contain other objects; no true foreground masks.',
                    'No further core training; D is previously trained frozen FlyVis.',
                    'Fixed 500 steps is not evidence of convergence.',
                    'No combined-module claim; no numerical efficacy from incomplete comparison sets.'],
        stop='Technical mismatch/nonfinite/provenance error stops whole suite. No retries or post-result hyperparameter changes. Failed scientific gate closes only that implementation; remaining predeclared jobs continue.',
        hard_deadline=HARD, save_deadline=SOFT, source_sha256=hashes)
    OUT.mkdir(exist_ok=True)
    write(OUT/'protocol.json', protocol)
    write(OUT/'authorization.json', {'user_reply':'7:40 16.10.2026','hard_deadline':HARD,'save_deadline':SOFT,'scope':protocol['scope']})
    print('Prepared', len(jobs), 'jobs; no GPU started')


def contextual_data(vids):
    from scripts.z13_train import Episodes
    data = Episodes(vids)
    crops, widths, videos = [], [], []
    for v in vids:
        with np.load(ROOT/'runs/z13_data'/f'{v}.npz', allow_pickle=False) as z:
            offsets, frames, ids = z['ep_offset'], z['fr_frame'], z['ep_tid']
            crops.append(z['fr_crop'])
        with np.load(ROOT/'runs/fly_signatures'/v/'signatures.npz', allow_pickle=False) as z:
            outputs = z['outputs']
        lookup = {(int(row[0]),int(row[5])):row[1:5] for row in outputs}
        for i, tid in enumerate(ids):
            box = lookup[(int(frames[offsets[i]+3]), int(tid))]
            size = box[2:] - box[:2]
            assert np.isfinite(size).all() and (size>0).all()
            widths.append(size); videos.append(v)
    data.crops = np.concatenate(crops)
    data.widths = np.asarray(widths)
    data.videos = videos
    assert len(data.widths) == len(data.off)-1
    return data


def batch(data, torch, device, eps, upto):
    hx, masks = data.batch(torch, device, eps, upto)
    idx = np.stack([data.off[e]+np.minimum(np.arange(hx.shape[1]),data.n_frames(e)-1) for e in eps])
    return hx, masks, data.crops[idx]


def build(torch, arm, mode):
    from scripts.z13_spatial_train import build as original
    from tracking.z13_spatial import make_spatial_readout
    from tracking.z13_candidate_modules import make_candidate
    est, bank, geom, bb, old, dev = original(arm, torch)
    base = torch.load(PILOT/arm/'spatial/checkpoint.pt', map_location='cpu', weights_only=False)
    core = bb if arm=='B' else bb.net
    core.load_state_dict(base['backbone'])
    for param in core.parameters(): param.requires_grad_(False)
    ro = make_candidate(torch, make_spatial_readout(torch))(34, geom.points, bank.eye_size, geom.neighbors, mode).to(dev)
    missing, unexpected = ro.load_state_dict(base['readout'], strict=False)
    allowed = {'foreground':['background_gain'], 'adaptive':['time_bias','time_slope'],
               'size':['size_head.0.weight','size_head.0.bias','size_head.2.weight','size_head.2.bias']}.get(mode, [])
    assert set(missing)==set(allowed) and not unexpected
    if mode=='size':
        for name, param in ro.named_parameters(): param.requires_grad_(name.startswith('size_head.'))
    del old
    return est, bank, geom, bb, ro, dev, base


def technical(torch, data, bb, ro, geom, bank, device, folder, arm, identity, optimizer, rng):
    from tracking.z13_checkpoint import save_checkpoint, restore_checkpoint
    from tracking.z13_candidate_modules import run_candidate
    from tracking.z13_readout import run_sequence
    from tracking.z13_spatial import make_spatial_readout
    hx, mask, crops = batch(data, torch, device, [0], [0])
    parent = make_spatial_readout(torch)(34, geom.points, bank.eye_size, geom.neighbors).to(device)
    parent.load_state_dict({k:v for k,v in ro.state_dict().items() if k in parent.state_dict()})
    ro.eval(); parent.eval()
    with torch.no_grad():
        got, size = run_candidate(torch, bb, ro, hx, mask, crops, geom, bank.eye_size)
        want = run_sequence(torch, bb, parent, hx, mask)
    assert torch.allclose(got,want,atol=1e-5,rtol=1e-5), 'Initial one-step parent equivalence failed'
    if size is not None: assert torch.equal(size,torch.zeros_like(size))
    path = folder/'technical_checkpoint.pt'
    # Put optimizer moments in the snapshot and verify a real next step after restoration.
    params = [q for q in ro.parameters() if q.requires_grad]
    before = {k:v.detach().clone() for k,v in ro.state_dict().items()}
    optimizer.zero_grad(); sum(q.square().sum() for q in params).backward(); optimizer.step()
    save_checkpoint(torch,path,arm,bb,ro,optimizer,rng,1,[],identity)
    saved = torch.load(path,map_location='cpu',weights_only=False)
    draws = rng.integers(100000,size=8); draws_t = torch.rand(8,device=device)
    optimizer.zero_grad(); sum(q.square().sum() for q in params).backward(); optimizer.step()
    next_state = {k:v.detach().clone() for k,v in ro.state_dict().items()}
    step,_ = restore_checkpoint(torch,path,arm,bb,ro,optimizer,rng,identity)
    assert step==1 and np.array_equal(draws,rng.integers(100000,size=8)) and torch.equal(draws_t,torch.rand(8,device=device))
    assert all(torch.equal(v.cpu(),saved['readout'][k]) for k,v in ro.state_dict().items())
    optimizer.zero_grad(); sum(q.square().sum() for q in params).backward(); optimizer.step()
    assert all(torch.equal(v,next_state[k]) for k,v in ro.state_dict().items()), 'Optimizer resume differs'
    ro.load_state_dict(before)
    write(folder/'technical.json',dict(passed=True,initial_forward_close=True,weights_optimizer_rng_restore_exact=True))
    del parent


def predict(torch, data, bb, ro, geom, bank, dev, control, folder):
    from tracking.z13_candidate_modules import run_candidate, box_iou
    from tracking.z13_readout import center_estimate, to_frame, run_sequence
    from tracking.z13_spatial import make_spatial_readout
    parent = None
    if ro.mode=='size':
        parent = make_spatial_readout(torch)(34,geom.points,bank.eye_size,geom.neighbors).to(dev)
        parent.load_state_dict({k:v for k,v in ro.state_dict().items() if k in parent.state_dict()}); parent.eval()
    by_episode = {}
    for idx,l in enumerate(data.labels):
        if not l[8]: by_episode.setdefault(l[0],[]).append(idx)
    eps = sorted(by_episode); rows=[]; progress=folder/'prediction_progress.json'
    start=0; max_parent_difference=0.
    if progress.exists():
        previous=read(progress); rows=previous['rows']; start=previous['next']
        max_parent_difference=previous.get('max_parent_difference',0.)
    ro.eval()
    with torch.no_grad():
        for left in range(start,len(eps),4):
            control.check()
            es=eps[left:left+4]
            hx,mask,crops=batch(data,torch,dev,es,[max(data.labels[i][1] for i in by_episode[e]) for e in es])
            logits,delta=run_candidate(torch,bb,ro,hx,mask,crops,geom,bank.eye_size)
            if parent is not None:
                expected=run_sequence(torch,bb,parent,hx,mask)
                assert torch.allclose(logits,expected,atol=1e-5,rtol=1e-5), 'Size changed center logits'
                max_parent_difference=max(max_parent_difference,float((logits-expected).abs().max()))
            probs=logits.softmax(-1).cpu().numpy()
            factors=delta.exp().cpu().numpy() if delta is not None else None
            for k,e in enumerate(es):
                for idx in by_episode[e]:
                    l=data.labels[idx]; p=probs[k,l[1]]
                    center=np.asarray(to_frame(center_estimate(p[:721],geom),l[7],bank.eye_size))
                    row=dict(label=idx,episode=int(e),video=data.videos[e],search_position=int(l[1]),kind=int(l[2]),p_present=1-float(p[721]),center=center.tolist())
                    if l[2]==1:
                        size=data.widths[e]; proposed=size if factors is None else size*factors[k,l[1]]
                        row.update(err=float(np.linalg.norm(center-l[4])),err_pred=float(np.linalg.norm(l[6]-l[4])),scale=float(max(l[5][2:]-l[5][:2])),
                            iou_last_size=box_iou(center,size,l[5]),iou_new_size=box_iou(center,proposed,l[5]))
                    rows.append(row)
            if left%100==0 or left+4>=len(eps) or control.stopped():
                write(progress,dict(next=min(left+4,len(eps)),rows=rows,max_parent_difference=max_parent_difference))
                control.status(stage='prediction',episodes_done=min(left+4,len(eps)),episodes_total=len(eps))
    if parent is not None: write(folder/'size_center_check.json',dict(passed=True,max_logits_difference=max_parent_difference))
    return rows


def worker():
    import torch
    from tracking.gpu_lock import GpuLock
    from tracking.z13_checkpoint import RunControl, TrainingPaused, save_checkpoint, restore_checkpoint
    from tracking.z13_distribution_target import spatial_targets, distribution_loss
    from tracking.z13_candidate_modules import run_candidate
    from tracking.z13_readout import to_eye
    p=read(OUT/'protocol.json'); torch.set_num_threads(2)
    control=RunControl(OUT,OUT/'stop.request',SOFT,resume=True)
    control.status(state='running',stage='checking_sources')
    for name,digest in p['source_sha256'].items(): assert sha(ROOT/name)==digest,name
    control.status(stage='loading_train')
    fit=contextual_data(p['fit']); evaluation=contextual_data(p['evaluation'])
    try:
        with GpuLock('flyvis_fovea_gpu'):
            control.check()
            for job in p['jobs']:
                arm,mode=job['arm'],job['mode']; folder=OUT/arm/mode; folder.mkdir(parents=True,exist_ok=True)
                ident=dict(protocol_sha256=sha(OUT/'protocol.json'),arm=arm,mode=mode)
                if (folder/'completed.json').exists():
                    assert read(folder/'identity.json')==ident
                    assert sha(folder/'predictions.json')==read(folder/'completed.json')['prediction_sha256']
                    continue
                control.check(); write(folder/'identity.json',ident)
                control.status(stage='technical_or_restore',arm=arm,mode=mode,step=0)
                torch.manual_seed(0); random.seed(0); np.random.seed(0)
                est,bank,geom,bb,ro,dev,base=build(torch,arm,mode)
                params=[q for q in ro.parameters() if q.requires_grad]
                opt=torch.optim.Adam(params,lr=p['lr']); rng=np.random.default_rng(0)
                checkpoint=folder/'checkpoint.pt'; hist=[]; start=0
                if checkpoint.exists(): start,hist=restore_checkpoint(torch,checkpoint,arm,bb,ro,opt,rng,ident)
                else:
                    technical(torch,fit,bb,ro,geom,bank,dev,folder,arm,ident,opt,rng)
                    torch.manual_seed(0); random.seed(0); np.random.seed(0); rng=np.random.default_rng(0)
                    opt=torch.optim.Adam(params,lr=p['lr'])
                    save_checkpoint(torch,checkpoint,arm,bb,ro,opt,rng,0,hist,ident)
                points=torch.as_tensor(geom.points,device=dev)
                useful=[i for i,r in enumerate(fit.labels) if r[2]>=0]
                ro.train(); started=time.monotonic(); last_saved=started
                for step in range(start,p['steps']):
                    control.check(); picked=rng.choice(useful,p['batch'],replace=False); opt.zero_grad(set_to_none=True); value=0.
                    for left in range(0,p['batch'],p['micro']):
                        labels=[fit.labels[i] for i in picked[left:left+p['micro']]]
                        hx,mask,crops=batch(fit,torch,dev,[r[0] for r in labels],[r[1] for r in labels])
                        logits,delta=run_candidate(torch,bb,ro,hx,mask,crops,geom,bank.eye_size)
                        inds=torch.arange(len(labels),device=dev); positions=torch.as_tensor([r[1] for r in labels],device=dev)
                        if mode=='size':
                            selected=delta[inds,positions]; positive=torch.as_tensor([r[2]==1 for r in labels],device=dev)
                            targets=np.stack([np.log((r[5][2:]-r[5][:2])/fit.widths[r[0]]) if r[2]==1 else np.zeros(2) for r in labels])
                            loss=torch.nn.functional.smooth_l1_loss(selected[positive],torch.as_tensor(targets,device=dev,dtype=selected.dtype)[positive]) if bool(positive.any()) else selected.sum()*0
                        else:
                            centers=[to_eye(r[4],r[7],bank.eye_size) if r[2]==1 else (0.,0.) for r in labels]
                            target,known=spatial_targets(torch,[r[2] for r in labels],centers,points,geom.spacing)
                            loss=distribution_loss(torch,logits[inds,positions],target,known)
                        assert bool(torch.isfinite(loss))
                        (loss*len(labels)/p['batch']).backward(); value+=float(loss.detach())*len(labels)/p['batch']
                    assert all(q.grad is None or bool(torch.isfinite(q.grad).all()) for q in params)
                    opt.step()
                    if mode=='foreground':
                        with torch.no_grad(): ro.background_gain.clamp_(0,2)
                    hist.append(dict(step=step+1,loss=value))
                    if (step+1)%10==0 or time.monotonic()-last_saved>=60 or step+1==p['steps'] or control.stopped():
                        save_checkpoint(torch,checkpoint,arm,bb,ro,opt,rng,step+1,hist,ident); last_saved=time.monotonic()
                        control.status(state='running',stage='training',arm=arm,mode=mode,step=step+1,total=p['steps'],checkpoint_verified=True,
                            seconds_per_step=(time.monotonic()-started)/max(step+1-start,1))
                    control.check()
                core=bb if arm=='B' else bb.net
                assert all(torch.equal(v.cpu(),base['backbone'][k].cpu()) for k,v in core.state_dict().items())
                if mode=='size': assert all(torch.equal(v.cpu(),base['readout'][k].cpu()) for k,v in ro.state_dict().items() if k in base['readout'])
                rows=predict(torch,evaluation,bb,ro,geom,bank,dev,control,folder)
                write(folder/'predictions.json',rows)
                write(folder/'completed.json',dict(step=500,rows=len(rows),backbone_unchanged=True,prediction_sha256=sha(folder/'predictions.json'),finished=now().isoformat()))
                del bb,ro,core,est,bank,opt,base,params; torch.cuda.empty_cache()
            control.status(state='completed',stage='all_fifteen_jobs_saved')
    except TrainingPaused:
        control.status(state='paused',saved=True)
        return 75
    return 0


def controller():
    import msvcrt
    flags=subprocess.CREATE_NO_WINDOW|subprocess.IDLE_PRIORITY_CLASS
    lock=(OUT/'controller.lock').open('a+b'); lock.write(b'0'); lock.flush(); lock.seek(0)
    msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    assert not (OUT/'controller_result.json').exists(), 'Controller already finished; explicit resume required'
    assert now()<datetime.fromisoformat(SOFT)
    write(OUT/'controller.json',dict(pid=os.getpid(),state='waiting_previous',deadline=HARD,save_deadline=SOFT,started=now().isoformat()))
    # No GPU initialization until the previous independent controller has closed its worker.
    while not (PREVIOUS/'controller_result.json').exists():
        if now()>=datetime.fromisoformat(SOFT):
            write(OUT/'controller_result.json',dict(exit_code=75,own_worker_ended=True,worker_started=False,ended=now().isoformat())); return
        time.sleep(10)
    previous=read(PREVIOUS/'controller_result.json')
    if previous['exit_code']!=0 or not previous['own_worker_ended']:
        write(OUT/'controller_result.json',dict(exit_code=76,own_worker_ended=True,worker_started=False,reason='Previous stage not complete; needs review',ended=now().isoformat())); return
    with (OUT/'stdout.log').open('a',encoding='utf-8') as log,(OUT/'stderr.log').open('a',encoding='utf-8') as err:
        proc=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--worker'],cwd=ROOT,stdout=log,stderr=err,creationflags=flags)
        write(OUT/'controller.json',dict(pid=os.getpid(),worker_pid=proc.pid,state='running',deadline=HARD,save_deadline=SOFT,started=now().isoformat()))
        forced_reason=None; stale_since=None; started=time.time()
        while proc.poll() is None:
            current=now()
            if current>=datetime.fromisoformat(SOFT) and not (OUT/'stop.request').exists():
                (OUT/'stop.request').write_text('Save before absolute author deadline',encoding='utf-8')
            status=OUT/'status.json'
            updated=status.stat().st_mtime if status.exists() else started
            if time.time()-updated>1200 and stale_since is None:
                stale_since=time.time()
                (OUT/'stop.request').write_text('No progress for 20 minutes; save and stop',encoding='utf-8')
            if current>=datetime.fromisoformat(HARD)-timedelta(seconds=30):
                forced_reason='Hard deadline safety margin'
            elif stale_since is not None and time.time()-stale_since>90:
                forced_reason='Stall after save grace period'
            if forced_reason:
                subprocess.run(['taskkill','/PID',str(proc.pid),'/T','/F'],creationflags=flags,check=False,timeout=15)
                proc.wait(timeout=10)
                break
            time.sleep(2)
        write(OUT/'controller_result.json',dict(exit_code=proc.returncode,own_worker_ended=proc.poll() is not None,forced_reason=forced_reason,ended=now().isoformat()))
    lock.close()


if __name__=='__main__':
    if '--prepare' in sys.argv: prepare()
    elif '--worker' in sys.argv: raise SystemExit(worker())
    else: controller()
