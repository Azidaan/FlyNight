"""Authorized technical recovery; frozen scientific sources and failed logs remain intact."""
from pathlib import Path
from datetime import datetime, timedelta, timezone
import os, sys, json, hashlib, time, subprocess
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
OLD=ROOT/'runs/z13_distribution_quality_20261015'
NEW=ROOT/'runs/z13_candidates_20261016'
SOFT=datetime.fromisoformat('2026-10-16T07:35:00+03:00')
HARD=datetime.fromisoformat('2026-10-16T07:40:00+03:00')
def now(): return datetime.now(timezone(timedelta(hours=3)))
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):
    from tracking.z13_checkpoint import atomic_json
    atomic_json(p,v)

def cpu_quality():
    import torch
    torch.set_num_threads(2)
    torch.set_default_device('cpu')
    assert torch.empty(1).device.type=='cpu' and not torch.cuda.is_initialized()
    from scripts.z13_spatial_train import quality,persist_torch
    from scripts.z13_train import success
    from tracking.z13_spatial_localizer import percentile_threshold
    p=read(OLD/'protocol.json')
    for name,digest in {**p['source_sha256'],**p['data_sha256']}.items(): assert sha(ROOT/name)==digest,name
    for job in p['jobs']:
        folder=OLD/job['arm']/job['name']; done=read(folder/'completed.json')
        assert done['step']==500 and done['backbone_unchanged']
        assert sha(folder/'predictions.json')==done['prediction_sha256']
    result={}
    for arm in 'BCD':
        assert now()<SOFT
        write(OLD/'recovery_status.json',dict(stage='cpu_quality',arm=arm,updated=now().isoformat()))
        quality(torch,p,arm,OLD)
        rows=read(OLD/arm/'threshold_predictions.json')
        selection=percentile_threshold([r['p_present'] for r in rows],p['fraction'])
        selected=[r for r in rows if r['p_present']>=selection['tau']]
        known=[r for r in selected if r['kind']>=0]
        result[arm]=dict(policy=selection,selected_known=len(known),selected_unknown=len(selected)-len(known),
            useful_known=sum(success(r) for r in known),legacy_90_50=read(OLD/arm/'tau.json')['tau'])
        ck=torch.load(OLD/arm/'model.pt',map_location='cpu',weights_only=False)
        assert all(torch.isfinite(t).all() for key in ('backbone','readout','quality') for t in ck[key].values())
        ck.update(format='z13-spatial-hota-policy-v1',tau={'tau':selection['tau']},policy=selection)
        persist_torch(torch,OLD/arm/'policy_model.pt',ck)
        write(OLD/arm/'policy.json',result[arm])
    assert not torch.cuda.is_initialized()
    write(OLD/'result.json',dict(finished=now().isoformat(),arms=result,hota_measured=False,scope=p['scope']))
    write(OLD/'recovery_verified.json',dict(passed=True,cpu_only=True,frozen_sources_unchanged=True,
        protocol_sha256=sha(OLD/'protocol.json'),recovery_script_sha256=sha(Path(__file__)),
        policies={a:sha(OLD/a/'policy_model.pt') for a in 'BCD'},finished=now().isoformat()))

def controller():
    import msvcrt
    lock=(NEW/'controller.lock').open('a+b'); lock.seek(0)
    msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    assert now()<SOFT and not (NEW/'stop.request').exists()
    assert not (NEW/'resume_controller_result.json').exists()
    assert read(OLD/'controller_result.json')['own_worker_ended']
    assert read(NEW/'controller_result.json')['own_worker_ended']
    write(NEW/'resume_authorization.json',dict(user='Продолжай; не допускать технических простоев',
        at=now().isoformat(),hard_deadline=HARD.isoformat(),save_deadline=SOFT.isoformat(),
        script_sha256=sha(Path(__file__)),protocol_sha256=sha(NEW/'protocol.json'),
        change='Separate CPU quality process, then original candidate worker; settings unchanged'))
    flags=subprocess.CREATE_NO_WINDOW|subprocess.IDLE_PRIORITY_CLASS
    for phase in ('quality','candidates'):
        assert now()<SOFT
        if phase=='candidates': assert read(OLD/'recovery_verified.json')['passed']
        with (NEW/f'resume_{phase}_stdout.log').open('a',encoding='utf-8') as log,(NEW/f'resume_{phase}_stderr.log').open('a',encoding='utf-8') as err:
            proc=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--'+phase],cwd=ROOT,
                stdout=log,stderr=err,creationflags=flags,env={**os.environ,'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2'})
            write(NEW/'resume_controller.json',dict(pid=os.getpid(),worker_pid=proc.pid,phase=phase,started=now().isoformat(),
                deadline=HARD.isoformat(),save_deadline=SOFT.isoformat()))
            reason=None; began=time.time(); stale=None
            while proc.poll() is None:
                if now()>=SOFT:
                    (NEW/'stop.request').write_text('Save for author deadline',encoding='utf-8')
                    if phase=='quality': (OLD/'stop.request').write_text('Save for author deadline',encoding='utf-8')
                status=NEW/'status.json' if phase=='candidates' else OLD/'recovery_status.json'
                age=time.time()-max(began,status.stat().st_mtime if status.exists() else began)
                if age>1200 and stale is None:
                    stale=time.time(); (NEW/'stop.request').write_text('No progress; save',encoding='utf-8')
                if now()>=HARD-timedelta(seconds=30): reason='Absolute deadline'
                elif stale is not None and time.time()-stale>90: reason='Stall after saving grace'
                if reason:
                    subprocess.run(['taskkill','/PID',str(proc.pid),'/T','/F'],creationflags=flags,check=False,timeout=15)
                    proc.wait(timeout=10); break
                time.sleep(2)
            end=dict(phase=phase,exit_code=proc.returncode,own_worker_ended=proc.poll() is not None,reason=reason,ended=now().isoformat())
            write(NEW/f'resume_{phase}_result.json',end)
            if proc.returncode!=0:
                write(NEW/'resume_controller_result.json',end); return
    write(NEW/'resume_controller_result.json',end)

if __name__=='__main__':
    if '--quality' in sys.argv: cpu_quality()
    elif '--candidates' in sys.argv:
        assert read(OLD/'recovery_verified.json')['passed']
        from scripts.z13_candidates_night import worker
        raise SystemExit(worker())
    else: controller()
