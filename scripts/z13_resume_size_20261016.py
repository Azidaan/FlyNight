"""Service-only recovery: compare size readouts on the same activity realization."""
from pathlib import Path
from datetime import timedelta
import sys, os, time, subprocess, inspect
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import z13_candidates_night as n
from scripts.z13_size_check_recovery import Capture, Replay
OUT=n.OUT

def install():
    # Keep the actual candidate forward unchanged; replay its input features ONLY
    # for the diagnostic parent forward, avoiding a second numerical core run.
    source=inspect.getsource(n.predict)
    old='            logits,delta=run_candidate(torch,bb,ro,hx,mask,crops,geom,bank.eye_size)'
    new='            capture=Capture(bb)\n            logits,delta=run_candidate(torch,capture,ro,hx,mask,crops,geom,bank.eye_size)'
    assert source.count(old)==1
    source=source.replace(old,new)
    old='                expected=run_sequence(torch,bb,parent,hx,mask)'
    new='                expected=run_sequence(torch,Replay(capture.features),parent,hx,mask)\n                assert torch.equal(logits,expected), "Same-activity parent logits differ"'
    assert source.count(old)==1
    source=source.replace(old,new)
    namespace={**n.__dict__,'Capture':Capture,'Replay':Replay}
    exec(compile(source,str(Path(__file__))+'::shared_activity_predict','exec'),namespace)
    original=n.predict; replacement=namespace['predict']
    def choose(torch,data,bb,ro,geom,bank,dev,control,folder):
        if ro.mode!='size': return original(torch,data,bb,ro,geom,bank,dev,control,folder)
        result=replacement(torch,data,bb,ro,geom,bank,dev,control,folder)
        n.write(folder/'shared_activity_check.json',dict(passed=True,requirement='exact equality of readout logits on identical backbone activity',
            weights_unchanged_verified_by_original_worker=True,recovery_source_sha256=n.sha(Path(__file__))))
        return result
    n.predict=choose
    return source

def worker():
    diagnostic=n.read(OUT/'size_recovery_diagnostic.json')
    assert diagnostic['finished'] and diagnostic['shared_exact'] and diagnostic['separate_pass_mismatch']
    install()
    return n.worker()

def controller():
    import msvcrt
    lock=(OUT/'controller.lock').open('a+b'); lock.seek(0)
    msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    assert n.now().isoformat()<n.SOFT and not (OUT/'stop.request').exists()
    assert not (OUT/'recovery2_controller_result.json').exists()
    assert n.read(OUT/'resume_controller_result.json')['own_worker_ended']
    diag=n.read(OUT/'size_recovery_diagnostic.json')
    assert diag['finished'] and diag['shared_exact'] and diag['separate_pass_mismatch']
    assert n.sha(OUT/'C/size/checkpoint.pt')==diag['checkpoint_sha256']
    source=install()
    n.write(OUT/'recovery2_protocol.json',dict(at=n.now().isoformat(),reason='Separate numerical core executions failed a readout-only invariant; same-activity outputs are exactly equal.',
        diagnostic='size_recovery_diagnostic.json',scientific_protocol_sha256=n.sha(OUT/'protocol.json'),
        service_sha256={str(p.relative_to(ROOT)):n.sha(p) for p in (Path(__file__),ROOT/'scripts/z13_size_check_recovery.py')},
        prediction_service_source=source,hard_deadline=n.HARD,save_deadline=n.SOFT,
        preserved={str(p.relative_to(OUT)):n.sha(p) for p in OUT.glob('*/*/completed.json')},
        checkpoint_sha256=diag['checkpoint_sha256'],training_repeated=False,
        change='Only diagnostic parent inference reuses the candidate backbone features. Original tolerance retained; additional exact-equality assertion.',
        previous_failure='resume_controller_result.json'))
    flags=subprocess.CREATE_NO_WINDOW|subprocess.IDLE_PRIORITY_CLASS
    with (OUT/'recovery2_stdout.log').open('a',encoding='utf-8') as log,(OUT/'recovery2_stderr.log').open('a',encoding='utf-8') as err:
        proc=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--worker'],cwd=ROOT,stdout=log,stderr=err,
            creationflags=flags,env={**os.environ,'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2'})
        n.write(OUT/'recovery2_controller.json',dict(pid=os.getpid(),worker_pid=proc.pid,started=n.now().isoformat(),deadline=n.HARD,save_deadline=n.SOFT))
        began=time.time(); stale=None; reason=None
        while proc.poll() is None:
            if n.now().isoformat()>=n.SOFT: (OUT/'stop.request').write_text('Save for author deadline',encoding='utf-8')
            status=OUT/'status.json'; age=time.time()-max(began,status.stat().st_mtime if status.exists() else began)
            if age>1200 and stale is None:
                stale=time.time(); (OUT/'stop.request').write_text('No progress; save',encoding='utf-8')
            from datetime import datetime
            if n.now()>=datetime.fromisoformat(n.HARD)-timedelta(seconds=30): reason='Absolute deadline'
            elif stale is not None and time.time()-stale>90: reason='Stall after saving grace'
            if reason:
                subprocess.run(['taskkill','/PID',str(proc.pid),'/T','/F'],creationflags=flags,check=False,timeout=15)
                proc.wait(timeout=10); break
            time.sleep(2)
        n.write(OUT/'recovery2_controller_result.json',dict(exit_code=proc.returncode,own_worker_ended=proc.poll() is not None,reason=reason,ended=n.now().isoformat()))

if __name__=='__main__':
    if '--worker' in sys.argv: raise SystemExit(worker())
    controller()
