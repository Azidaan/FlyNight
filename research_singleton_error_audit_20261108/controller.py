import sys,time,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_singleton_error_audit_20261108.common import *


def main():
    import psutil
    began=time.monotonic()
    while not (RUN/'deadline_registered.json').exists() and time.monotonic()-began<10:time.sleep(.1)
    assert (RUN/'deadline_registered.json').exists();prior=read(AGE/'started.json')
    while not (AGE/'queue_result.json').exists():
        p=psutil.Process(prior['pid']);assert abs(p.create_time()-prior['created'])<.01 and p.is_running(),'Recover the prior queue before diagnosis.'
        if datetime.now(MSK)>=datetime(2026,11,8,13,55,tzinfo=MSK):return
        time.sleep(10)
    try:
        p=psutil.Process(prior['pid'])
        if abs(p.create_time()-prior['created'])<.01:p.wait(timeout=15)
    except psutil.NoSuchProcess:pass
    assert read(AGE/'queue_result.json')['passed'] and read(OWN_FULL/'independent_audit.json')['passed']
    arms={a:OWN_FULL.relative_to(ROOT).as_posix() for a in ['SUPPORT','IMAGE','FLYVIS']}
    if (AGE/'full/independent_audit.json').exists():
        assert read(AGE/'full/independent_audit.json')['passed'];arms['AGE']=(AGE/'full').relative_to(ROOT).as_posix()
    write(RUN/'input_manifest.json',dict(arms=arms),new=True);sources={p.relative_to(ROOT).as_posix():sha(p) for p in Path(__file__).parent.glob('*.py')}
    names=['runs/singleton_error_audit_20261108/protocol.json','runs/singleton_error_audit_20261108/input_manifest.json','research_singleton_age_20261108/common.py',
        'research_singleton_verify_20261108/common.py','research_identity_assignment_run_20261108/common.py','research_singleton_free_20261108/full_audit.py','datasets/bdd100k_loader.py']
    for path in [OWN_FULL/'protocol.json',OWN_FULL/'independent_audit.json',AGE/'queue_result.json']:names.append(path.relative_to(ROOT).as_posix())
    for video,entry in read(OWN_FULL/'protocol.json')['videos'].items():
        names.extend([entry['detections'],f'data/bdd100k/labels/box_track_20/val/{video}.json',(OWN_FULL/'runtime_inputs'/(video+'.json')).relative_to(ROOT).as_posix()])
        for arm,folder in arms.items():
            for path in [ROOT/folder/'rollouts'/arm/(video+'.npz'),ROOT/folder/'rollouts'/arm/(video+'.events.json.gz'),ROOT/folder/'rollouts'/arm/(video+'.receipt.json'),ROOT/folder/'additional'/arm/(video+'.json')]:names.append(path.relative_to(ROOT).as_posix())
    sources.update({n:sha(ROOT/n) for n in names});write(RUN/'phase.json',dict(registered=datetime.now(MSK).isoformat(),sources=sources,read_only=True),new=True)
    status('direct_recovery_diagnosis_starting')
    with (RUN/'worker.log').open('x',encoding='utf-8') as out,(RUN/'worker.err').open('x',encoding='utf-8') as err:
        child=subprocess.Popen([sys.executable,'-u','research_singleton_error_audit_20261108/worker.py'],cwd=ROOT,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW|subprocess.IDLE_PRIORITY_CLASS)
        write(RUN/'worker_started.json',dict(pid=child.pid,created=psutil.Process(child.pid).create_time()),new=True);code=child.wait()
    lines=[s for s in (RUN/'worker.log').read_text(encoding='utf-8').splitlines() if s.startswith('SYSTEM_MEMORY_GUARD ')];assert lines,(code,'Missing guardian completion');r=json.loads(lines[-1].split(' ',1)[1]);ended=True
    for pid,birth in r['owned'].items():
        try:
            p=psutil.Process(int(pid))
            if abs(p.create_time()-birth)<.01:
                try:p.wait(timeout=5)
                except psutil.TimeoutExpired:ended=False
        except psutil.NoSuchProcess:pass
    write(RUN/'worker_ended.json',dict(returncode=code,guardian=r,all_owned_ended=ended),new=True);assert code==0 and r['returncode']==0 and r['reason'] is None and ended
    assert read(RUN/'report.json')['passed'];write(RUN/'queue_result.json',dict(passed=True,report_sha256=sha(RUN/'report.json')),new=True);status('direct_recovery_diagnosis_complete')


if __name__=='__main__':main()
