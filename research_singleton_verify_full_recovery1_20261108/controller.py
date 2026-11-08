"""Resume only missing results, then restore both dependent queues in order."""
import sys,time,subprocess,importlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_singleton_verify_full_recovery1_20261108.common import *
from research_singleton_verify_full_recovery1_20261108.status_io import for_run


def launch(name,group,stage,args=()):
    import psutil
    assert datetime.now(MSK)<datetime(2026,11,8,13,55,tzinfo=MSK);recovery_phase();folder=RECOVERY/'children';folder.mkdir(parents=True,exist_ok=True)
    command=[sys.executable,'-u','research_singleton_verify_full_recovery1_20261108/'+('worker.py' if group=='worker' else 'stage.py')]
    command.extend(args if group=='worker' else [group,stage,*args])
    with (folder/(name+'.log')).open('x',encoding='utf-8') as out,(folder/(name+'.err')).open('x',encoding='utf-8') as err:
        p=subprocess.Popen(command,cwd=ROOT,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW|subprocess.IDLE_PRIORITY_CLASS)
        write(folder/(name+'.started.json'),dict(pid=p.pid,created=psutil.Process(p.pid).create_time(),command=command),new=True);code=p.wait()
    lines=[s for s in (folder/(name+'.log')).read_text(encoding='utf-8').splitlines() if s.startswith('SYSTEM_MEMORY_GUARD ')];assert lines,(name,code,'No memory guard completion');r=json.loads(lines[-1].split(' ',1)[1]);ended=True
    for pid,birth in r['owned'].items():
        try:
            p=psutil.Process(int(pid))
            if abs(p.create_time()-birth)<.01:
                try:p.wait(timeout=5)
                except psutil.TimeoutExpired:ended=False
        except psutil.NoSuchProcess:pass
    write(folder/(name+'.ended.json'),dict(returncode=code,guardian=r,all_owned_ended=ended),new=True)
    assert code==0 and r['returncode']==0 and r['reason'] is None and ended,(name,code,r)


def full():
    paths,_=configure_paths();import research_singleton_free_20261108.full_manager as manager
    manager.FULL=FULL;manager.RUN=RUN;manager.phase=phase;manager.status=for_run(RUN)
    manager.raw_path=paths.raw_path;manager.output_path=paths.output_path;manager.source_arm=paths.source_arm
    mapping={'full_technical.py':'technical','full_evaluate.py':'evaluate','full_additional.py':'additional','full_statistics.py':'full_statistics'}
    def dispatch(name,script,args):
        if script=='full_worker.py':launch('full_'+name,'worker','worker',args)
        else:launch('full_'+name,'own',mapping[script],args)
    manager.launch=dispatch;manager.main()


def audit_full():
    sources=read(RUN/'full_phase.json')['sources'].copy()
    for path in [ROOT/'research_singleton_verify_full_audit_20261108/audit.py',ROOT/'research_singleton_free_20261108/full_audit.py',FULL/'comparison_report.json',FULL/'additional_report.json',FULL/'analyses_complete.json',RECOVERY/'phase.json']:
        sources[path.relative_to(ROOT).as_posix()]=sha(path)
    for name in ['trackers/byte_tracker.py','trackers/bot_sort.py','trackers/utils/stracks.py','trackers/utils/matching.py','utils/ops.py','utils/metrics.py']:
        path=ROOT/'.venv/Lib/site-packages/ultralytics'/name;sources[path.relative_to(ROOT).as_posix()]=sha(path)
    write(RUN/'full_audit_phase.json',dict(registered=datetime.now(MSK).isoformat(),sources=sources,read_only=True,recovery_dispatcher='full_recovery1'),new=True)
    launch('full_audit','own_audit','audit');assert read(FULL/'independent_audit.json')['passed']
    write(RUN/'full_audit_queue_result.json',dict(passed=True,audit_sha256=sha(FULL/'independent_audit.json'),recovered_dispatcher=True,
        original_observer_failure='runs/singleton_verify_20261108/full_audit_wait.err',recovery_phase_sha256=sha(RECOVERY/'phase.json')),new=True)


def subsequent():
    import research_singleton_age_20261108.controller as age
    age.run_stage=lambda name,script:launch('age_'+name,'age',script.removesuffix('.py'))
    age.status=for_run(ROOT/'runs/singleton_age_20261108');age.main()
    # The original diagnostic dispatcher is preserved; route just its worker
    # command through this recovery's guarded service-status entrypoint.
    import research_singleton_error_audit_20261108.controller as diagnostic
    original=diagnostic.subprocess.Popen
    def patched(command,*args,**kwargs):
        command=list(command)
        if any(str(v).replace('\\','/').endswith('research_singleton_error_audit_20261108/worker.py') for v in command):
            command=[sys.executable,'-u','research_singleton_verify_full_recovery1_20261108/stage.py','diagnosis','worker']
        return original(command,*args,**kwargs)
    diagnostic.subprocess.Popen=patched;diagnostic.status=for_run(ROOT/'runs/singleton_error_audit_20261108')
    try:diagnostic.main()
    finally:diagnostic.subprocess.Popen=original


def main():
    began=time.monotonic()
    while not (RECOVERY/'deadline_registered.json').exists() and time.monotonic()-began<10:time.sleep(.1)
    assert (RECOVERY/'deadline_registered.json').exists();recovery_phase();full();audit_full();subsequent();recovery_phase()
    write(RECOVERY/'queue_result.json',dict(passed=True,full_audit_sha256=sha(FULL/'independent_audit.json'),age_result_sha256=sha(ROOT/'runs/singleton_age_20261108/queue_result.json'),
        diagnosis_result_sha256=sha(ROOT/'runs/singleton_error_audit_20261108/queue_result.json')),new=True)


if __name__=='__main__':main()
