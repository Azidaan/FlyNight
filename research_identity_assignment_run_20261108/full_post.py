"""Light, bounded observer and sequential dispatcher for registered final analyses."""
import os,sys,time,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_identity_assignment_run_20261108.common import *
FULL=RUN/'full_precision'


def main():
    import psutil
    verify_phase('full_post_registration.json');started=read(FULL/'manager_started.json');manager=None
    try:
        process=psutil.Process(started['pid'])
        if abs(process.create_time()-started['created'])<.01:manager=process
    except psutil.NoSuchProcess:pass
    while not (FULL/'rollouts_complete.json').exists():
        if manager is None or not manager.is_running():
            write(FULL/'post_stopped.json',dict(reason='Manager ended before complete rollout manifest; no statistical success inferred.',time=datetime.now(MSK).isoformat()),new=True);return
        if datetime.now(MSK)>=datetime(2026,11,9,7,39,30,tzinfo=MSK):return
        time.sleep(10)
    # The final video receipt can appear before the manager's last exit instructions.
    if manager is not None:
        try:manager.wait(timeout=10)
        except psutil.NoSuchProcess:pass
    verify_phase('full_post_registration.json');p=read(FULL/'protocol.json');sources=dict(read(RUN/'full_post_registration.json')['sources'])
    sources[(FULL/'rollouts_complete.json').relative_to(ROOT).as_posix()]=sha(FULL/'rollouts_complete.json')
    for arm in p['parameters']:
        for video in p['videos']:
            for path in [FULL/'metrics'/arm/(video+'.json'),FULL/'metrics'/arm/(video+'.details.json.gz'),FULL/'outputs'/arm/(video+'.json')]:sources[path.relative_to(ROOT).as_posix()]=sha(path)
    write(RUN/'full_statistics_phase.json',dict(registered=datetime.now(MSK).isoformat(),sources=sources,purpose='Previously fixed full-pilot inference; no parameter retuning.'),new=True)
    write(RUN/'full_additional_phase.json',dict(registered=datetime.now(MSK).isoformat(),sources=sources,purpose='Recover additional full-output metrics defined before rollout.'),new=True)
    from research_identity_assignment_run_20261108.full_precision_manager import launch
    launch('statistics','all','full_precision_statistics.py',[])
    launch('additional','all','full_additional.py',[])
    verify_phase('full_post_registration.json')
    write(FULL/'analyses_complete.json',dict(passed=True,statistics_sha256=sha(FULL/'comparison_report.json'),additional_sha256=sha(FULL/'additional_report.json'),
        time=datetime.now(MSK).isoformat(),scope='Full analyses finished; publication and independent source audit remain separate.'),new=True)
    status('full_pilot_analyses_complete',full_report=str(FULL/'comparison_report.json'),additional=str(FULL/'additional_report.json'))


if __name__=='__main__':main()
