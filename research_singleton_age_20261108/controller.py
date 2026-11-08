"""Wait for the running audit, then execute the registered control sequentially."""
import os,sys,time,subprocess,shutil
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_singleton_age_20261108.common import *


def run_stage(name,script):
    import psutil
    assert datetime.now(MSK)<datetime(2026,11,8,13,55,tzinfo=MSK)
    folder=RUN/'children';folder.mkdir(parents=True,exist_ok=True)
    with (folder/(name+'.log')).open('x',encoding='utf-8') as out,(folder/(name+'.err')).open('x',encoding='utf-8') as err:
        process=subprocess.Popen([sys.executable,'-u','research_singleton_age_20261108/'+script],cwd=ROOT,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW|subprocess.IDLE_PRIORITY_CLASS)
        write(folder/(name+'.started.json'),dict(pid=process.pid,created=psutil.Process(process.pid).create_time()),new=True);code=process.wait()
    lines=[s for s in (folder/(name+'.log')).read_text(encoding='utf-8').splitlines() if s.startswith('SYSTEM_MEMORY_GUARD ')];assert lines,(name,code,'No memory guardian completion')
    report=json.loads(lines[-1].split(' ',1)[1]);ended=True
    for pid,birth in report['owned'].items():
        try:
            p=psutil.Process(int(pid))
            if abs(p.create_time()-birth)<.01:
                try:p.wait(timeout=5)
                except psutil.TimeoutExpired:ended=False
        except psutil.NoSuchProcess:pass
    write(folder/(name+'.ended.json'),dict(returncode=code,guardian=report,all_owned_ended=ended),new=True)
    assert code==0 and report['returncode']==0 and report['reason'] is None and ended,(name,code,report)


def freeze_comparison():
    sources=read(OWN/'full_phase.json')['sources'].copy()
    for path in Path(__file__).parent.glob('*.py'):sources[path.relative_to(ROOT).as_posix()]=sha(path)
    for path in [RUN/'protocol.json',OWN/'calibration_rows.json',OWN/'comparison_report.json',OWN/'independent_audit.json',OWN/'scores_complete.json',OWN_FULL/'protocol.json',OWN_FULL/'comparison_report.json',OWN_FULL/'independent_audit.json',OWN/'full_audit_queue_result.json']:
        sources[path.relative_to(ROOT).as_posix()]=sha(path)
    write(RUN/'comparison_phase.json',dict(registered=datetime.now(MSK).isoformat(),sources=sources,full_execution_ended=True),new=True)


def register_full():
    full=read(OWN_FULL/'protocol.json');full=dict(full,id='singleton_recency_control_full_v1_20261108',registered=datetime.now(MSK).isoformat(),recency_cap=read(RUN/'selection.json')['cap'],parent_protocol_sha256=sha(OWN_FULL/'protocol.json'))
    full['parameters']=dict(full['parameters'],AGE=dict(alpha=.2,threshold=-full['recency_cap']));full['primary_comparisons']=read(RUN/'protocol.json')['full_comparisons'];full['aliases']={}
    full.pop('cpu_arms',None);full.pop('gpu_arms',None);full.pop('selected_system',None)
    write(FULL/'protocol.json',full,new=True)
    for video in full['videos']:
        path=FULL/'runtime_inputs'/(video+'.json');path.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(OWN_FULL/'runtime_inputs'/(video+'.json'),path)
    sources=read(RUN/'comparison_phase.json')['sources'].copy()
    for path in [RUN/'selection.json',RUN/'comparison_report.json',RUN/'independent_audit.json',FULL/'protocol.json',OWN_FULL/'additional_report.json',OWN_FULL/'bootstrap_counts.npz']:
        sources[path.relative_to(ROOT).as_posix()]=sha(path)
    for path in (FULL/'runtime_inputs').glob('*.json'):sources[path.relative_to(ROOT).as_posix()]=sha(path)
    for path in Path(ROOT/'.venv/Lib/site-packages/ultralytics/trackers').rglob('*.py'):
        if path.name in ['byte_tracker.py','bot_sort.py','matching.py','stracks.py']:sources[path.relative_to(ROOT).as_posix()]=sha(path)
    write(RUN/'full_phase.json',dict(registered=datetime.now(MSK).isoformat(),sources=sources),new=True)


def main():
    import psutil
    began=time.monotonic()
    while not (RUN/'deadline_registered.json').exists() and time.monotonic()-began<10:time.sleep(.1)
    assert (RUN/'deadline_registered.json').exists();phase('registration_phase.json')
    prior=read(OWN/'full_audit_wait_started.json')
    while not (OWN/'full_audit_queue_result.json').exists():
        try:
            p=psutil.Process(prior['pid']);assert abs(p.create_time()-prior['created'])<.01 and p.is_running()
        except psutil.NoSuchProcess:raise RuntimeError('Prior full/audit controller ended without a passed result; preserve it and recover before this control.')
        if datetime.now(MSK)>=datetime(2026,11,8,13,55,tzinfo=MSK):return
        time.sleep(10)
    assert read(OWN/'full_audit_queue_result.json')['passed']
    try:
        p=psutil.Process(prior['pid'])
        if abs(p.create_time()-prior['created'])<.01:p.wait(timeout=15)
    except psutil.NoSuchProcess:pass
    freeze_comparison();status('recency_catalog_check_before_fit');run_stage('catalog','catalog_check.py');run_stage('conditional','conditional.py');run_stage('conditional_audit','conditional_audit.py')
    if read(RUN/'comparison_report.json')['full_gate']['proceed']:
        register_full();run_stage('full_worker','worker.py');run_stage('full_statistics','full_statistics.py');run_stage('full_audit','full_audit.py')
        assert read(FULL/'independent_audit.json')['passed']
    write(RUN/'queue_result.json',dict(passed=True,conditional_audit_sha256=sha(RUN/'independent_audit.json'),full_done=(FULL/'independent_audit.json').exists()),new=True)
    status('recency_control_complete',full_done=(FULL/'independent_audit.json').exists())


if __name__=='__main__':main()
