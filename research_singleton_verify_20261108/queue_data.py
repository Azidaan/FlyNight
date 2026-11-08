"""Sequential data recovery, then a separately frozen comparison stage."""
import sys,os,time,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_singleton_verify_20261108.common import *


def launch(name,script):
    import psutil
    assert datetime.now(MSK)<datetime(2026,11,8,13,55,tzinfo=MSK)
    folder=RUN/'children';folder.mkdir(exist_ok=True)
    with (folder/(name+'.log')).open('x',encoding='utf-8') as out,(folder/(name+'.err')).open('x',encoding='utf-8') as err:
        process=subprocess.Popen([sys.executable,'-u','research_singleton_verify_20261108/'+script],cwd=ROOT,stdout=out,stderr=err,
            creationflags=subprocess.CREATE_NO_WINDOW|subprocess.IDLE_PRIORITY_CLASS)
        write(folder/(name+'.started.json'),dict(pid=process.pid,created=psutil.Process(process.pid).create_time()),new=True);code=process.wait()
    lines=[s for s in (folder/(name+'.log')).read_text(encoding='utf-8').splitlines() if s.startswith('SYSTEM_MEMORY_GUARD ')]
    assert lines,(name,code);report=json.loads(lines[-1].split(' ',1)[1]);ended=True
    for pid,birth in report['owned'].items():
        try:
            p=psutil.Process(int(pid))
            if abs(p.create_time()-birth)<.01:
                try:p.wait(timeout=5)
                except psutil.TimeoutExpired:ended=False
        except psutil.NoSuchProcess:pass
    write(folder/(name+'.ended.json'),dict(returncode=code,guardian=report,all_owned_ended=ended),new=True)
    assert code==0 and report['returncode']==0 and report['reason'] is None and ended,(name,code,report)


def main():
    began=time.monotonic()
    while not (RUN/'deadline_registered.json').exists() and time.monotonic()-began<10:time.sleep(.1)
    assert (RUN/'deadline_registered.json').exists();phase('input_phase.json')
    launch('catalog','catalog_check.py');launch('inputs','prepare_inputs.py')
    source=read(RUN/'input_phase.json')['sources'].copy();source.update(read(BASE/'feature_phase.json')['sources'])
    for path in [ROOT/'research_singleton_verify_20261108/extract_features.py',RUN/'input_complete.json']:
        source[path.relative_to(ROOT).as_posix()]=sha(path)
    write(RUN/'feature_phase.json',dict(registered=datetime.now(MSK).isoformat(),sources=source),new=True)
    launch('features','extract_features.py')
    write(RUN/'data_complete.json',dict(passed=True,features_sha256=sha(RUN/'features_complete.json')),new=True)
    status('own_maps_ready_for_registered_comparison')
    while not (RUN/'comparison_registration.json').exists():
        if datetime.now(MSK)>=datetime(2026,11,8,13,55,tzinfo=MSK):return
        time.sleep(10)
    phase('comparison_phase.json');launch('comparison','compare.py')
    write(RUN/'data_queue_result.json',dict(passed=True,comparison_sha256=sha(RUN/'comparison_report.json')),new=True)


if __name__=='__main__':main()
