import os,sys,time,subprocess
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_identity_assignment_run_20261108.common import *
from research_singleton_verify_full_recovery1_20261108.status_io import for_run
RUN=ROOT/'runs/singleton_native_confirm_20261108';status=for_run(RUN)


def main():
    import psutil
    began=time.monotonic()
    while not (RUN/'metadata_deadline_registered.json').exists() and time.monotonic()-began<10:time.sleep(.1)
    assert (RUN/'metadata_deadline_registered.json').exists();assert read(RUN/'queue_result.json')['passed']
    stages={}
    for name in ['technical','native_trace']:
        assert datetime.now(MSK)<datetime(2026,11,8,13,55,tzinfo=MSK);status(name)
        with (RUN/(name+'.log')).open('x',encoding='utf-8') as out,(RUN/(name+'.err')).open('x',encoding='utf-8') as err:
            p=subprocess.Popen([sys.executable,'-u',f'research_singleton_native_confirm_20261108/{name}.py'],cwd=ROOT,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW|subprocess.IDLE_PRIORITY_CLASS)
            write(RUN/(name+'.started.json'),dict(pid=p.pid,created=psutil.Process(p.pid).create_time()),new=True);code=p.wait()
        lines=[s for s in (RUN/(name+'.log')).read_text(encoding='utf-8').splitlines() if s.startswith('SYSTEM_MEMORY_GUARD ')];assert lines,(name,code)
        report=json.loads(lines[-1].split(' ',1)[1]);ended=True
        for pid,birth in report['owned'].items():
            try:
                q=psutil.Process(int(pid))
                if abs(q.create_time()-birth)<.01:
                    try:q.wait(timeout=5)
                    except psutil.TimeoutExpired:ended=False
            except psutil.NoSuchProcess:pass
        stages[name]=dict(returncode=code,guardian=report,all_owned_ended=ended);write(RUN/(name+'.ended.json'),stages[name],new=True)
        assert code==0 and report['reason'] is None and report['returncode']==0 and ended,stages[name]
    write(RUN/'metadata_queue_result.json',dict(passed=True,stages=stages),new=True);status('native_metadata_complete')


if __name__=='__main__':main()
