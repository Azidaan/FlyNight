import os,sys,time,subprocess
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_native_confirm_risk_20261108.common import *


def run(name,script):
    import psutil
    assert datetime.now(MSK)<datetime(2026,11,8,13,55,tzinfo=MSK);folder=RUN/'children';folder.mkdir(exist_ok=True);status(name)
    with (folder/(name+'.log')).open('x',encoding='utf-8') as out,(folder/(name+'.err')).open('x',encoding='utf-8') as err:
        child=subprocess.Popen([sys.executable,'-u',script],cwd=ROOT,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW|subprocess.IDLE_PRIORITY_CLASS)
        write(folder/(name+'.started.json'),dict(pid=child.pid,created=psutil.Process(child.pid).create_time()),new=True);code=child.wait()
    lines=[s for s in (folder/(name+'.log')).read_text(encoding='utf-8').splitlines() if s.startswith('SYSTEM_MEMORY_GUARD ')];assert lines,(name,code)
    guardian=json.loads(lines[-1].split(' ',1)[1]);ended=True
    for pid,birth in guardian['owned'].items():
        try:
            p=psutil.Process(int(pid))
            if abs(p.create_time()-birth)<.01:
                try:p.wait(timeout=5)
                except psutil.TimeoutExpired:ended=False
        except psutil.NoSuchProcess:pass
    write(folder/(name+'.ended.json'),dict(returncode=code,guardian=guardian,all_owned_ended=ended),new=True);assert code==0 and guardian['returncode']==0 and guardian['reason'] is None and ended,(name,code,guardian)


def main():
    began=time.monotonic()
    while not (RUN/'deadline_registered.json').exists() and time.monotonic()-began<10:time.sleep(.1)
    assert (RUN/'deadline_registered.json').exists();phase('phase.json')
    for name in ['catalog_check','compare','audit']:run(name,f'research_native_confirm_risk_20261108/{name}.py')
    write(RUN/'queue_result.json',dict(passed=True,comparison_sha256=sha(RUN/'comparison_report.json'),independent_audit_sha256=sha(RUN/'independent_audit.json')),new=True);status('empirical_error_gate_complete')


if __name__=='__main__':main()
