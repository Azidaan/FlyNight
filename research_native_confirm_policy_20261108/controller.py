import os,sys,time,subprocess
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_native_confirm_policy_20261108.common import *


def run(name,script):
    import psutil
    assert datetime.now(MSK)<datetime(2026,11,8,13,55,tzinfo=MSK);folder=RUN/'children';folder.mkdir(exist_ok=True);status(name)
    with (folder/(name+'.log')).open('x',encoding='utf-8') as out,(folder/(name+'.err')).open('x',encoding='utf-8') as err:
        p=subprocess.Popen([sys.executable,'-u',script],cwd=ROOT,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW|subprocess.IDLE_PRIORITY_CLASS)
        write(folder/(name+'.started.json'),dict(pid=p.pid,created=psutil.Process(p.pid).create_time()),new=True);code=p.wait()
    lines=[s for s in (folder/(name+'.log')).read_text(encoding='utf-8').splitlines() if s.startswith('SYSTEM_MEMORY_GUARD ')];assert lines,(name,code)
    report=json.loads(lines[-1].split(' ',1)[1]);ended=True
    for pid,birth in report['owned'].items():
        try:
            q=psutil.Process(int(pid))
            if abs(q.create_time()-birth)<.01:
                try:q.wait(timeout=5)
                except psutil.TimeoutExpired:ended=False
        except psutil.NoSuchProcess:pass
    write(folder/(name+'.ended.json'),dict(returncode=code,guardian=report,all_owned_ended=ended),new=True)
    assert code==0 and report['returncode']==0 and report['reason'] is None and ended,(name,code,report)


def wait_registration(name):
    while not (RUN/name).exists():
        if datetime.now(MSK)>=datetime(2026,11,8,13,55,tzinfo=MSK):return None
        time.sleep(5)
    p=read(RUN/name)
    for n,h in p['sources'].items():assert sha(ROOT/n)==h,n
    return p


def main():
    began=time.monotonic()
    while not (RUN/'deadline_registered.json').exists() and time.monotonic()-began<10:time.sleep(.1)
    assert (RUN/'deadline_registered.json').exists()
    for name in ['catalog_check','fit_geometry','recover_evaluate26']:run(name,f'research_native_confirm_policy_20261108/{name}.py')
    write(RUN/'metadata_stage_complete.json',dict(passed=True),new=True);status('waiting_for_registered_spatial_data')
    p=wait_registration('data_registration.json')
    if p is None:return
    for entry in p['stages']:run(entry['name'],entry['script'])
    write(RUN/'data_stage_complete.json',dict(passed=True),new=True);status('waiting_for_registered_comparison')
    p=wait_registration('comparison_registration.json')
    if p is None:return
    for entry in p['stages']:run(entry['name'],entry['script'])
    write(RUN/'comparison_stage_complete.json',dict(passed=True),new=True);status('causal_link_comparison_complete')


if __name__=='__main__':main()
