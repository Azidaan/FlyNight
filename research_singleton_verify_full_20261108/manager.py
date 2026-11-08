"""Reuse the preserved sequential manager and exact metric-reuse machinery."""
import sys,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_singleton_verify_full_20261108.common import *


def launch(name,script,args):
    import psutil
    assert datetime.now(MSK)<datetime(2026,11,8,13,55,tzinfo=MSK)
    mapping={'full_technical.py':'technical.py','full_worker.py':'worker.py','full_evaluate.py':'evaluate.py','full_additional.py':'additional.py','full_statistics.py':'full_statistics.py'}
    filename=mapping[script];folder=FULL/'children';folder.mkdir(parents=True,exist_ok=True)
    with (folder/(name+'.log')).open('x',encoding='utf-8') as out,(folder/(name+'.err')).open('x',encoding='utf-8') as err:
        process=subprocess.Popen([sys.executable,'-u','research_singleton_verify_full_20261108/'+filename,*args],cwd=ROOT,stdout=out,stderr=err,
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
    paths,_=configure_paths()
    import research_singleton_free_20261108.full_manager as manager
    manager.FULL=FULL;manager.RUN=RUN;manager.phase=phase;manager.status=status;manager.launch=launch
    manager.raw_path=paths.raw_path;manager.output_path=paths.output_path;manager.source_arm=paths.source_arm
    manager.main()


if __name__=='__main__':main()
