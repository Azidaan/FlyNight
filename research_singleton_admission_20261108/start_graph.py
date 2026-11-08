import sys,time,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_native_confirm_policy_20261108.common import *


def main():
    import psutil
    folder=ROOT/'runs/singleton_admission_20261108';pdir='research_singleton_admission_20261108'
    with (folder/'graph_controller.log').open('x',encoding='utf-8') as out,(folder/'graph_controller.err').open('x',encoding='utf-8') as err:p=subprocess.Popen([sys.executable,'-u',pdir+'/graph_controller.py'],cwd=ROOT,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW|subprocess.IDLE_PRIORITY_CLASS)
    root=dict(pid=p.pid,created=psutil.Process(p.pid).create_time(),script=pdir+'/graph_controller.py');write(folder/'graph_started.json',root,new=True);roots=read(BASE/'deadline_roots.json');roots['roots'].append(root)
    for attempt in range(10):
        try:write(BASE/'deadline_roots.json',roots);break
        except PermissionError:
            if attempt==9:raise
            time.sleep(.1)
    print(root)


if __name__=='__main__':main()
