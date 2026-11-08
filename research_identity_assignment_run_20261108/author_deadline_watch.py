"""Independent author-window stop for explicitly registered task roots only."""
import os,sys,time,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_identity_assignment_run_20261108.common import *


def same_process(pid,birth):
    import psutil
    try:
        p=psutil.Process(int(pid))
        return p if abs(p.create_time()-birth)<.01 else None
    except psutil.NoSuchProcess:return None


def main():
    import psutil
    protocol=read(RUN/'author_window_20261108.json');deadline=datetime.fromisoformat(protocol['deadline']);owned={}
    while True:
        roots=read(RUN/'deadline_roots.json')['roots']
        for item in roots:
            p=same_process(item['pid'],item['created'])
            if p is None:continue
            argv=p.cmdline();assert any(item['script'] in a.replace('\\','/') for a in argv),(p.pid,'Root script mismatch')
            for process in [p,*p.children(recursive=True)]:
                try:owned[process.pid]=process.create_time()
                except psutil.NoSuchProcess:pass
        now=datetime.now(MSK)
        if now>=deadline-timedelta(minutes=5) and not (RUN/'author_stop_requested.json').exists():
            write(RUN/'author_stop_requested.json',dict(time=now.isoformat(),deadline=deadline.isoformat(),reason='Author window ends at14:00MSK.'),new=True)
        live={str(pid):birth for pid,birth in owned.items() if same_process(pid,birth) is not None}
        write(RUN/'author_deadline_status.json',dict(updated=now.isoformat(),deadline=deadline.isoformat(),own_processes=live,watch_pid=os.getpid()))
        if now>=deadline-timedelta(seconds=30):
            killed=[]
            for pid,birth in reversed(list(owned.items())):
                process=same_process(pid,birth)
                if process is not None:
                    try:process.kill();killed.append(pid)
                    except psutil.NoSuchProcess:pass
            remaining=[]
            for pid,birth in owned.items():
                process=same_process(pid,birth)
                if process is not None:
                    try:process.wait(timeout=5)
                    except psutil.TimeoutExpired:remaining.append(pid)
            write(RUN/'author_deadline_result.json',dict(time=datetime.now(MSK).isoformat(),deadline=deadline.isoformat(),killed=killed,
                remaining=remaining,all_owned_ended=not remaining,scope='Only PID+creation-time verified descendants of explicitly registered task roots.'),new=True)
            assert not remaining;return
        time.sleep(1)


if __name__=='__main__':main()
