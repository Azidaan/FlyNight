import subprocess
from research_singleton_working_domain_20261108.common import *


def main():
    import psutil
    phase()
    assert read(RUN/'authorization.json')['approved'] and read(RUN/'preflight.json')['passed']
    assert not (RUN/'execution_window.json').exists()
    now = datetime.now(MSK)
    write(RUN/'execution_window.json', dict(started=now.isoformat(),
        deadline=(now+timedelta(seconds=900)).isoformat(),
        soft_stop=(now+timedelta(seconds=870)).isoformat(),
        first_video_deadline=(now+timedelta(seconds=180)).isoformat()), new=True)
    status('starting_authorized_cpu_stage')
    def launch(script, stem):
        child = subprocess.Popen([sys.executable,'-u',script],cwd=ROOT,
            stdout=(RUN/(stem+'.log')).open('x',encoding='utf-8'),
            stderr=(RUN/(stem+'.err')).open('x',encoding='utf-8'),
            creationflags=subprocess.CREATE_NO_WINDOW|subprocess.IDLE_PRIORITY_CLASS)
        record=dict(pid=child.pid,created=psutil.Process(child.pid).create_time(),script=script)
        write(RUN/(stem+'.started.json'),record,new=True)
        return record
    controller=launch('research_singleton_working_domain_20261108/controller.py','controller')
    watcher=launch('research_singleton_working_domain_20261108/deadline_watch.py','deadline_watch')
    assert psutil.pid_exists(watcher['pid']) and psutil.pid_exists(controller['pid'])
    write(RUN/'armed.json',dict(controller=controller,watcher=watcher),new=True)
    print(dict(started=now.isoformat(),deadline=(now+timedelta(seconds=900)).isoformat(),controller=controller,watcher=watcher))


if __name__ == '__main__':
    main()
