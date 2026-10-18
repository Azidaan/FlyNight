"""Independent four-hour CPU controller; no scheduled model/chat calls."""
import json
import subprocess
import sys
import time
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.z13_target_center_dev import RUN,read,write,verify


def main():
    import psutil
    own={p.pid for p in psutil.Process().parents()}|{psutil.Process().pid}
    names={'z13_target_center_dev.py',Path(__file__).name}
    for p in psutil.process_iter(['cmdline']):
        if p.pid not in own and any(Path(x).name in names for x in (p.info['cmdline'] or [])):
            raise RuntimeError(f'Duplicate worker/controller {p.pid}')
    if (RUN/'controller.json').exists() or (RUN/'report.json').exists():
        raise RuntimeError('Inspect existing launch; no automatic duplicate')
    protocol=read(RUN/'protocol.json');verify(protocol)
    start=time.time();soft=start+protocol['budget_seconds'];hard=soft+30
    command=[sys.executable,'-u',str(ROOT/'scripts/z13_target_center_dev.py'),'--deadline',str(soft)]
    with (RUN/'stdout.log').open('w',encoding='utf-8') as out,(RUN/'stderr.log').open('w',encoding='utf-8') as err:
        child=subprocess.Popen(command,cwd=ROOT,stdout=out,stderr=err,creationflags=subprocess.IDLE_PRIORITY_CLASS)
        write(RUN/'controller.json',dict(pid=child.pid,command=command,started_at=datetime.fromtimestamp(start,timezone.utc).isoformat(),
            soft_deadline=datetime.fromtimestamp(soft,timezone.utc).isoformat(),hard_deadline=datetime.fromtimestamp(hard,timezone.utc).isoformat(),
            budget_seconds=protocol['budget_seconds'],device='cpu',chat_automation=False))
        timeout=False
        try: code=child.wait(timeout=max(1,hard-time.time()))
        except subprocess.TimeoutExpired:
            timeout=True
            subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],check=True,capture_output=True)
            code=child.wait(timeout=20)
        write(RUN/'controller_result.json',dict(exit_code=code,hard_timeout=timeout,worker_ended=True,seconds=time.time()-start))
        if timeout:write(RUN/'status.json',dict(state='paused',stage='Deadline; own worker ended; completed videos saved'))
        # Persist a completion inventory without making scientific claims about partial work.
        write(RUN/'completion_inventory.json',dict(report_exists=(RUN/'report.json').exists(),
            extracted_videos=len(list((RUN/'data').glob('*.npz'))),
            completed={a:len(list((RUN/a).glob('*/done.json'))) for a in ('D','C','R1')},
            controller_exit_code=code))


if __name__=='__main__':main()
