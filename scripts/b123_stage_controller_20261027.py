"""Контроллер отложенной ступени: её бюджет отсчитывается с фактического запуска."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,time,subprocess,sys
import psutil

ROOT=Path(__file__).resolve().parents[1]


def main():
    a=argparse.ArgumentParser();a.add_argument('--folder',required=True);a.add_argument('--script',required=True);args=a.parse_args()
    folder=(ROOT/args.folder).resolve();folder.relative_to((ROOT/'runs/b123_20261027').resolve())
    p=json.loads((folder/'protocol.json').read_text(encoding='utf-8'))
    stop=min(time.time()+p['budget_seconds'],datetime.fromisoformat(p['author_stop']).timestamp())
    assert time.time()+5<stop
    def write(name,data):
        path=folder/name;temp=path.with_suffix('.next');temp.write_bytes((json.dumps(data,ensure_ascii=False,indent=2)+'\n').encode());temp.replace(path)
    write('execution_deadline.json',dict(started=datetime.now(timezone.utc).isoformat(),deadline=datetime.fromtimestamp(stop,timezone.utc).isoformat(),budget_seconds=p['budget_seconds']))
    write('controller_status.json',dict(state='starting'))
    owned={};timed_out=False
    with (folder/'stdout.log').open('w',encoding='utf-8') as out,(folder/'stderr.log').open('w',encoding='utf-8') as err:
        child=subprocess.Popen([sys.executable,'-u',str(ROOT/args.script),'--run'],cwd=ROOT,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
        root=psutil.Process(child.pid);root.nice(psutil.IDLE_PRIORITY_CLASS);owned[root.pid]=root.create_time()
        write('launch.json',dict(pid=root.pid,created=root.create_time()))
        while child.poll() is None:
            try:
                for q in root.children(recursive=True):owned[q.pid]=q.create_time()
            except psutil.NoSuchProcess:pass
            if time.time()>=stop-2:timed_out=True;break
            time.sleep(.5)
        for pid,created in reversed(list(owned.items())):
            try:
                q=psutil.Process(pid)
                if abs(q.create_time()-created)<.01 and q.is_running():q.kill()
            except psutil.NoSuchProcess:pass
        child.wait(timeout=5)
    alive=[]
    for pid,created in owned.items():
        try:
            q=psutil.Process(pid)
            if abs(q.create_time()-created)<.01 and q.is_running():alive.append(pid)
        except psutil.NoSuchProcess:pass
    write('controller_result.json',dict(exit_code=child.returncode,timed_out=timed_out,all_owned_ended=not alive,
          remaining=alive,owned=owned,updated=datetime.now(timezone.utc).isoformat()))
    return 0 if child.returncode==0 and not timed_out and not alive else 1


if __name__=='__main__':raise SystemExit(main())
