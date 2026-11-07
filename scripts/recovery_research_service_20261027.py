"""Новая ограниченная очередь и независимая остановка; старую ночную очередь не меняет."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, json, subprocess, sys, time
import psutil

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/recovery_research_20261027'

def read(p):
    for i in range(10):
        try:return json.loads(Path(p).read_text(encoding='utf-8-sig'))
        except (PermissionError,json.JSONDecodeError):
            if i==9:raise
            time.sleep(.1)

def write(p,data,required=True):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_name(p.name+'.next')
    for i in range(10):
        try:
            temp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');temp.replace(p);return True
        except PermissionError:
            if i==9 and required:raise
            time.sleep(.1)
    return False

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def matching(pid,birth):
    try:
        p=psutil.Process(int(pid))
        return p if p.is_running() and abs(p.create_time()-float(birth))<.01 else None
    except psutil.NoSuchProcess:return None

def register(p,folder):
    # Единственный писатель реестра — очередь; guard его только читает.
    data=read(RUN/'registry.json');data['processes'].append(dict(pid=p.pid,created=p.create_time(),folder=str(folder.relative_to(ROOT))))
    write(RUN/'registry.json',data)

def launch_process(arguments,folder,name):
    folder.mkdir(parents=True,exist_ok=True)
    with (folder/(name+'.stdout.log')).open('w',encoding='utf-8') as out,(folder/(name+'.stderr.log')).open('w',encoding='utf-8') as err:
        child=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),*arguments],cwd=ROOT,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
    p=psutil.Process(child.pid);p.nice(psutil.IDLE_PRIORITY_CLASS)
    proof=dict(pid=p.pid,created=p.create_time(),source_sha256=sha(__file__),launched=now())
    write(folder/(name+'.launch.json'),proof)
    return p

def stage(folder,script):
    folder=(ROOT/folder).resolve();folder.relative_to(RUN.resolve())
    protocol=read(folder/'protocol.json')
    end=min(time.time()+protocol['budget_seconds'],datetime.fromisoformat(read(RUN/'authorization.json')['save_at']).timestamp())
    assert end>time.time()+10
    write(folder/'execution_deadline.json',dict(started=now(),deadline=datetime.fromtimestamp(end,timezone.utc).isoformat()))
    write(folder/'status.json',dict(state='starting',updated=now()))
    owned={};timed_out=False
    with (folder/'stdout.log').open('w',encoding='utf-8') as out,(folder/'stderr.log').open('w',encoding='utf-8') as err:
        child=subprocess.Popen([sys.executable,'-u',str(ROOT/script),'--run'],cwd=ROOT,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
        p=psutil.Process(child.pid);owned[p.pid]=p.create_time()
        write(folder/'launch.json',dict(pid=p.pid,created=p.create_time()))
        while child.poll() is None:
            try:
                for q in p.children(recursive=True):owned[q.pid]=q.create_time()
            except psutil.NoSuchProcess:pass
            if time.time()>=end-2 or (folder/'stop.request').exists():timed_out=True;break
            time.sleep(.5)
        for pid,birth in reversed(list(owned.items())):
            q=matching(pid,birth)
            if q:
                try:q.kill()
                except psutil.NoSuchProcess:pass
        child.wait(timeout=10)
    remaining=[pid for pid,birth in owned.items() if matching(pid,birth)]
    write(folder/'controller_result.json',dict(exit_code=child.returncode,timed_out=timed_out,owned=owned,remaining=remaining,all_owned_ended=not remaining,updated=now()))
    return int(child.returncode!=0 or timed_out or bool(remaining))

def queue():
    end=datetime.fromisoformat(read(RUN/'authorization.json')['save_at']).timestamp()
    while time.time()<end:
        jobs=[p for p in sorted((RUN/'ready_jobs').glob('*.json')) if not (RUN/'jobs_done'/p.name).exists()]
        if not jobs:
            write(RUN/'queue_status.json',dict(state='waiting_for_agent',updated=now()),required=False);time.sleep(10);continue
        path=jobs[0];job=read(path);folder=(ROOT/job['folder']).resolve();folder.relative_to(RUN.resolve())
        assert sha(ROOT/job['script'])==job['script_sha256'] and sha(folder/'protocol.json')==job['protocol_sha256']
        write(RUN/'queue_status.json',dict(state='running',job=path.name,updated=now()))
        process=launch_process(['--stage','--folder',job['folder'],'--script',job['script']],folder,'controller')
        register(process,folder)
        while matching(process.pid,process.create_time()):time.sleep(.5)
        result=read(folder/'controller_result.json')
        assert result['all_owned_ended']
        write(RUN/'jobs_done'/path.name,dict(job=job,result=result,finished=now()))
        if result['exit_code']!=0 or result['timed_out']:
            write(RUN/'agent_attention.json',dict(job=path.name,result=result,updated=now()))
            # Независимые готовые этапы продолжаются; агент получает первичную ошибку.
    write(RUN/'queue_status.json',dict(state='stopped_at_save_time',updated=now()))

def guard():
    auth=read(RUN/'authorization.json');save=datetime.fromisoformat(auth['save_at']).timestamp();stop=datetime.fromisoformat(auth['controller_stop']).timestamp()
    owned={};folders=set()
    while time.time()<stop:
        for item in read(RUN/'registry.json')['processes']:
            owned[item['pid']]=item['created'];p=matching(item['pid'],item['created'])
            if p:
                try:
                    for q in p.children(recursive=True):owned[q.pid]=q.create_time()
                except psutil.NoSuchProcess:pass
            if time.time()>=save:
                folder=(ROOT/item['folder']).resolve();folder.relative_to(RUN.resolve());folders.add(folder)
        for folder in folders:
            try:(folder/'stop.request').write_text('Author deadline: save and end.\n',encoding='utf-8')
            except PermissionError:pass
        write(RUN/'deadline_status.json',dict(state='watching',owned=owned,updated=now(),stop_at=auth['controller_stop']),required=False)
        time.sleep(1)
    for pid,birth in reversed(list(owned.items())):
        p=matching(pid,birth)
        if p:
            try:p.kill()
            except psutil.NoSuchProcess:pass
    until=time.time()+15
    while time.time()<until and any(matching(p,b) for p,b in owned.items()):time.sleep(.2)
    live=[p for p,b in owned.items() if matching(p,b)]
    write(RUN/'deadline_result.json',dict(all_owned_ended=not live,remaining=live,owned=owned,updated=now()))

def launch():
    assert not (RUN/'queue.launch.json').exists() and not (RUN/'guard.launch.json').exists()
    q=launch_process(['--queue'],RUN,'queue');register(q,RUN)
    g=launch_process(['--guard'],RUN,'guard')
    assert matching(q.pid,q.create_time()) and matching(g.pid,g.create_time())
    print(json.dumps(dict(queue_pid=q.pid,queue_birth=q.create_time(),guard_pid=g.pid,guard_birth=g.create_time())))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--stage',action='store_true');a.add_argument('--folder');a.add_argument('--script');a.add_argument('--queue',action='store_true');a.add_argument('--guard',action='store_true');a.add_argument('--launch',action='store_true');p=a.parse_args()
    if p.stage:raise SystemExit(stage(p.folder,p.script))
    elif p.guard:guard()
    elif p.queue:queue()
    elif p.launch:launch()
