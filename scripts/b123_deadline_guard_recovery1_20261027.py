"""Тот же независимый срок; временный отказ записи статуса не снимает защиту."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,time,os,sys,subprocess
import psutil
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'runs/b123_20261027';REC=RUN/'deadline_guard_recovery1'
def read(path):
    for attempt in range(6):
        try:return json.loads(path.read_text(encoding='utf-8-sig'))
        except (PermissionError,json.JSONDecodeError):
            if attempt==5:raise
            time.sleep(.05)
def write(path,value):
    temp=path.with_name(path.name+'.recovery1.next')
    for attempt in range(6):
        try:
            temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');temp.replace(path);return True
        except PermissionError:time.sleep(.05)
    # Срок остановки проверяется независимо от возможности сохранить очередной статус.
    return False
def matching(pid,birth):
    try:
        p=psutil.Process(int(pid));return p if p.is_running() and abs(p.create_time()-float(birth))<.01 else None
    except psutil.NoSuchProcess:return None
def run():
    auth=read(RUN/'authorization.json');save=datetime.fromisoformat(auth['save_at']).timestamp();stop=datetime.fromisoformat(auth['controller_stop']).timestamp()
    previous=read(REC/'previous_status.json');owned={int(k):v for k,v in previous['owned'].items()};sent=set(previous['saved_folders']);registry=read(RUN/'process_registry.json')
    while time.time()<stop:
        try:registry=read(RUN/'process_registry.json')
        except (PermissionError,json.JSONDecodeError):pass
        for entry in registry['processes']:
            pid,birth=entry['pid'],entry['created'];owned[int(pid)]=birth;p=matching(pid,birth)
            if p:
                try:
                    for child in p.children(recursive=True):owned[child.pid]=child.create_time()
                except psutil.NoSuchProcess:pass
            if time.time()>=save and entry.get('folder') not in sent:
                folder=(ROOT/entry['folder']).resolve();folder.relative_to(RUN.resolve())
                try:(folder/'stop.request').write_text('Author deadline: save and end.\n',encoding='utf-8');sent.add(entry['folder'])
                except PermissionError:pass
        status=dict(state='watching',owned=owned,saved_folders=sorted(sent),updated=datetime.now(timezone.utc).isoformat(),stop_at=auth['controller_stop'],service_recovery='deadline_guard_recovery1')
        write(RUN/'deadline_status.json',status);write(REC/'status.json',status);time.sleep(1)
    for pid,birth in reversed(list(owned.items())):
        p=matching(pid,birth)
        if p:
            try:p.kill()
            except psutil.NoSuchProcess:pass
    until=min(datetime.fromisoformat(auth['hard_stop']).timestamp()-2,time.time()+15)
    while time.time()<until and any(matching(pid,birth) for pid,birth in owned.items()):time.sleep(.2)
    alive=[pid for pid,birth in owned.items() if matching(pid,birth)]
    result=dict(all_owned_ended=not alive,remaining=alive,owned=owned,updated=datetime.now(timezone.utc).isoformat(),author_hard_stop=auth['hard_stop'],service_recovery='deadline_guard_recovery1')
    write(RUN/'deadline_result.json',result);write(REC/'result.json',result)
def launch():
    assert not REC.exists();oldpid=int((RUN/'deadline_guard.pid').read_text());assert not psutil.pid_exists(oldpid)
    REC.mkdir();(REC/'previous_status.json').write_bytes((RUN/'deadline_status.json').read_bytes())
    (REC/'previous_stderr.log').write_bytes((RUN/'deadline.stderr.log').read_bytes())
    with (REC/'stdout.log').open('w',encoding='utf-8') as out,(REC/'stderr.log').open('w',encoding='utf-8') as err:
        child=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--run'],cwd=ROOT,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
    proc=psutil.Process(child.pid);proc.nice(psutil.IDLE_PRIORITY_CLASS)
    proof=dict(pid=proc.pid,created=proc.create_time(),old_pid_ended=oldpid,old_error='PermissionError WinError5 atomic status replacement',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),author_deadline_unchanged=True,old_source_and_logs_preserved=True,
        authorization_sha256=hashlib.sha256((RUN/'authorization.json').read_bytes()).hexdigest())
    assert write(REC/'launch.json',proof)
    print(json.dumps(proof))
if __name__=='__main__':
    if '--run' in sys.argv:run()
    else:launch()
