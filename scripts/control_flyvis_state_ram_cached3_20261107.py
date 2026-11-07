"""Independent watchdog for the RAM-bounded recovery; never touches other trees."""
from pathlib import Path
from datetime import datetime,timedelta
from zoneinfo import ZoneInfo
import hashlib,json,os,subprocess,sys,time
import psutil
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"runs/flyvis_state_subgroups_ram_cached3_20261107"
STAGE=sys.argv[1];EXEC=OUT/("execution_"+STAGE)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,obj):
    temp=p.with_suffix(".next");temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");os.replace(temp,p)
def main():
    assert STAGE in ("probe","run","audit");EXEC.mkdir(exist_ok=False)
    manifest=json.loads((OUT/"recovery_manifest.json").read_text())
    for path,digest in manifest["new_sources"].items():assert sha(ROOT/path)==digest,path
    assert psutil.virtual_memory().available>=12*1024**3
    p=json.loads((OUT/"protocol.json").read_text());deadline=datetime.fromisoformat(p["deadline_msk"])-timedelta(seconds=60)
    assert datetime.now(ZoneInfo("Europe/Moscow"))<deadline
    status=OUT/({"probe":"probe_status.json","run":"status.json","audit":"audit_status.json"}[STAGE])
    own={};reason=None;peak_rss=peak_private=0;minimum_available=psutil.virtual_memory().available
    with (EXEC/"stdout.log").open("w",encoding="utf-8") as stdout,(EXEC/"stderr.log").open("w",encoding="utf-8") as stderr:
        child=subprocess.Popen([str(ROOT/".venv/Scripts/python.exe"),str(ROOT/"scripts/flyvis_state_ram_cached3_20261107.py"),STAGE],
            cwd=ROOT,stdout=stdout,stderr=stderr,creationflags=subprocess.CREATE_NO_WINDOW)
        proc=psutil.Process(child.pid);own[proc.pid]=proc.create_time()
        save(EXEC/"launch.json",dict(pid=proc.pid,birth=own[proc.pid],controller_pid=os.getpid(),controller_birth=psutil.Process().create_time()))
        last=None;last_change=time.monotonic();last_save=0;started=time.monotonic()
        while child.poll() is None:
            try:
                for x in [proc]+proc.children(recursive=True):
                    if x.pid not in own:
                        own[x.pid]=x.create_time();x.nice(psutil.IDLE_PRIORITY_CLASS)
            except (psutil.NoSuchProcess,psutil.AccessDenied):pass
            resident=private=0
            for pid,birth in own.items():
                try:
                    x=psutil.Process(pid)
                    if abs(x.create_time()-birth)<.01:
                        mem=x.memory_info();resident+=mem.rss;private+=getattr(mem,"private",mem.vms)
                except psutil.NoSuchProcess:pass
            available=psutil.virtual_memory().available;minimum_available=min(minimum_available,available)
            peak_rss=max(peak_rss,resident);peak_private=max(peak_private,private)
            if time.monotonic()-last_save>=1:
                save(EXEC/"memory_status.json",dict(owned=own,private=private,rss=resident,available=available,peak_rss=peak_rss,peak_private=peak_private,minimum_available=minimum_available,elapsed_seconds=time.monotonic()-started))
                last_save=time.monotonic()
            if status.exists():
                try:
                    now_hash=sha(status)
                    if now_hash!=last:last=now_hash;last_change=time.monotonic()
                except OSError:pass
            if available<12*1024**3:reason="system_memory_reserve_below12GiB";break
            if private>4*1024**3+64*1024**2:reason="tree_memory_cap";break
            if datetime.now(ZoneInfo("Europe/Moscow"))>=deadline:reason="deadline";break
            if time.monotonic()-last_change>600:reason="progress_watchdog";break
            time.sleep(.25)
        if reason:
            for pid,birth in reversed(list(own.items())):
                try:
                    x=psutil.Process(pid)
                    if abs(x.create_time()-birth)<.01:x.terminate()
                except (psutil.NoSuchProcess,psutil.AccessDenied):pass
            try:child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                for pid,birth in own.items():
                    try:
                        x=psutil.Process(pid)
                        if abs(x.create_time()-birth)<.01:x.kill()
                    except (psutil.NoSuchProcess,psutil.AccessDenied):pass
        code=child.wait()
    # Termination is asynchronous on Windows. Wait for the actual identities.
    for pid,birth in own.items():
        try:
            x=psutil.Process(pid)
            if abs(x.create_time()-birth)<.01:
                try:x.wait(timeout=5)
                except psutil.TimeoutExpired:
                    x.kill();x.wait(timeout=3)
        except psutil.NoSuchProcess:pass
    live=[]
    for pid,birth in own.items():
        try:
            x=psutil.Process(pid)
            if abs(x.create_time()-birth)<.01:live.append(dict(pid=pid,birth=birth))
        except psutil.NoSuchProcess:pass
    result=dict(stage=STAGE,exit_code=code,reason=reason,owned=own,all_owned_ended=not live,live=live,
        peak_rss=peak_rss,peak_private=peak_private,minimum_system_available=minimum_available,ended_msk=datetime.now(ZoneInfo("Europe/Moscow")).isoformat(),
        os_worker_job_limit=4*1024**3,system_free_reserve=12*1024**3)
    save(EXEC/"controller_result.json",result)
    if STAGE=="run":save(OUT/"controller_result.json",result)
    print(json.dumps(result));assert code==0 and reason is None and not live
if __name__=="__main__":main()
