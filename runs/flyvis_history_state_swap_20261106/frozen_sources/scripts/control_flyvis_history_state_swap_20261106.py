"""Hidden bounded CPU controller with deadline, progress watchdog and owned-child audit."""
from pathlib import Path
from datetime import datetime,timedelta
from zoneinfo import ZoneInfo
import hashlib,json,os,subprocess,sys,time
import psutil
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/flyvis_history_state_swap_20261106'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
 t=p.with_suffix('.next');t.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');os.replace(t,p)
def main():
 p=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'));assert sha(Path(__file__))==p['frozen_sources']['scripts/control_flyvis_history_state_swap_20261106.py'];assert sha(ROOT/'scripts/flyvis_history_state_swap_20261106.py')==p['frozen_sources']['scripts/flyvis_history_state_swap_20261106.py'];deadline=datetime.fromisoformat(p['deadline_msk'])-timedelta(seconds=60);assert datetime.now(ZoneInfo('Europe/Moscow'))<deadline;own={};reason=None;rc=None
 with (OUT/'worker.stdout.log').open('w',encoding='utf-8') as stdout,(OUT/'worker.stderr.log').open('w',encoding='utf-8') as stderr:
  child=subprocess.Popen([str(ROOT/'.venv/Scripts/python.exe'),str(ROOT/'scripts/flyvis_history_state_swap_20261106.py'),'run'],cwd=str(ROOT),stdout=stdout,stderr=stderr,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0));proc=psutil.Process(child.pid);own[proc.pid]=proc.create_time();write(OUT/'worker.launch.json',dict(pid=proc.pid,birth=own[proc.pid],controller_pid=os.getpid(),controller_birth=psutil.Process().create_time(),deadline_msk=p['deadline_msk']));last=sha(OUT/'status.json');lastchange=time.monotonic()
  while child.poll() is None:
   try:
    for x in proc.children(recursive=True):own.setdefault(x.pid,x.create_time())
   except (psutil.NoSuchProcess,psutil.AccessDenied):pass
   try:
    h=sha(OUT/'status.json')
    if h!=last:last=h;lastchange=time.monotonic()
   except (OSError,PermissionError):pass
   if datetime.now(ZoneInfo('Europe/Moscow'))>=deadline:reason='deadline';break
   if time.monotonic()-lastchange>p['watchdog_seconds']:reason='progress_watchdog';break
   time.sleep(3)
  if reason:
   for pid,birth in list(own.items())[::-1]:
    try:
     x=psutil.Process(pid)
     if abs(x.create_time()-birth)<.01:x.terminate()
    except (psutil.NoSuchProcess,psutil.AccessDenied):pass
   try:child.wait(timeout=15)
   except subprocess.TimeoutExpired:
    for pid,birth in list(own.items())[::-1]:
     try:
      x=psutil.Process(pid)
      if abs(x.create_time()-birth)<.01:x.kill()
     except (psutil.NoSuchProcess,psutil.AccessDenied):pass
  rc=child.wait()
 live=[]
 for pid,birth in own.items():
  try:
   x=psutil.Process(pid)
   if abs(x.create_time()-birth)<.01 and x.is_running():live.append(dict(pid=pid,birth=birth))
  except psutil.NoSuchProcess:pass
 done=dict(exit_code=rc,reason=reason,owned=own,all_owned_ended=not live,live=live,ended_msk=datetime.now(ZoneInfo('Europe/Moscow')).isoformat());write(OUT/'controller_result.json',done);print(json.dumps(done));assert rc==0 and not live and reason is None
if __name__=='__main__':main()
