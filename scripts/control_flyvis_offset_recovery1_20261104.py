"""Последовательный контроллер вычисления с контрольными точками и жёстким сроком."""
from pathlib import Path
from datetime import datetime,timezone
import json,os,psutil,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/flyvis_offset_recovery1_20261104'
def write(name,obj):
 path=OUT/name;t=path.with_suffix('.next');t.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');t.replace(path)
def main():
 p=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'));end=datetime.fromisoformat(p['hard_stop']).timestamp();save=datetime.fromisoformat(p['save_by']).timestamp()
 me=psutil.Process();me.nice(psutil.IDLE_PRIORITY_CLASS);write('controller_launch.json',dict(pid=me.pid,created=me.create_time()))
 stages=[('checks',['runs/flyvis_offset_recovery1_20261104/check_catalog.py']),('prepare',['scripts/flyvis_offset_recovery1_20261104.py','prepare']),('stock',['scripts/flyvis_offset_recovery1_20261104.py','stock']),('fast',['scripts/flyvis_offset_recovery1_20261104.py','fast']),('analysis',['scripts/analyze_flyvis_offset_recovery1_20261104.py']),('audit',['scripts/audit_flyvis_offset_recovery1_20261104.py'])];reports=[];all_owned={}
 for stage,args in stages:
  if time.time()>=save:write('chain_result.json',dict(complete=False,reason='save_window',stages=reports,all_owned=all_owned));return 1
  env=dict(os.environ);env['CUDA_VISIBLE_DEVICES']='';owned={};reason=None
  with (OUT/(stage+'_stdout.log')).open('w',encoding='utf-8') as stdout,(OUT/(stage+'_stderr.log')).open('w',encoding='utf-8') as stderr:
   child=subprocess.Popen([sys.executable,'-u',*args],cwd=ROOT,env=env,stdout=stdout,stderr=stderr,creationflags=subprocess.CREATE_NO_WINDOW)
   proc=psutil.Process(child.pid);proc.nice(psutil.IDLE_PRIORITY_CLASS);owned[proc.pid]=proc.create_time();write(stage+'_launch.json',dict(pid=proc.pid,created=proc.create_time()));started=time.time()
   status=OUT/'status.json' if stage=='prepare' else OUT/stage/'status.json'
   while child.poll() is None:
    try:
     for q in proc.children(recursive=True):owned[q.pid]=q.create_time()
    except psutil.NoSuchProcess:pass
    if time.time()>=end or (stage in ('prepare','stock','fast') and time.time()-max(started,status.stat().st_mtime if status.exists() else started)>600):
     reason='hard_stop' if time.time()>=end else 'watchdog'
     for pid,birth in reversed(list(owned.items())):
      try:
       q=psutil.Process(pid)
       if abs(q.create_time()-birth)<.01:q.kill()
      except psutil.NoSuchProcess:pass
     break
    time.sleep(.5)
   child.wait(timeout=30)
  alive=[]
  for pid,birth in owned.items():
   try:
    if abs(psutil.Process(pid).create_time()-birth)<.01:alive.append(pid)
   except psutil.NoSuchProcess:pass
  rec=dict(stage=stage,exit_code=child.returncode,reason=reason,owned=owned,all_owned_ended=not alive,remaining=alive,ended_utc=datetime.now(timezone.utc).isoformat());write(stage+'_controller_result.json',rec);reports.append(rec);all_owned.update(owned)
  if child.returncode or reason or alive:write('chain_result.json',dict(complete=False,failed_stage=stage,stages=reports,all_owned=all_owned));return 1
 write('chain_result.json',dict(complete=True,stages=reports,all_owned=all_owned,all_owned_ended=True));return 0
if __name__=='__main__':raise SystemExit(main())
