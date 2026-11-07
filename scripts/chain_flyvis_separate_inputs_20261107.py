"""One hidden finite chain; retain first errors and guarded process identities."""
import os,sys,json,time,subprocess
from pathlib import Path
import psutil
ROOT=Path(__file__).resolve().parents[1];os.chdir(ROOT)
OUT=ROOT/"runs/flyvis_separate_inputs_20261107";SCRIPT=ROOT/"scripts/flyvis_input_separation_20261107.py"
def save(p,x):
 tmp=p.with_suffix(".pending");tmp.write_text(json.dumps(x,indent=2,ensure_ascii=False),encoding="utf8");os.replace(tmp,p)
def main():
 assert not (OUT/"chain_result.json").exists()
 launch=OUT/"chain_launch.json"
 if launch.exists():
  old=json.loads(launch.read_bytes())
  try:assert psutil.Process(old["pid"]).create_time()!=old["birth"],"Previous chain alive"
  except psutil.NoSuchProcess:pass
 me=psutil.Process();save(launch,dict(pid=me.pid,birth=me.create_time()))
 steps=[]
 for name,args,done in [
  ("inputs",[str(SCRIPT),"--inputs"],"inputs_completed.json"),
  ("gpu",[str(SCRIPT),"--gpu"],"inference_completed.json"),
  ("audit",[str(ROOT/"scripts/audit_flyvis_separate_inputs_20261107.py")],"independent_audit.json")]:
  if (OUT/done).exists():steps.append(dict(stage=name,reused=True));continue
  log=OUT/(name+".log");assert not log.exists(),"Keep old failures; use a new recovery controller"
  with log.open("xb") as f:
   p=subprocess.Popen([sys.executable,"-u",*args],stdout=f,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW,cwd=ROOT)
   born=psutil.Process(p.pid).create_time();save(OUT/"chain_status.json",dict(stage=name,pid=p.pid,birth=born,at=time.time()))
   code=p.wait()
  steps.append(dict(stage=name,returncode=code,pid=p.pid,birth=born))
  if code!=0:
   save(OUT/"chain_result.json",dict(passed=False,failed=name,steps=steps));return code
  assert (OUT/done).exists()
 save(OUT/"chain_result.json",dict(passed=True,steps=steps));return 0
if __name__=="__main__":raise SystemExit(main())
