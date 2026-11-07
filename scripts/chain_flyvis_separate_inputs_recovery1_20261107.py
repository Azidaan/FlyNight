"""Finite separately versioned chain after an interface failure, with owned disk monitoring."""
import os,sys,json,time,subprocess,shutil,hashlib
from pathlib import Path
import psutil
ROOT=Path(__file__).resolve().parents[1];os.chdir(ROOT);OUT=ROOT/"runs/flyvis_separate_inputs_20261107";REC=OUT/"recovery1"
def write(p,x):
 t=p.with_suffix(".pending");t.write_text(json.dumps(x,indent=2,ensure_ascii=False),encoding="utf8");os.replace(t,p)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 REC.mkdir(exist_ok=False)
 old=json.loads((OUT/"chain_result.json").read_bytes());assert old["passed"] is False and old["failed"]=="inputs"
 paths=["scripts/flyvis_input_separation_recovery1_20261107.py",str(Path(__file__).relative_to(ROOT))]
 write(REC/"protocol.json",dict(reason="Original BoxEye returns B,T,1,721; adapter removes ONLY singleton channel. Original source/gates/protocol/error unchanged.",source_sha256={p:sha(p) for p in paths},old_failure_sha256=sha(OUT/"chain_result.json"),old_protocol_sha256=sha(OUT/"protocol.json")))
 me=psutil.Process();write(REC/"launch.json",dict(pid=me.pid,birth=me.create_time()))
 result=dict(passed=False,owned={},steps=[],minimum_free=shutil.disk_usage(ROOT).free)
 for name,script,args,done in [
  ("inputs","scripts/flyvis_input_separation_recovery1_20261107.py",[],"inputs_completed.json"),
  ("gpu","scripts/flyvis_input_separation_20261107.py",["--gpu"],"inference_completed.json"),
  ("audit","scripts/audit_flyvis_separate_inputs_20261107.py",[],"independent_audit.json")]:
  if (OUT/done).exists():result["steps"].append(dict(stage=name,reused=True));continue
  assert shutil.disk_usage(ROOT).free>5*1024**3
  with (REC/(name+".log")).open("xb") as log:
   child=subprocess.Popen([sys.executable,"-u",script,*args],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
   proc=psutil.Process(child.pid);result["owned"][proc.pid]=proc.create_time()
   while child.poll() is None:
    try:
     for p in proc.children(recursive=True):result["owned"].setdefault(p.pid,p.create_time())
    except psutil.NoSuchProcess:pass
    free=shutil.disk_usage(ROOT).free;result["minimum_free"]=min(result["minimum_free"],free)
    write(REC/"status.json",dict(stage=name,owned=result["owned"],minimum_free=result["minimum_free"]))
    if free<5*1024**3:
     result["disk_stopped"]=True
     for pid,birth in reversed(list(result["owned"].items())):
      try:
       p=psutil.Process(int(pid))
       if p.create_time()==birth:p.kill()
      except psutil.NoSuchProcess:pass
     break
    time.sleep(.3)
   code=child.wait(timeout=10)
  result["steps"].append(dict(stage=name,code=code))
  if code!=0:write(REC/"controller.json",result);return code
  assert (OUT/done).exists()
 result["passed"]=True;write(REC/"controller.json",result);return 0
if __name__=="__main__":raise SystemExit(main())
