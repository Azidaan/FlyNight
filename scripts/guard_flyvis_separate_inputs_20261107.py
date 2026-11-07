"""External disk reserve for a finite input-separation chain; no scientific changes."""
import json,os,sys,time,subprocess,hashlib
from pathlib import Path
import psutil,shutil
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"runs/flyvis_separate_inputs_20261107"
def dump(p,d):
 t=p.with_suffix(".pending");t.write_text(json.dumps(d,indent=2),encoding="utf8");os.replace(t,p)
def main():
 assert shutil.disk_usage(ROOT).free>=5*1024**3
 info=OUT/"disk_supervisor.json";assert not info.exists()
 receipt=dict(reserve_bytes=5*1024**3,minimum_free_bytes=shutil.disk_usage(ROOT).free,owned={},stopped=None,returncode=None)
 command=[sys.executable,"-u",str(ROOT/"scripts/chain_flyvis_separate_inputs_20261107.py")]
 with (OUT/"chain.log").open("xb") as log:
  p=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
  root=psutil.Process(p.pid);receipt["owned"][p.pid]=root.create_time()
  while p.poll() is None:
   try:
    assert root.create_time()==receipt["owned"][p.pid]
    for c in root.children(recursive=True):receipt["owned"].setdefault(c.pid,c.create_time())
   except psutil.NoSuchProcess:pass
   free=shutil.disk_usage(ROOT).free;receipt["minimum_free_bytes"]=min(free,receipt["minimum_free_bytes"])
   if free<receipt["reserve_bytes"]:
    receipt["stopped"]="disk_reserve"
    for pid,birth in reversed(list(receipt["owned"].items())):
     try:
      q=psutil.Process(int(pid))
      if q.create_time()==birth:q.kill()
     except psutil.NoSuchProcess:pass
    break
   dump(info,receipt);time.sleep(.3)
  receipt["returncode"]=p.wait(timeout=10);dump(info,receipt)
 return receipt["returncode"] if not receipt["stopped"] else 125
if __name__=="__main__":raise SystemExit(main())
