"""Независимая остановка зарегистрированного дерева в указанный автором срок."""
from pathlib import Path
from datetime import datetime,timezone
import json,psutil,sys,time
OUT=Path(__file__).resolve().parents[1]/'runs/flyvis_offset_recovery2_20261104'
def write(name,obj):(OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
 me=psutil.Process();write('guard_launch.json',dict(pid=me.pid,created=me.create_time()))
 deadline=datetime.fromisoformat(json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))['hard_stop']).timestamp()
 while time.time()<deadline:
  if (OUT/'chain_result.json').exists():write('guard_result.json',dict(reason='chain_finished',at=datetime.now(timezone.utc).isoformat()));return 0
  time.sleep(2)
 live=[]
 for f in ['controller_launch.json','checks_launch.json','prepare_launch.json','stock_launch.json','fast_launch.json','analysis_launch.json','audit_launch.json']:
  path=OUT/f
  if path.exists():
   q=json.loads(path.read_text());
   try:
    x=psutil.Process(q['pid'])
    if abs(x.create_time()-q['created'])<.01:live.append(x)
   except psutil.NoSuchProcess:pass
 owned={}
 for x in live:
  try:
   for p in [*x.children(recursive=True),x]:owned[p.pid]=p.create_time()
  except psutil.NoSuchProcess:pass
 for pid,birth in reversed(list(owned.items())):
  try:
   x=psutil.Process(pid)
   if abs(x.create_time()-birth)<.01:x.kill()
  except psutil.NoSuchProcess:pass
 time.sleep(1);remaining=[]
 for pid,birth in owned.items():
  try:
   if abs(psutil.Process(pid).create_time()-birth)<.01:remaining.append(pid)
  except psutil.NoSuchProcess:pass
 write('guard_result.json',dict(reason='hard_stop',owned=owned,remaining=remaining,all_owned_ended=not remaining,at=datetime.now(timezone.utc).isoformat()));return 0 if not remaining else 1
if __name__=='__main__':raise SystemExit(main())
