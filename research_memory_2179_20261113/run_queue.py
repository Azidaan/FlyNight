"""Fingerprint-bound sequential executor, with one owner per GPU and a system reserve."""
from common import *
import os,uuid,psutil,msvcrt
sys.path.insert(0,str(ROOT/'runs/flyvis_repairs_20261106/unification_20261107'))
import system_memory_guard as guard
path=Path(sys.argv[1]).resolve();path.relative_to(OUT.resolve());q=read(path)
assert sha(__file__)==q['runner_sha256']
deadline=datetime.fromisoformat(q['deadline'])
assert datetime.now(ZoneInfo('Europe/Moscow'))<deadline
psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
report=path.with_name(path.stem+'_execution.json');assert not report.exists()
state=dict(started=now(),parent=dict(pid=os.getpid(),birth=psutil.Process().create_time()),jobs=[],complete=False)
def persist():
 t=report.with_suffix('.pending');t.write_bytes((json.dumps(state,ensure_ascii=False,indent=2)+'\n').encode());t.replace(report)
persist()
with (guard.HERE/'sequential_memory_guard.lock').open('a+b') as lock:
 if lock.tell()==0:lock.write(b'0');lock.flush()
 lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
 try:
  for job in q['jobs']:
   for n,h in q['inputs'].items():assert sha(ROOT/n)==h,n
   token=uuid.uuid4().hex;rd=guard.HERE/'uncapped_checks'/token;rd.mkdir(exist_ok=False)
   (rd/'supervisor.json').write_bytes(json.dumps(state['parent']).encode())
   marker=guard.MARK+token;cmd=[sys.executable,'-u',str(ROOT/job['entry']),*job.get('args',[]),marker]
   item=dict(entry=job['entry'],token=token,report=str(rd/'memory_guard.json'),started=now());state['jobs'].append(item);persist()
   remaining=(deadline-datetime.now(ZoneInfo('Europe/Moscow'))).total_seconds();assert remaining>0
   result=guard.supervise(cmd,marker,rd/'memory_guard.json',timeout=min(job.get('timeout',600),remaining))
   item.update(ended=now(),result=result);persist();print(job['entry'],result['returncode'],result['reason'],flush=True)
   if result['returncode'] or result['reason']:break
 finally:
  lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)
state.update(complete=True,ended=now(),passed=len(state['jobs'])==len(q['jobs']) and all(x['result']['returncode']==0 and x['result']['reason'] is None for x in state['jobs']));persist()
raise SystemExit(0 if state['passed'] else 1)
