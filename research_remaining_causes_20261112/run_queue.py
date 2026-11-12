"""Execute a fingerprinted sequence using the existing independent memory guard."""
from common import *
import os,time,uuid,psutil,msvcrt
sys.path.insert(0,str(ROOT/'runs/flyvis_repairs_20261106/unification_20261107'))
import system_memory_guard as guard
qpath=Path(sys.argv[1]).resolve();q=read(qpath)
assert sha(Path(__file__))==q['runner_sha256']
for n,h in q['inputs'].items():assert sha(ROOT/n)==h,n
deadline=datetime.fromisoformat(q['deadline']);assert datetime.now(ZoneInfo('Europe/Moscow'))<deadline
psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
report=qpath.with_name(qpath.stem+'_execution.json');assert not report.exists()
state=dict(started=now(),parent=dict(pid=os.getpid(),birth=psutil.Process().create_time()),jobs=[],complete=False)
def persist():
 t=report.with_suffix('.pending');t.write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf8');t.replace(report)
persist()
with (guard.HERE/'sequential_memory_guard.lock').open('a+b') as lock:
 if lock.tell()==0:lock.write(b'0');lock.flush()
 lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
 try:
  for job in q['jobs']:
   for n,h in q['inputs'].items():assert sha(ROOT/n)==h,n
   token=uuid.uuid4().hex;rd=guard.HERE/'uncapped_checks'/token;rd.mkdir(exist_ok=False)
   (rd/'supervisor.json').write_text(json.dumps(state['parent']),encoding='utf8')
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
