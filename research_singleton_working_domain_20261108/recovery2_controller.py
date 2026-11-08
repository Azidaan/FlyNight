import os,sys,subprocess,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_singleton_working_domain_20261108.common import *
RECOVERY=RUN/'recovery2'
def main():
 import psutil,time
 while not (RECOVERY/'armed.json').exists():time.sleep(.1)
 p=read(RECOVERY/'protocol.json')
 for n,h in p['sources'].items():assert sha(ROOT/n)==h,n
 with (RECOVERY/'worker.log').open('x',encoding='utf-8') as out,(RECOVERY/'worker.err').open('x',encoding='utf-8') as err:
  child=subprocess.Popen([sys.executable,'-u','research_singleton_working_domain_20261108/effective_worker_recovery2.py'],cwd=ROOT,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW|subprocess.IDLE_PRIORITY_CLASS)
  write(RECOVERY/'worker.started.json',dict(pid=child.pid,created=psutil.Process(child.pid).create_time()),new=True);code=child.wait()
 lines=[s for s in (RECOVERY/'worker.log').read_text('utf-8').splitlines() if s.startswith('SYSTEM_MEMORY_GUARD ')];assert lines,(code,'No supervisor completion')
 guardian=json.loads(lines[-1].split(' ',1)[1]);write(RECOVERY/'worker.ended.json',dict(returncode=code,guardian=guardian),new=True);assert code==0 and guardian['returncode']==0 and guardian['reason'] is None
 for pid,birth in guardian['owned'].items():
  try:
   q=psutil.Process(int(pid))
   if abs(q.create_time()-birth)<.01:q.wait(timeout=5)
  except psutil.NoSuchProcess:pass
 for n,h in p['sources'].items():assert sha(ROOT/n)==h,n
 phase();assert read(RECOVERY/'independent_audit.json')['passed']
 write(RECOVERY/'queue_result.json',dict(passed=True,data_only=True,original_failure_preserved=True,new_native_calls=0,original_execution_deadline_unchanged=True,report_sha256=sha(RECOVERY/'report.json'),independent_audit_sha256=sha(RECOVERY/'independent_audit.json')),new=True)
 write(RUN/'queue_result.json',dict(passed=True,data_only=True,recovered_by='runs/singleton_working_domain_20261108/recovery2/queue_result.json',report_sha256=sha(RECOVERY/'report.json'),independent_audit_sha256=sha(RECOVERY/'independent_audit.json'),no_automatic_followup=True),new=True)
if __name__=='__main__':main()
