"""Bounded parallel CPU workers with independent ownership/memory/deadline checks."""
from common import *
import os,time,uuid,subprocess,threading
from concurrent.futures import ThreadPoolExecutor,as_completed
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
import psutil
sys.path.insert(0,str(ROOT/'runs/flyvis_repairs_20261106/unification_20261107'))
import system_memory_guard as guard
psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
p=read(OUT/'batch_protocol.json');deadline=datetime.fromisoformat(p['deadline']);assert datetime.now(ZoneInfo('Europe/Moscow'))<deadline
for n,h in p['inputs'].items():assert sha(ROOT/n)==h,n
mode=sys.argv[1];jobs=[('geometry832','worker.py',180),('overlap574','worker.py',600)] if mode=='workers' else [('geometry832','audit.py',300),('overlap574','audit.py',600)]
report=OUT/(mode+'_batch.json');state=dict(started=now(),parent=dict(pid=os.getpid(),birth=psutil.Process().create_time()),mode=mode,maximum_workers=2,jobs={},complete=False);mutex=threading.Lock()
def persist():
 temp=report.with_suffix('.pending');temp.write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf8');temp.replace(report)
persist()
def run(job):
 folder,entry,timeout=job;token=uuid.uuid4().hex;run_dir=guard.HERE/'uncapped_checks'/token;run_dir.mkdir(exist_ok=False);(run_dir/'supervisor.json').write_text(json.dumps(dict(pid=os.getpid(),birth=psutil.Process().create_time())),encoding='utf8');mark=guard.MARK+token;cmd=[sys.executable,'-u',str(OUT/folder/entry),mark]
 with mutex:state['jobs'][folder]=dict(entry=entry,token=token,report=str(run_dir/'memory_guard.json'),started=now());persist()
 remaining=(deadline-datetime.now(ZoneInfo('Europe/Moscow'))).total_seconds();assert remaining>0
 result=guard.supervise(cmd,mark,run_dir/'memory_guard.json',timeout=min(timeout,remaining))
 with mutex:state['jobs'][folder].update(ended=now(),result=result);persist()
 return folder,result
with ThreadPoolExecutor(max_workers=2) as pool:
 for future in as_completed([pool.submit(run,j) for j in jobs]):
  name,result=future.result();print(name,result['returncode'],result['reason'],flush=True)
state['complete']=True;state['ended']=now();state['passed']=all(x['result']['returncode']==0 and x['result']['reason'] is None for x in state['jobs'].values());persist()
raise SystemExit(0 if state['passed'] else 1)
