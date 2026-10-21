"""Independent sequential owned tree, save07:35/drain only current job to07:39:30."""
from datetime import datetime
from pathlib import Path
import os
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_center_hota_20261021 import RUN,read,write,verify,sha,now,MSK
from scripts.z13_hota_deadline_guard import WindowsProcesses


def main():
    import msvcrt
    p=read(RUN/'protocol.json');verify(p);assert read(RUN/'technical.json')['passed']
    assert read(RUN/'probe_controller_result.json')['worker_ended'] and read(RUN/'probe_controller_result.json')['exit_code']==0
    if (RUN/'full_controller_result.json').exists():raise RuntimeError('Existing full attempt; preserve and inspect')
    soft=datetime.fromisoformat(p['save_at']);hard=datetime.fromisoformat(p['hard_stop']);assert now()<soft
    lock=(RUN/'controller.lock').open('a+b')
    if lock.tell()==0:lock.write(b'0');lock.flush()
    lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    identity=dict(protocol_sha256=sha(RUN/'protocol.json'),technical_sha256=sha(RUN/'technical.json'),masa_head_sha256=sha(RUN/'masa_head.joblib'))
    contract=RUN/'execution_identity.json'
    if contract.exists():assert read(contract)==identity
    else:write(contract,identity)
    jobs=[]
    for v in p['videos']:
        jobs.extend(dict(id=a+'_sig_'+v,stage='signatures',arm=a,video=v) for a in p['arms'])
        jobs.extend([dict(id='masa_'+v,stage='masa',video=v),dict(id='stitch_'+v,stage='stitch',video=v)])
    jobs.append(dict(id='eval',stage='eval'));write(RUN/'jobs.json',jobs)
    flags=subprocess.CREATE_NO_WINDOW|subprocess.IDLE_PRIORITY_CLASS;proc=None;known={};processes=WindowsProcesses();outcome='paused';reason='';completed=0
    def collect():
        live={pid for pid,born in known.items() if processes.creation(pid)==born};tree=processes.children()
        for _ in range(16):
            added=set()
            for pid,parent in tree:
                if parent in live and pid not in live:
                    born=processes.creation(pid)
                    if born is not None:known[pid]=born;added.add(pid)
            if not added:break
            live|=added
        return live
    def end_owned():
        live=collect()
        for pid in sorted(live):
            if processes.creation(pid)==known[pid]:subprocess.run(['taskkill','/PID',str(pid),'/T','/F'],capture_output=True)
        if proc is not None:
            try:proc.wait(timeout=20)
            except subprocess.TimeoutExpired:pass
    try:
        write(RUN/'full_controller.json',dict(pid=os.getpid(),started_at=now().isoformat(),save_stop_at=soft.isoformat(),hard_stop_at=hard.isoformat(),jobs=len(jobs)))
        for index,job in enumerate(jobs):
            marker=RUN/'completed_jobs'/(job['id']+'.json')
            if marker.exists():assert read(marker)['identity']==identity;completed+=1;continue
            if now()>=soft or (RUN/'soft.stop').exists():reason='Save deadline: no new job; remaining recorded';break
            verify(p);assert sha(RUN/'masa_head.joblib')==identity['masa_head_sha256']
            switches=['--worker','--attempt','full','--stage',job['stage'],'--budget',str(max(1,int((hard-now()).total_seconds())))]
            for key in ('arm','video'):
                if key in job:switches+=['--'+key,job[key]]
            logpath=RUN/'job_logs'/(job['id']+'.log');logpath.parent.mkdir(exist_ok=True)
            with logpath.open('a',encoding='utf-8') as log:
                proc=subprocess.Popen([sys.executable,'-u',str(ROOT/'scripts/z13_center_hota_20261021.py'),*switches],cwd=ROOT,creationflags=flags,env={**os.environ,'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2'},stdout=log,stderr=log)
                born=processes.creation(proc.pid);assert born is not None;known[proc.pid]=born;stage_start=time.monotonic()
                write(RUN/'active_job.json',dict(job=job,worker_pid=proc.pid,worker_birth=born,started=now().isoformat(),log=logpath.relative_to(ROOT).as_posix()))
                while proc.poll() is None:
                    collect()
                    if now()>=soft:write(RUN/'save_checkpoint.json',dict(at=now().isoformat(),completed_jobs=completed,total_jobs=len(jobs),current=job,remaining=jobs[index:],note='Only current job may drain before hard deadline'))
                    if now()>=hard:end_owned();reason='Hard deadline: interrupted own current job, no completion marker';break
                    write(RUN/'queue_status.json',dict(state='running',job=job,completed_jobs=completed,total_jobs=len(jobs),elapsed_current_seconds=int(time.monotonic()-stage_start),save_at=soft.isoformat(),hard_stop_at=hard.isoformat(),updated=now().isoformat()))
                    time.sleep(1)
                code=proc.wait(timeout=20)
                if reason:break
                if code!=0:raise RuntimeError(job['id']+' failed exit '+str(code)+'; completed jobs retained; inspect service error')
                # Stage process returned; any remaining children are its own compiler workers.
                end_owned();write(marker,dict(identity=identity,job=job,exit_code=0,completed=now().isoformat()));completed+=1;proc=None
        else:outcome='completed'
    except Exception as exc:outcome='failed';reason=str(exc);raise
    finally:
        end_owned();alive={pid:born for pid,born in known.items() if processes.creation(pid)==born}
        pending=[j for j in jobs if not (RUN/'completed_jobs'/(j['id']+'.json')).exists()]
        result=dict(state=outcome,reason=reason,completed_jobs=completed,total_jobs=len(jobs),remaining=pending,known_owned_processes={str(k):v for k,v in known.items()},remaining_owned_processes={str(k):v for k,v in alive.items()},all_own_workers_ended=not alive,ended_at=now().isoformat())
        write(RUN/'full_controller_result.json',result);write(RUN/'queue_status.json',result);lock.close()


if __name__=='__main__':main()
