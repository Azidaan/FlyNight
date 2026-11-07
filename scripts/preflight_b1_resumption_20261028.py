"""Проверки до единственного оставшегося GPU-видео."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
from datetime import datetime
import sys,subprocess,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.continuation_service_20261028 import RUN,service
read,write,sha,now=service.read,service.write,service.sha,service.now

def main():
    folder=RUN/'preflight';p=read(folder/'protocol.json');end=datetime.fromisoformat(read(folder/'execution_deadline.json')['deadline']).timestamp()
    seed=read(RUN/'b1_tracker/resume_seed.json')
    assert len(seed['completed_videos'])==19
    for path,digest in seed['output_sha256'].items():assert sha(ROOT/path)==digest,path
    for path,digest in p['sha256'].items():assert sha(ROOT/path)==digest,path
    outcomes=[]
    for i,args in enumerate([['scripts/validate_project_history.py'],['-m','pytest','tests/test_project_history.py','-q'],
        ['-m','pytest','tests/test_b1_tracker_continuation.py','tests/test_b1_cached_eye.py','tests/test_b1_audit_checkpoints.py','-q']]):
        write(folder/'status.json',dict(state='running',step=i,command=args,updated=now()),required=False)
        with (folder/f'check_{i}.log').open('w',encoding='utf-8') as out:
            r=subprocess.run([sys.executable,*args],cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,timeout=max(1,end-time.time()-10))
        outcomes.append(dict(command=args,exit_code=r.returncode));assert r.returncode==0,args
    write(folder/'report.json',dict(passed=True,checks=outcomes,reused_videos=19,updated=now()))

if __name__=='__main__':main()
