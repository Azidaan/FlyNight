"""Independent CPU budget; serialize catalog publication after the GPU recorder."""
from datetime import datetime,timezone
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_parallel_next import read,write,sha
RUN=ROOT/'runs/z13_map_utility_20261018'


def main():
    import psutil
    p=read(RUN/'protocol.json')
    for rel,digest in p['sha256'].items():assert sha(ROOT/rel)==digest,rel
    if (RUN/'controller.json').exists():raise RuntimeError('Existing controller; no duplicate')
    own={os.getpid()}|{x.pid for x in psutil.Process().parents()}
    for process in psutil.process_iter(['cmdline']):
        if process.pid not in own and any(Path(x).name in {'z13_map_utility.py',Path(__file__).name} for x in process.info['cmdline'] or []):
            raise RuntimeError(f'Duplicate process{process.pid}')
    soft=datetime.fromisoformat(p['save_stop_at']).timestamp();hard=datetime.fromisoformat(p['hard_stop_at']).timestamp()
    if time.time()>=soft:raise RuntimeError('No new computation after cutoff')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    command=[sys.executable,'-u',str(ROOT/'scripts/z13_map_utility.py')]
    with (RUN/'stdout.log').open('w',encoding='utf-8') as out,(RUN/'stderr.log').open('w',encoding='utf-8') as err:
        child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=out,stderr=err,creationflags=subprocess.IDLE_PRIORITY_CLASS)
        born=psutil.Process(child.pid).create_time()
        write(RUN/'controller.json',dict(pid=child.pid,create_time=born,command=command,started_at=datetime.now(timezone.utc).isoformat(),
             soft_deadline=p['save_stop_at'],hard_deadline=p['hard_stop_at'],device='cpu',chat_automation=False))
        stopped=timed_out=False
        while child.poll() is None:
            if time.time()>=soft and not stopped:
                (RUN/'stop.request').write_text('CPU budget ended; save completed videos\n',encoding='utf-8');stopped=True
            if time.time()>=hard:
                process=psutil.Process(child.pid);assert abs(process.create_time()-born)<.01
                subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],check=True,capture_output=True)
                child.wait(timeout=20);timed_out=True;break
            time.sleep(2)
    write(RUN/'controller_result.json',dict(exit_code=child.returncode,worker_ended=True,hard_timeout=timed_out,ended=datetime.now(timezone.utc).isoformat()))
    # The scientific worker is finished. Waiting here performs no new experiment.
    gpu_result=ROOT/'runs/z13_aligned_core_20261018/postprocess_result.json'
    gpu_protocol=read(ROOT/'runs/z13_aligned_core_20261018/protocol.json')
    publish_deadline=datetime.fromisoformat(gpu_protocol['hard_stop_at']).timestamp()+600
    while not gpu_result.exists():
        if time.time()>=publish_deadline:
            write(RUN/'postprocess_result.json',dict(exit_code=1,chat_calls=0,
                  reason='GPU recorder completion absent; scientific outputs saved, catalog publication deferred'))
            return
        time.sleep(10)
    with (RUN/'postprocess.stdout.log').open('w',encoding='utf-8') as out,(RUN/'postprocess.stderr.log').open('w',encoding='utf-8') as err:
        result=subprocess.run([sys.executable,str(ROOT/'scripts/z13_record_map_utility.py')],cwd=ROOT,env=env,stdout=out,stderr=err)
    write(RUN/'postprocess_result.json',dict(exit_code=result.returncode,chat_calls=0))


if __name__=='__main__':main()
