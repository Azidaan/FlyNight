"""Finite CPU controller followed by checked local recording; no chat scheduling."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.z13_parallel_next import read,write

RUN = ROOT / 'runs/z13_context_transfer_20261020'


def main():
    import psutil
    assert (RUN / 'full_authorization.json').exists()
    if (RUN / 'service.json').exists() or (RUN / 'controller.json').exists():
        raise RuntimeError('Existing full service; refusing duplicate')
    write(RUN / 'service.json',dict(pid=os.getpid(),create_time=psutil.Process().create_time(),chat_calls=0))
    env = dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    code = subprocess.call([sys.executable,'-u',str(ROOT/'scripts/z13_context_transfer_20261020.py'),'controller'],cwd=ROOT,env=env)
    end = read(RUN / 'controller_result.json') if (RUN / 'controller_result.json').exists() else None
    if code == 0 and end and end['worker_ended']:
        with (RUN / 'postprocess.stdout.log').open('w',encoding='utf-8') as out, (RUN / 'postprocess.stderr.log').open('w',encoding='utf-8') as err:
            recorded = subprocess.call([sys.executable,str(ROOT/'scripts/z13_record_context_transfer_20261020.py'),'--publish'],cwd=ROOT,env=env,stdout=out,stderr=err)
        write(RUN / 'postprocess_result.json',dict(exit_code=recorded,chat_calls=0))
    else:
        write(RUN / 'service_failure.json',dict(controller_exit=code,controller_result=end,chat_calls=0))


if __name__ == '__main__':main()
