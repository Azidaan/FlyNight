"""Use the documented script entrypoints for DRB and neck checks, CPU only."""
from pathlib import Path
import os,sys,json,importlib.util,time
os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from system_memory_guard import guard_entrypoint
job=guard_entrypoint()
import psutil
# System reserve is monitored independently by the supervisor.
import torch
torch.set_default_device('cpu');torch.set_num_threads(2)
torch.cuda.is_available=lambda:False
torch.cuda.device_count=lambda:0
def forbidden(*args,**kwargs):raise RuntimeError('CPU-only verification forbids CUDA initialization')
torch.cuda._lazy_init=forbidden
name=sys.argv[1];assert name in ('test_drb','test_neck_v3')
out=job.run_dir/(name+'_manual.json')
assert not out.exists()
spec=importlib.util.spec_from_file_location(name,ROOT/'tests'/(name+'.py'))
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
started=time.monotonic();code=1;error=None
try:code=int(module.main())
except BaseException as exc:error=repr(exc);raise
finally:
    result=dict(returncode=code,error=error,failures=module.FAILURES,seconds=time.monotonic()-started,
        cuda_initialized=torch.cuda.is_initialized(),gpu_checks_not_run=True,job=job.read())
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)
raise SystemExit(code)
