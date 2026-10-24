"""Small CPU-only autograd proof for this Windows CUDA visibility failure."""
from pathlib import Path
import os,sys,subprocess,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_candidate_reserve_20261021 import write,now


if __name__=='__main__':
    out=ROOT/'runs/z13_competing_candidates_20261024/cpu_visibility_diagnostic.json';assert not out.exists()
    source="import torch,json;torch.set_default_device('cpu');x=torch.tensor(2.,requires_grad=True);(x*x).backward();print(json.dumps(dict(gradient=float(x.grad),device=str(x.device),cuda_initialized=torch.cuda.is_initialized())))"
    rows={}
    for name,value in [('negative','-1'),('empty',''),('visible','0')]:
        p=subprocess.run([str(ROOT/'.venv/Scripts/python.exe'),'-X','faulthandler','-c',source],cwd=ROOT,env={**os.environ,'CUDA_VISIBLE_DEVICES':value},capture_output=True,text=True,timeout=30)
        rows[name]=dict(value=value,exit_code=p.returncode,stdout=p.stdout,stderr=p.stderr)
    assert rows['negative']['exit_code']!=0
    for name in ('empty','visible'):
        assert rows[name]['exit_code']==0
        r=json.loads(rows[name]['stdout']);assert r==dict(gradient=4.,device='cpu',cuda_initialized=False)
    write(out,dict(at=now().isoformat(),probe='Gradient of x*x at CPU x=2, no model or dataset',rows=rows,confirmed='This installed Windows build crashes CPU autograd with visibility-1; empty visibility preserves CPU gradient and no CUDA initialization. Thread count/copy/MKLDNN changes did not explain the prior failure.'))
