"""Check all generated semantic worker paths and finisher construction, without research updates."""
from pathlib import Path
from datetime import timedelta
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys
import inspect
import hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import z13_region_semantic_20261025 as stage
from scripts.z13_candidate_reserve_20261021 import read,write,sha,now
OUT=ROOT/'runs/z13_region_semantic_execution_check_20261025'


def main():
    import torch
    assert not OUT.exists();torch.set_num_threads(2);torch.set_default_device('cpu')
    p=dict(steps=1,batch=128,seeds=[0,1,2],fit20=['synthetic'],fit84=['synthetic'],evaluate26=['synthetic'],save_at=(now()+timedelta(seconds=600)).isoformat())
    write(OUT/'training/protocol.json',p);write(OUT/'old/training/report.json',dict(mean_seed_per_video=dict(D=dict(synthetic=0.))))
    def fake(torch,arm,split):
        n=130;mask=torch.ones(n,2,dtype=torch.bool)
        d=dict(x=torch.ones(n,2,68),seed=torch.ones(n,68),geo=torch.zeros(n,2,8),mask=mask,
            uid=torch.arange(n),kind=torch.ones(n,dtype=torch.int8),video=torch.zeros(n,dtype=torch.int16),good=torch.zeros_like(mask),harm=torch.zeros_like(mask),
            semantic_good=torch.zeros_like(mask),baseline_good=torch.ones(n,dtype=torch.bool))
        d['good'][0,1]=True;d['harm'][1,0]=True;d['semantic_good'][:,1]=True;d['baseline_good'][0]=False
        return d
    ns={**stage.original.namespace(),'RUN':OUT,'BASE':OUT/'old','load_split':fake,'semantic_choice_loss':stage.semantic_choice_loss,
        'primary':stage.primary,'status':lambda *a,**kw:None,'verify_sources':lambda p:None}
    code=stage.training_source();exec(compile(code,str(Path(__file__)),'exec'),ns);ns['train_worker']()
    report=read(OUT/'training/report.json');assert report['complete'] and report['models']==12 and not report['cuda_initialized']
    # Build and compile the complete final verifier without invoking it on live results.
    prefix=inspect.getsource(stage.finish).split('    ns={**vars(flow)',1)[0]+'    return code\n'
    ns=dict(vars(stage));exec(compile(prefix,str(Path(__file__)),'exec'),ns);finish_code=ns['finish']();compile(finish_code,str(Path(__file__)),'exec')
    write(OUT/'receipt.json',dict(at=now().isoformat(),all12_synthetic_execution_paths_passed=True,steps_per_model=1,
        finisher_transforms_and_compile_passed=True,scientific_training_sha256=hashlib.sha256(code.encode()).hexdigest(),
        actual_protocol_matches=hashlib.sha256(code.encode()).hexdigest()==read(stage.RUN/'protocol.json')['executed_training_sha256'],source_sha256=sha(Path(__file__)),research_metrics_produced=False))
    print('All12 synthetic worker paths and complete finisher construction passed')


if __name__=='__main__':main()
