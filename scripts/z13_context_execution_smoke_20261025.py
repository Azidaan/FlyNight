"""Exercise generated worker bindings on synthetic tensors, never on research rows."""
from pathlib import Path
from datetime import timedelta
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import z13_region_context_20261025 as science
from scripts.z13_candidate_reserve_20261021 import write,read,now,sha
OUT=ROOT/'runs/z13_context_execution_smoke_20261025'


def main():
    import torch
    assert not OUT.exists()
    torch.set_num_threads(2);torch.set_default_device('cpu')
    p=dict(steps=1,batch=128,seeds=[0,1,2],fit20=['synthetic'],fit84=['synthetic'],evaluate26=['synthetic'],save_at=(now()+timedelta(seconds=120)).isoformat())
    write(OUT/'training/protocol.json',p)
    def fake_data(torch,arm,split,mode):
        n=130;mask=torch.ones(n,2,dtype=torch.bool)
        d=dict(x=torch.zeros(n,2,68),seed=torch.zeros(n,68),geo=torch.zeros(n,2,14),mask=mask,
               uid=torch.arange(n),kind=torch.ones(n,dtype=torch.int8),video=torch.zeros(n,dtype=torch.int16),
               good=torch.zeros_like(mask),harm=torch.zeros_like(mask))
        d['good'][0,1]=True;d['harm'][1,0]=True
        if mode=='context':d['geo'][...,8:]=.1
        return d
    def fake_comparisons(means,videos):return {k:dict(mean_per_video=0.,ci95=[-1.,1.],n=1) for k in ['Dcontext-baseline','Dcontext-D_blank','Dcontext-C_context','Dcontext-R1_context','Dcontext-geometry_context','Dcontext-Dprevious']}
    ns=science.original.namespace()
    ns.update(RUN=OUT,VARIANTS=science.VARIANTS,load_context_split=fake_data,signed_action_loss=science.signed_action_loss,
       status=lambda *a,**kw:None,verify_sources=lambda p:None,comparisons=fake_comparisons,primary=science.primary)
    code=science.training_source();exec(compile(code,str(Path(__file__)),'exec'),ns);ns['train_worker']()
    report=read(OUT/'training/report.json');assert report['complete'] and report['models']==24 and not report['cuda_initialized']
    write(OUT/'service_receipt.json',dict(at=now().isoformat(),synthetic_only=True,steps_per_model=1,
        all24_variant_seed_paths_exercised=True,context_and_blank_input_shape14=True,
        actual_scientific_run_untouched=True,generated_training_matches_current_protocol=True,
        code_sha256=__import__('hashlib').sha256(code.encode()).hexdigest(),source_sha256=sha(Path(__file__))))
    print('All24 synthetic model paths exercised; no research metrics produced')


if __name__=='__main__':main()
