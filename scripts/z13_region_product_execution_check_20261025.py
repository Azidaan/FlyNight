"""Service preflight: all generated worker paths on synthetic inputs only."""
from pathlib import Path
from datetime import timedelta
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import z13_region_product_20261025 as science
from scripts.z13_candidate_reserve_20261021 import write,read,now,sha
OUT=ROOT/'runs/z13_region_product_execution_check_20261025'


def main():
    import torch
    assert not OUT.exists();torch.set_num_threads(2);torch.set_default_device('cpu')
    p=dict(steps=1,batch=128,seeds=[0,1,2],fit20=['synthetic'],fit84=['synthetic'],evaluate26=['synthetic'],save_at=(now()+timedelta(seconds=600)).isoformat())
    write(OUT/'training/protocol.json',p)
    def fake(torch,arm,split,mode):
        n=130;mask=torch.ones(n,2,dtype=torch.bool)
        d=dict(x=torch.ones(n,2,68),seed=torch.ones(n,68),geo=torch.zeros(n,2,82),mask=mask,
            uid=torch.arange(n),kind=torch.ones(n,dtype=torch.int8),video=torch.zeros(n,dtype=torch.int16),good=torch.zeros_like(mask),harm=torch.zeros_like(mask))
        d['good'][0,1]=True;d['harm'][1,0]=True
        if mode=='product':d['geo'][...,14:]=.1
        return d
    def comparisons(means,videos):return {k:dict(mean_per_video=0.,ci95=[-1.,1.],n=1) for k in ['Dproduct-baseline','Dproduct-D_blank','Dproduct-C_product','Dproduct-R1_product','Dproduct-geometry_product','Dproduct-Dprevious']}
    ns=science.context.original.namespace();ns.update(RUN=OUT,VARIANTS=science.VARIANTS,load_product_split=fake,
        signed_action_loss=science.context.signed_action_loss,status=lambda *a,**kw:None,verify_sources=lambda p:None,comparisons=comparisons,primary=science.primary)
    code=science.executed_source();exec(compile(code,str(Path(__file__)),'exec'),ns);ns['train_worker']()
    report=read(OUT/'training/report.json');assert report['models']==24 and report['complete'] and not report['cuda_initialized']
    write(OUT/'receipt.json',dict(at=now().isoformat(),all24_synthetic_variant_seed_paths_passed=True,
        steps_per_model=1,real_inputs162=True,research_metrics_produced=False,code_sha256=__import__('hashlib').sha256(code.encode()).hexdigest(),source_sha256=sha(Path(__file__))))
    print('All24 synthetic CPU forward/backward/save/calibration paths passed')


if __name__=='__main__':main()
