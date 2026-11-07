"""Interface-only BoxEye axis correction; preserve first failure and all numerical gates."""
import os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ["CUDA_VISIBLE_DEVICES"]="";os.environ["OMP_NUM_THREADS"]=os.environ["MKL_NUM_THREADS"]="2"
from scripts import flyvis_input_separation_20261107 as original
sys.path.insert(0,str(original.GUARD))
from system_memory_guard import guard_entrypoint
if __name__=="__main__":
 obs=guard_entrypoint(timeout=4800)
 import torch
 torch.set_default_device("cpu");torch.cuda.is_available=lambda:False
 from tracking.flyvis_adapter import activate_flyvis
 activate_flyvis(device="cpu")
 from flyvis.datasets import rendering
 Base=rendering.BoxEye
 class ShapeAdapter(Base):
  def __call__(self,*args,**kwargs):
   out=super().__call__(*args,**kwargs)
   assert out.ndim==4 and out.shape[2]==1 and out.shape[-1]==721,out.shape
   return out[:,:,0,:]
 rendering.BoxEye=ShapeAdapter
 original.prepare_inputs(obs)
