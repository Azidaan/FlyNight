"""Synthetic CPU backward smoke check, no project data or scientific fitting."""
import argparse
import faulthandler
import os
from pathlib import Path
import sys
faulthandler.enable(all_threads=True)
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
ap=argparse.ArgumentParser();ap.add_argument('--threads',type=int,default=2);ap.add_argument('--no-mkldnn',action='store_true');ap.add_argument('--cuda-mask',choices=['negative','empty','unset'],default='negative');a=ap.parse_args()
if a.cuda_mask=='unset':os.environ.pop('CUDA_VISIBLE_DEVICES',None)
else:os.environ['CUDA_VISIBLE_DEVICES']='' if a.cuda_mask=='empty' else '-1'
import torch
from tracking.z13_candidate_ranker import make_ranker
torch.set_default_device('cpu');torch.set_num_threads(a.threads);torch.manual_seed(0)
if a.no_mkldnn:torch.backends.mkldnn.enabled=False
m=make_ranker(torch,True);o=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.01)
x=torch.randn(128,3,68);s=torch.randn(128,68);g=torch.randn(128,3,8);mask=torch.ones(128,3,dtype=torch.bool);y=torch.zeros(128,3)
for i in range(32):
    z=m(x,s,g,mask);loss=torch.nn.functional.binary_cross_entropy_with_logits(z,y,pos_weight=torch.tensor(20.))
    print('backward step',i,flush=True)
    o.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),10.);o.step()
print('CPU backward OK',a.threads,a.no_mkldnn,float(loss.detach()))
