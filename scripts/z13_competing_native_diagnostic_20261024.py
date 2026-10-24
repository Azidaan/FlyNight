"""Reproduce original D/seed0 training directly from real cached rows, no replay."""
from pathlib import Path
import os,sys,time,faulthandler
os.environ['CUDA_VISIBLE_DEVICES']='-1';os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));faulthandler.enable(all_threads=True)
from scripts import z13_competing_candidates_20261024 as S


def main():
    import numpy as np,torch
    torch.set_default_device('cpu');torch.set_num_threads(2)
    from tracking.z13_candidate_ranker import make_ranker
    from scripts.z13_candidate_ranker_20261021 import pack
    import scripts.z13_candidate_identity_20261021 as reader
    l20=S.read(S.CACHE/'fit/labels.json');extra=S.read(S.OLD/'evaluate/labels.json');labels=l20+extra
    seq,_=reader.load('fit','D');a=pack(l20,seq);reader.CACHE=S.OLD;seq,_=reader.load('evaluate','D');b=pack(extra,seq)
    ta=tuple(np.concatenate([x,y]) for x,y in zip(a,b));ids=np.asarray([i for i,l in enumerate(labels) if l['kind']>=0]);nref=sum(l['kind']>=0 for l in l20)
    t=[torch.as_tensor(x[ids],device='cpu') for x in ta[:5]]
    if '--clone' in sys.argv:
        copied=[v.clone() for v in t];assert all(torch.equal(a,b) for a,b in zip(t,copied));t=copied
    if '--no-mkldnn' in sys.argv:torch.backends.mkldnn.enabled=False
    ref=S.read(S.OLD/'D/activity/seed0/training.json');torch.manual_seed(0);model=make_ranker(torch,True)
    if '--trace-backward' in sys.argv:
        original=torch.Tensor.backward
        def backward(tensor,*args,**kwargs):
            seen=set();stack=[tensor.grad_fn]
            while stack:
                node=stack.pop()
                if node is None or node in seen:continue
                seen.add(node);name=node.name()
                node.register_prehook(lambda gradients,name=name:(print('BACKWARD ENTER',name,flush=True),None)[1])
                node.register_hook(lambda *unused,name=name:(print('BACKWARD EXIT',name,flush=True),None)[1])
                stack.extend(n for n,_ in node.next_functions)
            return original(tensor,*args,**kwargs)
        torch.Tensor.backward=backward
    out=S.RUN/'native_diagnostic';S.write(out/'inputs.json',dict(at=S.now().isoformat(),tensors=[dict(shape=list(x.shape),dtype=str(x.dtype),contiguous=x.is_contiguous(),finite=bool(torch.isfinite(x).all())) for x in t],without_prior_inference=True))
    began=time.monotonic();loss,_=S.P.train_steps(torch,model,t,nref,0,ref['pos_weight'],maxsteps=32,progress=lambda step:print('Original step',step,flush=True));delta=max(abs(a-b) for a,b in zip(loss,ref['loss'][:32]));assert delta<=1e-6
    S.write(out/'report.json',dict(at=S.now().isoformat(),original32_max_abs=delta,seconds=time.monotonic()-began,passed=True,without_prior_inference=True));print('Original32 fresh-process replay passed',delta,flush=True)


if __name__=='__main__':main()
