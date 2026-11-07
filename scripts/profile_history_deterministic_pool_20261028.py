"""Проверить эквивалентность пула и реальный CUDAградиент, не обучениесерии."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='0';os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
from pathlib import Path
import sys,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.continuation_service_20261028 import RUN,service


def main():
    import numpy as np,torch
    from tracking.gpu_lock import GpuLock
    from tracking.history_pair_model import HistoryPairHead,weighted_pair_loss
    from tracking.history_pair_model_deterministic import DeterministicHistoryPairHead
    from tracking.history_pair_features import from_cache
    folder=RUN/'history_deterministic_pool_profile'
    old=RUN/'history_factorial_training';c=service.read(old/'controller_result.json')
    assert c['exit_code']==1 and c['all_owned_ended'] and not any(service.matching(k,v) for k,v in c['owned'].items())
    assert not (old/'models').exists() or not list((old/'models').rglob('*.pt'))
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    v=service.read(RUN/'history_visual_inputs/report.json')['completed'][0]['video'];pair=RUN/'history_pair_dataset/videos'/v;visual=RUN/'history_visual_inputs/videos'/v
    with np.load(pair/'inputs.npz') as z:x={k:z[k] for k in z.files}
    with np.load(pair/'targets.npz') as z:y=z['identity']
    with np.load(visual/'index.npz') as z:index={k:z[k] for k in z.files}
    emb=np.load(visual/'embeddings.npy',mmap_mode='r');images=np.load(visual/'images.npy',mmap_mode='r')
    rows=np.flatnonzero(y>=0)[:32];f=from_cache(x,index,emb,images,rows,'134');cpu={k:torch.from_numpy(v) for k,v in f.items()}
    torch.manual_seed(42);original=HistoryPairHead();fixed=DeterministicHistoryPairHead();fixed.load_state_dict(original.state_dict())
    assert original.state_dict().keys()==fixed.state_dict().keys()
    a=original(cpu);b=fixed(cpu);assert torch.equal(a,b)
    weighted_pair_loss(a,torch.from_numpy(y[rows]),torch.ones(2)).backward()
    weighted_pair_loss(b,torch.from_numpy(y[rows]),torch.ones(2)).backward()
    assert all(torch.equal(p.grad,q.grad) for p,q in zip(original.parameters(),fixed.parameters()))
    with GpuLock('Local\\DissertationSingleGpu'):
        original=original.cuda();fixed=fixed.cuda();gpu={k:v.cuda() for k,v in cpu.items()}
        with torch.no_grad():a=original(gpu);b=fixed(gpu)
        assert torch.equal(a,b),'Forward changed'
        fixed.zero_grad(set_to_none=True)
        loss=weighted_pair_loss(fixed(gpu),torch.from_numpy(y[rows]).cuda(),torch.ones(2,device='cuda'));loss.backward()
        assert all(torch.isfinite(v.grad).all() for v in fixed.parameters())
        gradients=[v.grad.detach().clone() for v in fixed.parameters()]
        fixed.zero_grad(set_to_none=True)
        weighted_pair_loss(fixed(gpu),torch.from_numpy(y[rows]).cuda(),torch.ones(2,device='cuda')).backward()
        assert all(torch.equal(g,p.grad) for g,p in zip(gradients,fixed.parameters()))
    service.write(folder/'report.json',dict(passed=True,cpu_forward_exact=True,cpu_all_parameter_gradients_exact=True,gpu_forward_exact=True,gpu_backward_deterministic_twice=True,parameters=sum(p.numel() for p in fixed.parameters()),rows=len(rows),video=v,old_controller_sha256=service.sha(old/'controller_result.json'),training_repeated=False,scientific_settings_unchanged=True,updated=service.now()))


if __name__=='__main__':main()
