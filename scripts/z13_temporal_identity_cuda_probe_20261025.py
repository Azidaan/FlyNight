"""Fresh-process CUDA transfer/backward probe after a preserved stalled context."""
from pathlib import Path
from datetime import datetime,timedelta,timezone
import json,os,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
RUN=ROOT/'runs/z13_temporal_identity_20261025/cuda_probe'


def main():
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    import numpy as np
    import torch
    from tracking.gpu_lock import GpuLock
    from scripts.z13_temporal_pair_readiness_20261025 import write,now
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    before=time.monotonic();torch.set_num_threads(2)
    with GpuLock('flyvis_fovea_gpu'):
        x=torch.tensor([1.,2.,3.],device='cuda:0',requires_grad=True)
        for _ in range(10):
            index=torch.as_tensor(np.arange(3),device='cuda:0')
            y=(x[index]**2).sum();y.backward();assert torch.equal(x.grad,2*x.detach());x.grad=None
        torch.cuda.synchronize()
        write(RUN/'report.json',dict(complete=True,device=str(x.device),iterations=10,gradient_and_host_device_transfer_verified=True,
            seconds=time.monotonic()-before,at=now().isoformat()))
        write(RUN/'status.json',dict(state='completed',updated=now().isoformat()))


if __name__=='__main__':
    if '--prepare' in sys.argv:
        RUN.mkdir(parents=True,exist_ok=False)
        start=datetime.now(timezone(timedelta(hours=3)))
        (RUN/'protocol.json').write_text(json.dumps(dict(device='cuda',budget_seconds=50,save_at=(start+timedelta(seconds=50)).isoformat(),hard_stop=(start+timedelta(seconds=60)).isoformat(),stop_request_path=(RUN/'stop.request').relative_to(ROOT).as_posix()),indent=1),encoding='utf8')
        (RUN/'status.json').write_text(json.dumps(dict(state='ready',updated=start.isoformat())),encoding='utf8')
    elif '--worker' in sys.argv:
        main()
