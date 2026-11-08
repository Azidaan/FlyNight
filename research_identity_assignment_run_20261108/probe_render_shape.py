import os,sys
from pathlib import Path
from types import SimpleNamespace
os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_identity_assignment_run_20261108.common import *


def main(obs):
    import numpy as np,torch
    from tracking.flyvis_adapter import activate_flyvis
    from tracking.foveated_eyes import EyeBank
    activate_flyvis(device='cpu')
    from flyvis.datasets.rendering import BoxEye
    eye=BoxEye(extent=15,kernel_size=13)
    bank=SimpleNamespace(torch=torch,eye_size=391,eye=eye,estimator=SimpleNamespace(flyvis=SimpleNamespace(device=torch.device('cpu'))))
    with torch.inference_mode():out=EyeBank._render(bank,np.zeros((720,1280),np.float32),[(0,0,64,64)])
    write(RUN/'render_shape_probe.json',dict(render_shape=list(out.shape),after_index0=list(out[:,0].shape),row_shape=list(out[:,0].cpu().numpy()[0].shape)),new=True)
    print(read(RUN/'render_shape_probe.json'))


if __name__=='__main__':main(guard())
