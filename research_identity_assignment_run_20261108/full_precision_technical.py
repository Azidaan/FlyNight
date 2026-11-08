"""Independent native float64/float32 renderer and lossless container witness."""
import os,sys
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_identity_assignment_run_20261108.common import *


def main(obs):
    verify_phase('full_precision_technical_phase.json')
    import cv2,torch,numpy as np
    from types import SimpleNamespace
    from tracking.retina import make_retina
    from tracking.flyvis_adapter import activate_flyvis
    from tracking.foveated_eyes import EyeBank
    from research_identity_assignment_run_20261108.full_precision_io import FULL,Receptors,FrameStore
    activate_flyvis(device='cpu');torch.set_num_threads(2)
    from flyvis.datasets.rendering import BoxEye
    eye=BoxEye(extent=15,kernel_size=13)
    bank=SimpleNamespace(torch=torch,eye_size=391,eye=eye,estimator=SimpleNamespace(flyvis=SimpleNamespace(device=torch.device('cpu'))))
    renderer=Receptors.__new__(Receptors);renderer.torch=torch
    points=np.array(read(ROOT/'runs/z13_local_peak_20261021/geometry.json')['points'],int)
    renderer.px=torch.from_numpy(points[:,0].copy());renderer.py=torch.from_numpy(points[:,1].copy())
    renderer.kernel=torch.ones((1,1,13,13),dtype=torch.float32);renderer.bias=torch.zeros(1,dtype=torch.float32)
    p=read(FULL/'protocol.json');video=next(iter(p['videos']));e=p['videos'][video]
    cap=cv2.VideoCapture(str(ROOT/e['video_path']));retina=make_retina('lamina',e['fps'],blur_downscale=4)
    native=[]
    for _ in range(3):
        ok,img=cap.read();assert ok;native.append(retina.process(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)))
    cap.release();dtypes=[str(x.dtype) for x in native]
    path=FULL/'technical_container.float64';path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as f:
        for value in native:f.write(np.asarray(value,dtype=np.float64,order='C').tobytes())
    store=FrameStore(path,(3,720,1280),dtypes);errors=[]
    for i,value in enumerate(native):
        restored=store[i];assert restored.dtype==value.dtype and np.array_equal(restored,value)
        for crop in [[0,0,64,64],[315,122,515,322],[560,0,1280,720]]:
            with torch.inference_mode():expected=EyeBank._render(bank,value,[crop]).cpu().numpy().reshape(721).astype(np.float16)
            actual=renderer.render(restored,crop);assert np.array_equal(expected,actual)
            errors.append(float(np.max(abs(expected.astype(float)-actual.astype(float)))))
    assert not torch.cuda.is_initialized();verify_phase('full_precision_technical_phase.json')
    write(FULL/'technical_gate.json',dict(passed=True,native_dtypes=dtypes,renderer_witnesses=len(errors),all_input_half_exact=True,
        dtype_restore_exact=True,max_error=max(errors),original_scoring_gate_sha256=sha(RUN/'full/technical_gate.json'),
        scope='Corrected IO only. Scientific planner, scoring, tracker, parameters and evaluation unchanged.'),new=True)


if __name__=='__main__':main(guard())
