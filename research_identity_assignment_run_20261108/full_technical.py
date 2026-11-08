"""Technical witnesses for the model-free input filter and identical spatial costs."""
import os,sys
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from research_identity_assignment_run_20261108.common import *
from research_identity_assignment_run_20261108.full_retina import FULL,Receptors


def main(obs):
    import torch,cv2
    from types import SimpleNamespace
    from tracking.flyvis_adapter import activate_flyvis
    from tracking.foveated_eyes import EyeBank
    from research_identity_assignment_run_20261108.full_planner import Planner,Representations
    verify_phase('full_technical_phase.json');torch.set_num_threads(2);activate_flyvis(device='cpu')
    from flyvis.datasets.rendering import BoxEye
    eye=BoxEye(extent=15,kernel_size=13)
    bank=SimpleNamespace(torch=torch,eye_size=391,eye=eye,estimator=SimpleNamespace(flyvis=SimpleNamespace(device=torch.device('cpu'))))
    renderer=Receptors.__new__(Receptors);renderer.torch=torch
    renderer.points=np.array(read(ROOT/'runs/z13_local_peak_20261021/geometry.json')['points'],int)
    renderer.px=torch.from_numpy(renderer.points[:,0].copy());renderer.py=torch.from_numpy(renderer.points[:,1].copy())
    renderer.kernel=torch.ones((1,1,13,13),dtype=torch.float32,device='cpu');renderer.bias=torch.zeros(1,dtype=torch.float32,device='cpu')
    rng=np.random.default_rng(42);gray=(rng.random((720,1280))*255).astype(np.float32)
    crops=[[0,0,64,64],[315,122,515,322],[560,0,1280,720]];errors=[]
    for crop in crops:
        with torch.inference_mode():expected=EyeBank._render(bank,gray,[crop]).cpu().numpy().reshape(721).astype(np.float16)
        value=renderer.render(gray,crop);assert np.array_equal(value,expected)
        errors.append(float(np.max(np.abs(value.astype(float)-expected.astype(float)))))
    # Select the first eligible calibration snapshot by causal metadata, never by a score or outcome.
    exports=read(RUN/'export_complete.json')['videos'];chosen=None
    for video,entry in sorted(exports.items()):
        data=read(ROOT/entry['path']);scores=read(RUN/'scores'/(video+'.json'))['frames']
        for frame,snap in sorted(data['snapshots'].items(),key=lambda kv:int(kv[0])):
            if not any(q['split']=='fit20' for q in snap['queries']):continue
            if any(c['geometry_valid'] for c in scores[frame]['columns'].values()):chosen=(video,entry,data,scores,frame,snap);break
        if chosen:break
    assert chosen;video,entry,data,scores,frame,snap=chosen;differences={}
    with np.load((ROOT/entry['path']).with_suffix('.npz'),allow_pickle=False) as z:C=z['cost_'+frame]
    for arm in ['G_R1','G_FLYVIS','G_PERM']:
        rep=Representations(video,'R1' if arm=='G_R1' else 'C',archive=RUN/'features')
        planner=Planner(video,arm,rep);plan=planner.build(snap['rows'],np.array(snap['boxes']),np.array(snap['confidence']),np.array(snap['classes']),C,int(frame))
        assert plan['votes']==snap['votes'];errors=[]
        for j,col in plan['columns'].items():
            old=scores[frame]['columns'][j];assert col['rows']==old['rows'] and col['geometry_valid']==old['geometry_valid']
            if col['geometry_valid']:
                assert col['support']==old['support'];a=col['scores'][arm];b=old['scores'][arm]
                assert [v is None for v in a]==[v is None for v in b]
                errors.extend(abs(x-y) for x,y in zip(a,b) if x is not None)
        differences[arm]=max(errors,default=0);assert differences[arm]<1e-12
    assert not torch.cuda.is_initialized();verify_phase('full_technical_phase.json')
    write(FULL/'technical_gate.json',dict(passed=True,renderer_witnesses=len(crops),renderer_half_exact=True,
        first_causal_calibration_snapshot=dict(video=video,frame=int(frame)),spatial_score_max_errors=differences,
        scope='Input filtering and active-arm scoring match the archived implementation; no full-tracker scientific result yet.'),new=True)


if __name__=='__main__':main(guard())
