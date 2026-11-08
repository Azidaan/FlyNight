"""New-class zero rollout and meaningful native-lock tests before calibration."""
import os,sys
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_singleton_free_20261108.common import *


def main(obs):
    phase()
    import numpy as np,torch,pytest
    from tracking.free_singleton_assignment import FreeSingletonAssociation
    from research_identity_assignment_run_20261108.replay import ReplayTracker,Detections
    from tracking.offline_association import make_tracker_args
    torch.set_num_threads(2)
    assert pytest.main(['-q','tests/test_free_singleton_assignment.py','--basetemp',str(RUN/'pytest_temp')])==0
    class Zero(FreeSingletonAssociation,ReplayTracker):pass
    p=read(BASE/'zero_phase.json');video=p['video'];cfg=read(ROOT/'research_identity_assignment_protocol_20261108/protocol_v2.json')['native_config']
    with np.load(ROOT/f'runs/train_cache/{video}/detections/raw_video.npz',allow_pickle=False) as z:raw={n:z[n] for n in z.files}
    with np.load(BASE/'zero_native_outputs.npz',allow_pickle=False) as z:expected={n:z[n] for n in z.files}
    called=[]
    tracker=Zero(make_tracker_args(**cfg),alpha=0,beta=0,arm='U_FLYVIS',provider=lambda *a:called.append(1))
    outputs=[];order=np.argsort(raw['frame_idx'],kind='stable');frames=raw['frame_idx'][order]
    count=len(expected['offsets'])-1 if 'offsets' in expected else int(expected['frame_count'])
    for f in range(count):
        lo,hi=np.searchsorted(frames,[f,f+1]);ids=order[lo:hi]
        out=tracker.update(Detections(raw['xyxy'][ids],raw['conf'][ids],raw['cls'][ids],ids))
        if len(out):outputs.append(np.column_stack([np.full(len(out),f),out[:,:7]]).astype(np.float32))
    actual=np.concatenate(outputs) if outputs else np.zeros((0,8),np.float32)
    assert np.array_equal(actual,expected['outputs']) and not called
    from research_singleton_free_20261108.simulate import Simulator,FreeReplay
    import research_identity_assignment_run_20261108.replay as replay
    inventory=read(BASE/'singleton_inventory_fit20_recovery1.json')['rows']
    witness=min((r for r in inventory if r['group']=='free_first_stage'),key=lambda r:r['uid'])
    sim=Simulator(witness['video']);sim.directory=RUN/'technical_counterfactual'/witness['video'];sim.directory.mkdir(parents=True,exist_ok=True)
    frame=str(witness['frame']);snap=sim.export['snapshots'][frame]
    expected_update=sim.evaluate('U_GEO',.2,0,[witness['frame']])['frames'][frame]
    plan=dict(sim.score['frames'][frame]);plan['_native_cost']=sim.matrices['cost_'+frame]
    original=replay.ReplayTracker;replay.ReplayTracker=FreeReplay
    try:actual_update=replay.apply_state(sim.blob(frame),sim.cfg,sim.inputs(frame),snap,plan,.2,0,'U_GEO')
    finally:replay.ReplayTracker=original
    assert expected_update['final']==actual_update['final'] and expected_update['all_final']==actual_update['all_final']
    assert np.array_equal(expected_update['output'],actual_update['output'])
    sim.close()
    assert not torch.cuda.is_initialized() and not any(n=='flyvis' or n.startswith('flyvis.') for n in sys.modules)
    phase();write(RUN/'technical.json',dict(passed=True,tests=5,zero_frames=count,zero_raw_arrays_exact=True,
        provider_calls=len(called),core_imported=False,native_raw_source_sha256=sha(BASE/'zero_native_outputs.npz'),
        cached_complete_update_witness_uid=witness['uid'],cached_update_new_runtime_exact=True) ,new=True)


if __name__=='__main__':main(guard())
