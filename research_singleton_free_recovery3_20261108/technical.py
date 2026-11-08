"""Compare native eight-column outputs by the archive's per-frame offsets."""
import os,sys
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_singleton_free_20261108.common import *


def main(obs):
    import numpy as np,torch
    phase();phase('recovery3_phase.json');torch.set_num_threads(2)
    prior=ROOT/'runs/flyvis_repairs_20261106/unification_20261107/uncapped_checks/e308709bd5a9460eb49e0aca77845ab5/memory_guard.worker.log'
    assert '5 passed' in prior.read_text(encoding='utf-8')
    assert sha(ROOT/'tests/test_free_singleton_assignment.py')==read(RUN/'phase.json')['sources']['tests/test_free_singleton_assignment.py']
    from tracking.free_singleton_assignment import FreeSingletonAssociation
    from research_identity_assignment_run_20261108.replay import ReplayTracker,Detections
    from tracking.offline_association import make_tracker_args
    class Zero(FreeSingletonAssociation,ReplayTracker):pass
    video=read(BASE/'zero_phase.json')['video'];cfg=read(ROOT/'research_identity_assignment_protocol_20261108/protocol_v2.json')['native_config']
    with np.load(ROOT/f'runs/train_cache/{video}/detections/raw_video.npz',allow_pickle=False) as z:raw={n:z[n] for n in z.files}
    with np.load(BASE/'zero_native_outputs.npz',allow_pickle=False) as z:expected=z['outputs'];offsets=z['offsets']
    called=[];tracker=Zero(make_tracker_args(**cfg),alpha=0,beta=0,arm='U_FLYVIS',provider=lambda *a:called.append(1))
    order=np.argsort(raw['frame_idx'],kind='stable');frames=raw['frame_idx'][order];actual=[];lengths=[];first_mismatch=None
    for f in range(len(offsets)-1):
        lo,hi=np.searchsorted(frames,[f,f+1]);ids=order[lo:hi]
        out=tracker.update(Detections(raw['xyxy'][ids],raw['conf'][ids],raw['cls'][ids],ids));out=np.asarray(out)
        lengths.append(len(out))
        if len(out):actual.append(out.astype(np.float32))
        baseline=expected[offsets[f]:offsets[f+1]]
        if first_mismatch is None and (len(out)!=len(baseline) or (len(out) and not np.array_equal(out,baseline))):first_mismatch=f
    actual=np.concatenate(actual) if actual else np.zeros((0,8),np.float32);actual_offsets=np.r_[0,np.cumsum(lengths)]
    saved=RUN/'zero_native_outputs_recovery3.npz'
    with saved.open('xb') as stream:np.savez_compressed(stream,outputs=actual,offsets=actual_offsets)
    assert first_mismatch is None and np.array_equal(actual,expected) and np.array_equal(actual_offsets,offsets) and not called,first_mismatch
    from research_singleton_free_20261108.simulate import Simulator,FreeReplay
    import research_identity_assignment_run_20261108.replay as replay
    witness=min((r for r in read(BASE/'singleton_inventory_fit20_recovery1.json')['rows'] if r['group']=='free_first_stage'),key=lambda r:r['uid'])
    sim=Simulator(witness['video']);sim.directory=RUN/'technical_counterfactual'/witness['video'];sim.directory.mkdir(parents=True,exist_ok=True)
    frame=str(witness['frame']);snap=sim.export['snapshots'][frame]
    expected_update=sim.evaluate('U_GEO',.2,0,[witness['frame']])['frames'][frame]
    plan=dict(sim.score['frames'][frame]);plan['_native_cost']=sim.matrices['cost_'+frame]
    original=replay.ReplayTracker;replay.ReplayTracker=FreeReplay
    try:actual_update=replay.apply_state(sim.blob(frame),sim.cfg,sim.inputs(frame),snap,plan,.2,0,'U_GEO')
    finally:replay.ReplayTracker=original
    assert expected_update['final']==actual_update['final'] and expected_update['all_final']==actual_update['all_final']
    assert np.array_equal(expected_update['output'],actual_update['output']);sim.close()
    assert not torch.cuda.is_initialized() and not any(n=='flyvis' or n.startswith('flyvis.') for n in sys.modules)
    phase();phase('recovery3_phase.json');write(RUN/'technical.json',dict(passed=True,tests=5,tests_reused_log_sha256=sha(prior),
        zero_frames=len(offsets)-1,zero_native_eight_columns_and_offsets_exact=True,provider_calls=len(called),core_imported=False,
        native_raw_source_sha256=sha(BASE/'zero_native_outputs.npz'),actual_native_sha256=sha(saved),
        cached_complete_update_witness_uid=witness['uid'],cached_update_new_runtime_exact=True,
        corrected_comparator='Archive contains native[x1,y1,x2,y2,ID,confidence,class,local_detection_index], with frame boundaries in offsets. No prepended frame column.'),new=True)


if __name__=='__main__':main(guard())
