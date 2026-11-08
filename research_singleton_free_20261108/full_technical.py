"""Exact scoring parity for all active controls and verified old-map reuse."""
import os,sys
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_singleton_free_20261108.common import *


def main(obs):
    import numpy as np,torch
    phase('full_phase.json');torch.set_num_threads(2)
    from research_singleton_free_20261108.full_planner import Planner
    from research_singleton_free_20261108.full_features import FULL,OLD,Representations,Receptors
    from research_identity_assignment_run_20261108.full_retina import Receptors as OriginalReceptors
    from research_identity_assignment_run_20261108.export_states import sequence_key
    witness=read(BASE/'full/technical_gate.json')['first_causal_calibration_snapshot'];video=witness['video'];frame=str(witness['frame'])
    item=read(BASE/'export_complete.json')['videos'][video];snap=read(ROOT/item['path'])['snapshots'][frame]
    with np.load((ROOT/item['path']).with_suffix('.npz'),allow_pickle=False) as z:C=z['cost_'+frame]
    scores=read(BASE/'scores'/(video+'.json'))['frames'][frame]['columns'];image_norm=read(ROOT/'runs/spatial_identity_maps_20261107/image_normalization.json')
    features=read(BASE/'features_complete.json')['files']
    class Archive:
        def __init__(self,kind):self.kind=kind
        def get(self,sequence,now):
            key=sequence_key(video,sequence);path=BASE/'features'/(key+'.npz');assert sha(path)==features[key]
            with np.load(path,allow_pickle=False) as z:value=z[self.kind]
            if self.kind=='IMAGE':value=(value.astype(float)-image_norm['mean'])/np.where(np.array(image_norm['sd'])>=1e-12,image_norm['sd'],1)
            return value
    differences={}
    for arm,kind in [('U_IMAGE','IMAGE'),('U_R1','R1'),('U_DIS','DIS'),('U_FLYVIS','C'),('U_PERM','C')]:
        planner=Planner(video,arm,Archive(kind));plan=planner.build(snap['rows'],np.array(snap['boxes']),np.array(snap['confidence']),np.array(snap['classes']),C,int(frame))
        assert plan['votes']==snap['votes'];errors=[];mapped=arm.replace('U_','G_',1)
        for j,col in plan['columns'].items():
            old=scores[j];assert col['rows']==old['rows'] and col['geometry_valid']==old['geometry_valid']
            if col['geometry_valid']:
                assert col['support']==old['support'];a=col['scores'][mapped];b=old['scores'][mapped]
                assert [v is None for v in a]==[v is None for v in b]
                errors.extend(abs(x-y) for x,y in zip(a,b) if x is not None)
        differences[arm]=max(errors,default=0);assert differences[arm]<1e-12
    p=read(FULL/'protocol.json');first=next(iter(p['videos']));old=read(OLD/'cache_reuse'/(first+'.json'))
    choices=[(name,value) for name,value in old['features_reused'].items() if name.endswith('.C.npy')]
    assert choices;name,value=min(choices);rep=Representations(first,'C',obs,None);seq=value['sequence'];actual=rep.get(seq,max(f for f,c in seq))
    expected=np.load(ROOT/name,allow_pickle=False);assert np.array_equal(actual,expected) and rep.new==0 and rep.reused
    assert Receptors.render is OriginalReceptors.render
    assert not torch.cuda.is_initialized() and not any(n=='flyvis' or n.startswith('flyvis.') for n in sys.modules)
    phase('full_phase.json');write(FULL/'technical.json',dict(passed=True,score_max_errors=differences,
        cached_map_exact=True,cached_map_source=name,new_core_calls=0,original_input_filter_method_reused=True,
        native_precision_gate_sha256=sha(OLD/'technical_gate.json')),new=True)


if __name__=='__main__':main(guard())
