import os,sys
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_singleton_verify_full_20261108.common import *


def main(obs):
    import numpy as np,torch
    phase('full_phase.json');torch.set_num_threads(2)
    from research_singleton_verify_full_20261108.engine import Planner,feature_module
    from research_identity_assignment_run_20261108.full_retina import Receptors as OriginalReceptors
    from tracking.spatial_identity_assignment import geometric_match
    inventory=read(RUN/'inventory_recovery1.json');features=read(RUN/'features_complete.json')['files'];exports=read(BASE/'export_complete.json')['videos']
    norm=read(ROOT/'runs/spatial_identity_maps_20261107/image_normalization.json');maximum={};tested=[]
    for event in inventory['events']:
        if not event['calibration']:continue
        video=event['video'];saved=read(RUN/'scores'/(video+'.json'))['frames'][str(event['frame'])][f"{event['row']},{event['column']}"]
        if saved['scores']['FLYVIS'] is None:continue
        snap=read(ROOT/exports[video]['path'])['snapshots'][str(event['frame'])];row=snap['rows'][event['row']];box=snap['boxes'][event['column']]
        class Archive:
            def __init__(self,kind):self.kind=kind
            def get(self,sequence,now):
                key=sequence_key(video,sequence);path=RUN/'features'/(key+'.npz');assert sha(path)==features[key]
                with np.load(path,allow_pickle=False) as z:value=z[self.kind]
                if self.kind=='IMAGE':value=(value.astype(float)-norm['mean'])/np.where(np.array(norm['sd'])>=1e-12,norm['sd'],1)
                return value
        for kind in ['SUPPORT','RANDOM','IMAGE','R1','DIS','FLYVIS','PERM']:
            planner=Planner(video,kind,-1,Archive('C' if kind in ['FLYVIS','PERM'] else kind))
            transform=geometric_match(row['mean'],row['covariance'],row['history'],row['current_crop'],box,planner.g)
            value,support=planner.own_value(row,box,event['frame'],event['column'],transform);expected=saved['scores'][kind]
            assert support==saved['support'] and (value is None)==(expected is None)
            error=0 if value is None else abs(value-expected);assert error<1e-12,(video,kind,error)
            maximum[kind]=max(maximum.get(kind,0),error)
        tested.append(dict(video=video,frame=event['frame'],row=event['row']))
        if len(tested)==3:break
    assert len(tested)==3 and feature_module.Receptors.render is OriginalReceptors.render
    assert not torch.cuda.is_initialized() and not any(n=='flyvis' or n.startswith('flyvis.') for n in sys.modules)
    phase('full_phase.json');write(FULL/'technical.json',dict(passed=True,score_max_errors=maximum,witnesses=tested,
        common_input_filter_exact_method=True,new_core_calls=0,conditional_feature_check_sha256=sha(RUN/'first_new_feature_check.json')) ,new=True)


if __name__=='__main__':main(guard())
