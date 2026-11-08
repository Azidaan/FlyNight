"""Apply every fixed ID-only policy to the same proved native outputs."""
import os,sys,shutil
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_native_confirm_full_20261108.common import *


def main(obs):
    import numpy as np
    from research_native_confirm_risk_20261108.simulate import simulate
    from research_identity_assignment_run_20261108.full_run import scalabel
    full_phase();p=read(FULL/'protocol.json');assert read(FULL/'features_complete.json')['passed'];manifest={}
    for number,(video,entry) in enumerate(p['videos'].items()):
        metadata=read(FULL/'native_trace'/(video+'.json'))
        with np.load(FULL/'native_trace'/(video+'.npz'),allow_pickle=False) as z:native=z['outputs']
        with np.load(ROOT/entry['raw_array'],allow_pickle=False) as z:assert np.array_equal(native[:,:8],z['outputs']) and int(z['frame_count'])==entry['frames']
        scores=read(FULL/'scores'/(video+'.json'))['edges'];anchors=read(FULL/'runtime_inputs'/(video+'.json'))['anchors']
        for arm in p['parameters']:
            folder=FULL/'rollouts'/arm;folder.mkdir(parents=True,exist_ok=True);receipt=folder/(video+'.receipt.json')
            if receipt.exists():
                r=read(receipt)
                for n,h in r['artifacts'].items():assert sha(ROOT/n)==h
                manifest[f'{arm}/{video}']=sha(receipt);continue
            out,decisions=simulate(native,metadata,scores,p['variants'][arm]);raw=out[:,:8];output=[scalabel(video,anchor,out[out[:,0]==int(frame),1:]) for frame,anchor in anchors.items()]
            if arm=='B0' or p['thresholds'].get(arm)=='off':assert np.array_equal(out,native) and output==read(ROOT/entry['baseline'])
            rawpath=folder/(video+'.npz');jsonpath=FULL/'outputs'/arm/(video+'.json');eventpath=folder/(video+'.decisions.json');write(jsonpath,output,new=True);write(eventpath,dict(decisions=decisions),new=True)
            if arm=='B0':shutil.copyfile(ROOT/entry['raw_array'],rawpath)
            else:
                with rawpath.open('xb') as f:np.savez_compressed(f,outputs=raw,frame_count=entry['frames'])
            r=dict(passed=True,video=video,arm=arm,frames=entry['frames'],physical_outputs_exact=True,native_states_and_associations_unchanged=True,links=sum(len(d['links']) for d in decisions),revoked=sum(len(d['revoked']) for d in decisions),artifacts={n.relative_to(ROOT).as_posix():sha(n) for n in [rawpath,jsonpath,eventpath]});write(receipt,r,new=True);manifest[f'{arm}/{video}']=sha(receipt)
        status('native_confirm_full_identity_policies',video=video,complete=number+1,total=len(p['videos']),guard=obs.read())
    full_phase();write(FULL/'rollouts_complete.json',dict(passed=True,receipts=manifest,physical_outputs_exact=True),new=True)


if __name__=='__main__':main(guard())
