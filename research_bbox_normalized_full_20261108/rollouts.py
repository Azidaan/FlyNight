import os,sys,shutil
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_bbox_normalized_full_20261108.common import *


def main(obs):
    import numpy as np
    from research_bbox_normalized_signals_20261108.compare import simulate
    from research_identity_assignment_run_20261108.full_run import scalabel
    full_phase();p=read(FULL/'protocol.json');assert read(FULL/'features_complete.json')['passed'];receipts={}
    for video,entry in p['videos'].items():
        metadata=read(FULL/'native_trace'/(video+'.json'))
        with np.load(FULL/'native_trace'/(video+'.npz'),allow_pickle=False) as z:native=z['outputs']
        scores=read(FULL/'scores'/(video+'.json'))['edges'];anchors=read(FULL/'runtime_inputs'/(video+'.json'))['anchors'];cached_outputs={};baseline=read(ROOT/entry['baseline'])
        for arm,cfg in p['variants'].items():
            out,decisions=simulate(native,metadata,scores,cfg);output=[scalabel(video,anchor,out[out[:,0]==int(frame),1:]) for frame,anchor in anchors.items()];folder=FULL/'rollouts'/arm;folder.mkdir(parents=True,exist_ok=True);raw=folder/(video+'.npz');jsonpath=FULL/'outputs'/arm/(video+'.json');eventpath=folder/(video+'.decisions.json');jsonpath.parent.mkdir(parents=True,exist_ok=True)
            if arm=='B0':assert np.array_equal(out,native) and output==baseline;shutil.copyfile(ROOT/entry['raw_array'],raw)
            else:
                with raw.open('xb') as f:np.savez_compressed(f,outputs=out[:,:8],frame_count=entry['frames'])
            if output==baseline:shutil.copyfile(ROOT/entry['baseline'],jsonpath)
            else:write(jsonpath,output,new=True)
            write(eventpath,dict(decisions=decisions),new=True);receipt=folder/(video+'.receipt.json');write(receipt,dict(passed=True,physical_outputs_exact=True,video=video,arm=arm,links=sum(len(d['links']) for d in decisions),revoked=sum(len(d['revoked']) for d in decisions),artifacts={n.relative_to(ROOT).as_posix():sha(n) for n in [raw,jsonpath,eventpath]}),new=True);receipts[arm+'/'+video]=sha(receipt)
        status('normalized_full_fixed_identity_policies',video=video,complete=list(p['videos']).index(video)+1,total=len(p['videos']),guard=obs.read())
    full_phase();write(FULL/'rollouts_complete.json',dict(passed=True,receipts=receipts,physical_outputs_exact=True),new=True)


if __name__=='__main__':main(guard())
