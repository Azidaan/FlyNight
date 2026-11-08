"""Pool exact ready maps and retinal inputs; no model or image execution."""
import os,sys,shutil
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from collections import defaultdict,Counter
from research_bbox_spatial_signals_20261108.common import *


def main(obs):
    import numpy as np
    phase();inventory=read(READINESS/'inventory.json');assert inventory['input_restore_gate']['proceed'];requests={};sources={};counts=Counter();pool=RUN/'reuse_pool';(pool/'features').mkdir(parents=True,exist_ok=True)
    for key,r in inventory['requests'].items():
        value={k:v for k,v in r.items() if k!='ready'}
        if r['ready']:
            source=ROOT/r['ready']['path'];assert sha(source)==r['ready']['sha256'];shutil.copyfile(source,pool/'features'/(key+'.npz'));value['reuse_parent_sha256']=sha(source);sources[r['ready']['path']]=sha(source);counts['ready_sequence_maps']+=1
        else:counts['missing_'+r['role']+'_maps']+=1
        requests[key]=value
    grouped=defaultdict(list)
    for r in requests.values():grouped[r['video']].append(r)
    for video in grouped:
        values={}
        for prefix in ['identity_assignment_20261108','singleton_verify_20261108','native_confirm_policy_20261108']:
            folder=ROOT/'runs'/prefix/'retinal';marker=folder/(video+'.json');npz=folder/(video+'.npz')
            if not marker.exists():continue
            assert sha(npz)==read(marker)['npz_sha256'];sources[npz.relative_to(ROOT).as_posix()]=sha(npz)
            with np.load(npz,allow_pickle=False) as z:
                for frame,crop,h in zip(z['frames'],z['crops'],z['hex']):
                    point=(int(frame),tuple(map(int,crop)))
                    if point in values:assert np.array_equal(values[point],h)
                    else:values[point]=h.copy()
        keys=sorted(values);folder=pool/'retinal';folder.mkdir(exist_ok=True);npz=folder/(video+'.npz')
        with npz.open('xb') as f:np.savez_compressed(f,frames=np.asarray([p[0] for p in keys]),crops=np.asarray([p[1] for p in keys]).reshape(-1,4),hex=np.stack([values[p] for p in keys]) if keys else np.empty((0,721),np.float16))
        write(folder/(video+'.json'),dict(npz_sha256=sha(npz)),new=True)
        for n in [f'runs/z13_data/{video}.npz',f'data/bdd100k/videos/train/{video}.mov',f'runs/train_cache/{video}/motion/flyvis_tiled.json',f'runs/train_cache/{video}/motion/dis_quarter.npz',f'runs/train_cache/{video}/motion/dis_quarter.json']:sources[n]=sha(ROOT/n)
        status('spatial_ready_data_pool',video=video,complete=list(grouped).index(video)+1,total=len(grouped),counts=dict(counts),guard=obs.read())
    write(RUN/'inventory_recovery1.json',dict(passed=True,requests=requests,events=inventory['events'],counts=dict(counts),sources=sources),new=True)
    inherited=read(ROOT/'runs/singleton_verify_20261108/feature_phase.json')['sources'].copy()
    for n,h in inherited.items():assert sha(ROOT/n)==h,n
    inherited.update(phase()['sources']);inherited.update(sources);inherited['runs/bbox_spatial_signals_20261108/inventory_recovery1.json']=sha(RUN/'inventory_recovery1.json')
    for path in pool.rglob('*'):
        if path.is_file():inherited[path.relative_to(ROOT).as_posix()]=sha(path)
    write(RUN/'input_phase.json',dict(registered=datetime.now(MSK).isoformat(),sources=inherited),new=True)


if __name__=='__main__':main(guard())
