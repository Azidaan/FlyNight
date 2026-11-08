"""Plan missing maps from the immutable native trace; pool exact ready inputs."""
import os,sys,time,shutil
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from collections import Counter,defaultdict
from research_native_confirm_policy_20261108.common import *


def main(obs):
    import numpy as np
    assert read(RUN/'metadata_stage_complete.json')['passed'];requests={};events=[];videos={**read(TRACE/'native_trace_complete.json')['videos'],**read(RUN/'evaluate26_metadata_complete.json')['videos']};counts=Counter();sources={}
    ready={}
    for folder in [OWN,BASE]:
        manifest=read(folder/'features_complete.json')['files']
        for key,h in manifest.items():ready.setdefault(key,(folder/'features'/(key+'.npz'),h))
    for video in videos:
        metadata=TRACE/'native_trace'/(video+'.json');data=read(metadata);sources[metadata.relative_to(ROOT).as_posix()]=sha(metadata)
        for event in data['events']:
            for e in event['edges']:
                row=e['row_data'];ref=[[h['frame'],h['crop']] for h in row['history']];cur=ref+row['search'];keys=[]
                for role,seq in [('reference',ref),('current',cur)]:
                    key=sequence_key(video,seq);keys.append(key)
                    if key not in requests:requests[key]=dict(key=key,video=video,sequence=seq,role=role,current_uid=None)
                    else:assert requests[key]['sequence']==seq and requests[key]['video']==video
                events.append(dict(video=video,frame=event['frame'],old_tid=e['old_tid'],new_tid=e['new_tid'],column=e['column'],reference_key=keys[0],current_key=keys[1]))
    pool=RUN/'reuse_pool';(pool/'features').mkdir(parents=True,exist_ok=True)
    for key,r in requests.items():
        if key in ready:
            source,h=ready[key];assert sha(source)==h;dest=pool/'features'/(key+'.npz');shutil.copyfile(source,dest);r['reuse_parent_sha256']=h;sources[source.relative_to(ROOT).as_posix()]=h;counts['ready_sequence_maps']+=1
        else:counts['missing_'+r['role']+'_maps']+=1
    grouped=defaultdict(list)
    for r in requests.values():grouped[r['video']].append(r)
    original=read(ROOT/'runs/z13_temporal_pair_readiness_20261025/protocol.json')['sha256']
    for video,rows in grouped.items():
        values={}
        for folder in [BASE,OWN]:
            marker=folder/'retinal'/(video+'.json');npz=folder/'retinal'/(video+'.npz')
            if not marker.exists():continue
            m=read(marker);assert sha(npz)==m['npz_sha256'];sources[npz.relative_to(ROOT).as_posix()]=sha(npz)
            with np.load(npz,allow_pickle=False) as z:
                for f,c,h in zip(z['frames'],z['crops'],z['hex']):
                    point=(int(f),tuple(map(int,c)))
                    if point in values:assert np.array_equal(values[point],h)
                    else:values[point]=h.copy()
        keys=sorted(values);folder=pool/'retinal';folder.mkdir(exist_ok=True);npz=folder/(video+'.npz')
        with npz.open('xb') as f:np.savez_compressed(f,frames=np.asarray([p[0] for p in keys]),crops=np.asarray([p[1] for p in keys]).reshape(-1,4),hex=np.stack([values[p] for p in keys]) if keys else np.empty((0,721),np.float16))
        write(folder/(video+'.json'),dict(npz_sha256=sha(npz),scope='Exact pooled already saved views; empty when no prior parent view exists.'),new=True)
        archive=f'runs/z13_data/{video}.npz';assert sha(ROOT/archive)==original[archive];sources[archive]=original[archive]
        for n in [f'data/bdd100k/videos/train/{video}.mov',f'runs/train_cache/{video}/motion/flyvis_tiled.json',f'runs/train_cache/{video}/motion/dis_quarter.npz',f'runs/train_cache/{video}/motion/dis_quarter.json']:sources[n]=sha(ROOT/n)
    write(RUN/'inventory_recovery1.json',dict(passed=True,requests=requests,events=events,counts=dict(counts),sources=sources,scope='All46native traces. No26labels; only missing representations. Fixed native histories make maps shared by every arm.'),new=True)
    inherited=read(OWN/'feature_phase.json')['sources'].copy()
    for n,h in inherited.items():assert sha(ROOT/n)==h,n
    inherited.update(sources)
    for path in Path(__file__).parent.glob('*.py'):inherited[path.relative_to(ROOT).as_posix()]=sha(path)
    inherited['runs/native_confirm_policy_20261108/inventory_recovery1.json']=sha(RUN/'inventory_recovery1.json')
    for path in pool.rglob('*'):
        if path.is_file():inherited[path.relative_to(ROOT).as_posix()]=sha(path)
    write(RUN/'input_phase.json',dict(registered=datetime.now(MSK).isoformat(),sources=inherited),new=True)
    write(RUN/'data_registration.json',dict(registered=datetime.now(MSK).isoformat(),sources=inherited,stages=[dict(name='retinal_inputs',script='research_native_confirm_policy_20261108/input_wrapper.py'),dict(name='spatial_features',script='research_native_confirm_policy_20261108/feature_wrapper.py')]),new=True);print(dict(requests=len(requests),events=len(events),counts=dict(counts)))


if __name__=='__main__':main(guard())
