import os,sys
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from collections import defaultdict,Counter
from research_bbox_fit_maps_20261108.common import *


def main(obs):
    import numpy as np
    from tracking.own_template_verification import correspondence,absolute_score
    phase();assert read(RUN/'data_complete.json')['passed'];inventory=read(READINESS/'inventory.json');manifest=read(RUN/'features_complete.json')['files'];g=read(ROOT/'runs/z13_local_peak_20261021/geometry.json');normal=read(ROOT/'runs/spatial_identity_maps_20261107/image_normalization.json');mean=np.array(normal['mean']);sd=np.where(np.array(normal['sd'])>=1e-12,normal['sd'],1);grouped=defaultdict(list);artifacts={};counts=Counter()
    for event in inventory['events']:grouped[event['video']].append(event)
    for video,events in grouped.items():
        cache={};scores={}
        def get(key):
            if key not in cache:
                path=RUN/'features'/(key+'.npz');assert sha(path)==manifest[key]
                with np.load(path,allow_pickle=False) as z:cache[key]={k:z[k] for k in z.files}
                cache[key]['IMAGE']=(cache[key]['IMAGE'].astype(float)-mean)/sd
            return cache[key]
        for e in events:
            row=e['row_data'];values={a:None for a in ['SUPPORT','RANDOM','IMAGE','R1','DIS','FLYVIS','PERM']};support=0
            if not e['unavailable']:
                current=get(e['current_key']);reference=get(e['reference_key']);pairs=correspondence(e['transform'],row['history'],row['current_crop'],e['box'],g,current['DIS_VALID'],reference['DIS_VALID']);support=0 if pairs is None else len(pairs['unique'])
                for arm in values:
                    name='C' if arm in ['SUPPORT','RANDOM','FLYVIS','PERM'] else arm
                    values[arm]=absolute_score(arm,current[name],reference[name],pairs,e['transform']['scale'],video=video,tid=e['old_tid'],last_frame=row['history'][-1]['frame'],frame=e['frame'],column=e['column'])
            key=f"{e['frame']}|{e['old_tid']}|{e['new_tid']}";scores[key]=dict(scores=values,support=support)
            for arm,value in values.items():counts[arm+('_defined' if value is not None else '_undefined')]+=1
        path=RUN/'scores'/(video+'.json');write(path,dict(passed=True,edges=scores),new=True);artifacts[path.relative_to(ROOT).as_posix()]=sha(path);status('small_cost_spatial_scoring',video=video,counts=dict(counts),guard=obs.read())
    phase();write(RUN/'scores_complete.json',dict(passed=True,artifacts=artifacts,counts=dict(counts)),new=True)


if __name__=='__main__':main(guard())
