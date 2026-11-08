import os,sys
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_native_confirm_policy_20261108.common import *


def main(obs):
    import numpy as np
    from tracking.own_template_verification import correspondence,absolute_score
    phase('comparison_phase.json');inventory=read(RUN/'inventory_recovery1.json');features=read(RUN/'features_complete.json')['files'];geometry=read(ROOT/'runs/z13_local_peak_20261021/geometry.json');norm=read(ROOT/'runs/spatial_identity_maps_20261107/image_normalization.json');mean=np.asarray(norm['mean']);sd=np.maximum(np.asarray(norm['sd']),1e-12);byvideo={};manifest={}
    for e in inventory['events']:byvideo.setdefault(e['video'],[]).append(e)
    for number,(video,items) in enumerate(byvideo.items()):
        data=read(TRACE/'native_trace'/(video+'.json'));edges={(event['frame'],e['old_tid'],e['new_tid']):e for event in data['events'] for e in event['edges']};cache={};scores={}
        def get(key):
            if key not in cache:
                p=RUN/'features'/(key+'.npz');assert sha(p)==features[key]
                with np.load(p,allow_pickle=False) as z:cache[key]={k:z[k] for k in z.files}
                cache[key]['IMAGE']=(cache[key]['IMAGE'].astype(float)-mean)/sd
            return cache[key]
        for item in items:
            key=(item['frame'],item['old_tid'],item['new_tid']);e=edges[key];row=e['row_data'];cur=get(item['current_key']);ref=get(item['reference_key']);pair=correspondence(e['transform'],row['history'],row['current_crop'],e['box'],geometry,cur['DIS_VALID'],ref['DIS_VALID']);assert pair is not None and len(pair['unique'])==e['support'],(video,key,'common support drift')
            values={}
            for arm in ['RANDOM','IMAGE','R1','DIS','FLYVIS','PERM']:
                kind='C' if arm in ['RANDOM','FLYVIS','PERM'] else arm
                values[arm]=absolute_score(arm,cur[kind],ref[kind],pair,e['transform']['scale'],video=video,tid=e['old_tid'],last_frame=row['history'][-1]['frame'],frame=item['frame'],column=e['column'])
            scores['|'.join(map(str,key))]=dict(scores=values,support=e['support'],reference_key=item['reference_key'],current_key=item['current_key'])
        p=RUN/'scores'/(video+'.json');write(p,dict(passed=True,video=video,edges=scores),new=True);manifest[video]=sha(p);status('native_confirmation_spatial_scores',complete=number+1,total=len(byvideo),video=video,edges=len(scores),guard=obs.read())
    phase('comparison_phase.json');write(RUN/'scores_complete.json',dict(passed=True,videos=manifest),new=True)


if __name__=='__main__':main(guard())
