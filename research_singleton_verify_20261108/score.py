"""Score every required own-template pair without labels or parameter selection."""
import os,sys
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from collections import defaultdict,Counter
from research_singleton_verify_20261108.common import *


def score_all(obs):
    import numpy as np
    from tracking.own_template_verification import correspondence,absolute_score
    from tracking.spatial_identity_assignment import geometric_match
    phase('comparison_phase.json');inventory=read(RUN/'inventory_recovery1.json');features=read(RUN/'features_complete.json')['files']
    g=read(ROOT/'runs/z13_local_peak_20261021/geometry.json');norm=read(ROOT/'runs/spatial_identity_maps_20261107/image_normalization.json')
    mean=np.array(norm['mean']);sd=np.where(np.array(norm['sd'])>=1e-12,norm['sd'],1)
    exports=read(BASE/'export_complete.json')['videos'];events=defaultdict(list)
    for e in inventory['events']:events[e['video']].append(e)
    result={};checks=Counter();counts=Counter();technical=None
    for number,(video,items) in enumerate(events.items()):
        source=ROOT/exports[video]['path'];assert sha(source)==exports[video]['sha256'];snapshots=read(source)['snapshots'];cache={};rows={}
        def get(key):
            if key not in cache:
                path=RUN/'features'/(key+'.npz');assert sha(path)==features[key]
                with np.load(path,allow_pickle=False) as z:cache[key]={k:z[k] for k in z.files}
                cache[key]['IMAGE']=(cache[key]['IMAGE'].astype(float)-mean)/sd
            return cache[key]
        for e in items:
            snap=snapshots[str(e['frame'])];row=snap['rows'][e['row']];j=e['column'];transform=row['geometry'][row['choices'].index(j)]
            current=get(e['current_key']);reference=get(e['reference_key'])
            pairs=correspondence(transform,row['history'],row['current_crop'],snap['boxes'][j],g,current['DIS_VALID'],reference['DIS_VALID'])
            if technical is None and e['calibration']:
                direct=geometric_match(row['mean'],row['covariance'],row['history'],row['current_crop'],snap['boxes'][j],g)
                assert direct['scale']==transform['scale'] and direct['offset']==transform['offset']
                mask=current['DIS_VALID'][direct['p']]&reference['DIS_VALID'][direct['q']]
                if pairs is not None:
                    assert np.array_equal(pairs['p'],np.asarray(direct['p'])[mask]) and np.array_equal(pairs['q'],np.asarray(direct['q'])[mask])
                technical=dict(passed=True,video=video,frame=e['frame'],row=e['row'],cached_transform_mapping_exact=True)
            values={}
            for kind in ['SUPPORT','RANDOM','IMAGE','R1','DIS','FLYVIS','PERM']:
                name='C' if kind in ['FLYVIS','PERM','SUPPORT','RANDOM'] else kind
                values[kind]=absolute_score(kind,current[name],reference[name],pairs,transform['scale'],video=video,tid=row['tid'],
                    last_frame=row['history'][-1]['frame'],frame=e['frame'],column=j)
                counts[kind+('_defined' if values[kind] is not None else '_undefined')]+=1
            rows.setdefault(str(e['frame']),{})[f"{e['row']},{j}"]=dict(**e,scores=values,support=0 if pairs is None else len(pairs['unique']))
            checks['own_pairs']+=1
        dest=RUN/'scores'/(video+'.json');write(dest,dict(passed=True,frames=rows,feature_sources={k:features[k] for e in items for k in [e['reference_key'],e['current_key']]}),new=True)
        result[video]=sha(dest);status('absolute_own_map_scoring',complete=number+1,total=len(events),counts=dict(counts),guard=obs.read())
    assert technical is not None;write(RUN/'scores_complete.json',dict(passed=True,videos=result,counts=dict(counts),checks=dict(checks),technical=technical),new=True)


if __name__=='__main__':score_all(guard())
