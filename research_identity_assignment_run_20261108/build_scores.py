"""Label-free spatial scores for every competing row on the same support."""
import os,sys
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
from pathlib import Path
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_identity_assignment_run_20261108.common import *
from tracking.spatial_identity_assignment import centered_similarity,flow_similarity


def score_column(snapshot,j,col,features,norm,video):
    result=dict(rows=col['rows'],geometry_valid=col['geometry_valid'],scores={},support=[])
    if not col['geometry_valid']:return result
    current=features[col['current_key']];values={a:[] for a in ['G_IMAGE','G_R1','G_DIS','G_FLYVIS','G_PERM']}
    for item in col['items']:
        row=snapshot['rows'][item['row']];ref=features[item['reference_key']];mapping=item['mapping']
        pp=np.array(mapping['p'],int);qq=np.array(mapping['q'],int)
        valid=current['DIS_VALID'][pp]&ref['DIS_VALID'][qq]
        pp=pp[valid];qq=qq[valid];unique,inverse=np.unique(qq,return_inverse=True)
        result['support'].append(len(unique))
        if len(unique)<3:result['geometry_valid']=False;return result
        counts=np.bincount(inverse)[:,None]
        def aggregate(x):
            a=np.zeros((len(unique),x.shape[1]));np.add.at(a,inverse,x[pp]);return a/counts
        for arm,name in [('G_FLYVIS','C'),('G_R1','R1'),('G_IMAGE','IMAGE')]:
            x=current[name].astype(float);y=ref[name].astype(float)
            if name=='IMAGE':
                scale=np.where(np.array(norm['sd'])>=1e-12,norm['sd'],1)
                x=(x-norm['mean'])/scale;y=(y-norm['mean'])/scale
            values[arm].append(centered_similarity(aggregate(x),y[unique]))
        values['G_DIS'].append(flow_similarity(aggregate(current['DIS'].astype(float)),ref['DIS'][unique],mapping['scale']))
        ids=np.array(mapping['reference_ids'],int);ids=ids[ref['DIS_VALID'][ids]]
        fc=aggregate(current['C'].astype(float));permuted=[]
        for rep in range(32):
            seed=int.from_bytes(hashlib.sha256(f"{video}|{row['tid']}|{row['history'][-1]['frame']}|{rep}|42".encode()).digest()[:8],'little')
            y=ref['C'].astype(float).copy();y[ids]=y[np.random.default_rng(seed).permutation(ids)]
            permuted.append(centered_similarity(fc,y[unique]))
        values['G_PERM'].append(float(np.mean(permuted)) if all(x is not None for x in permuted) else None)
    result['scores']=values
    return result


def main(obs):
    verify_phase('score_phase.json');complete=read(RUN/'features_complete.json');assert complete['passed']
    exports=read(RUN/'export_complete.json')['videos'];folder=RUN/'scores';folder.mkdir(exist_ok=True)
    norm=read(ROOT/'runs/spatial_identity_maps_20261107/image_normalization.json');manifest={};summary=[]
    for number,(video,entry) in enumerate(exports.items()):
        path=folder/(video+'.json')
        if path.exists():manifest[video]=sha(path);continue
        data=read(ROOT/entry['path']);keys=set()
        for snap in data['snapshots'].values():
            for col in snap['columns'].values():
                if col['geometry_valid']:keys.add(col['current_key']);keys.update(x['reference_key'] for x in col['items'])
        features={}
        for key in keys:
            f=RUN/'features'/(key+'.npz');assert sha(f)==complete['files'][key]
            with np.load(f,allow_pickle=False) as z:features[key]={k:z[k] for k in z.files}
        output={};counts=dict(columns=0,common_geometry_columns=0,nonzero_potential={})
        for frame,snap in data['snapshots'].items():
            columns={j:score_column(snap,int(j),col,features,norm,video) for j,col in snap['columns'].items()}
            output[frame]=dict(votes=snap['votes'],columns=columns)
            for col in columns.values():
                counts['columns']+=1;counts['common_geometry_columns']+=int(col['geometry_valid'])
                for arm,values in col['scores'].items():
                    if col['geometry_valid'] and all(x is not None for x in values) and np.ptp(values)>1e-12:
                        counts['nonzero_potential'][arm]=counts['nonzero_potential'].get(arm,0)+1
        write(path,dict(video=video,frames=output,counts=counts,export_sha256=entry['sha256']),new=True)
        manifest[video]=sha(path);summary.append(dict(video=video,**counts))
        status('label_free_scores',complete=number+1,total=len(exports),guard=obs.read())
    verify_phase('score_phase.json');write(RUN/'scores_complete.json',dict(passed=True,videos=manifest,summary=summary),new=True)


if __name__=='__main__':main(guard())
