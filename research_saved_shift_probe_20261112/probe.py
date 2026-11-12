from common import *
import importlib.util
import numpy as np
from tracking.observed_spatial_reader import ObservedSpatialReader
from tracking.full_template_correspondence import relative_template_plan
from tracking.hex_linear_interpolation import sample_linear
from tracking.channel_agreement_correspondence import agreement_channels
from tracking.z13_readout import HexGeometry
p=verify();spec=importlib.util.spec_from_file_location('saved_agreement_utils',ROOT/'research_agreement_readout_20261112/common.py');u=importlib.util.module_from_spec(spec);spec.loader.exec_module(u)
ag=read(ROOT/'research_agreement_readout_20261112/protocol.json');pred=read(ROOT/'research_agreement_readout_20261112/predictions.json');feat=read(ROOT/'research_agreement_readout_20261112/features.json')
geom=read(ROOT/'runs/z13_local_peak_20261021/geometry.json');reader=ObservedSpatialReader(geom['points'],geom['eye_size'],[1280,720]);neighbors=HexGeometry(np.array(geom['points']),geom['eye_size']).neighbors
records=[];arrays={};maxparity=0.;maxfeature=0.
for dataset in ['train','screen']:
    for i,r in enumerate(ag[dataset]):
        plan=relative_template_plan(reader,r['reference_box'],r['reference_crop'],r['current_box'],r['current_crop'],r['offsets']);cells=plan['cells'];assert len(cells)==r['full_cells']
        maps=[u.representations(u.arrays(ag['snapshots'][k]),neighbors) for k in r['keys']]
        hashes=[{a:hashlib.sha256(m[a].tobytes()).hexdigest() for a in ARMS} for m in maps]
        for arm in ARMS:
            fit=next(f for f in pred['fits'] if f['model']==arm+'__agreement' and f['video']==('FINAL' if dataset=='screen' else r['video']))
            theta=np.array(fit['fit']['theta']);g=(np.array(r['geometry'])-fit['normalization']['mean'])/fit['normalization']['scale'];a=maps[0][arm][cells].copy()
            prefix=f'{dataset}_{i}_{arm}';arrays[prefix+'_a']=a;arrays[prefix+'_map']=maps[1][arm]
            for selector in ARMS:
                winner=r['old_variants'][selector]['winner'];query=plan['query'][:,winner].copy();original_query=plan['query'][:,winner]
                assert np.array_equal(query,original_query) and query.shape==(len(cells),2) and np.isfinite(query).all()
                b=sample_linear(reader.mesh,maps[1][arm],query);q=agreement_channels(a,b,mean_penalty=True);assert q is not None and np.isfinite(q).all()
                x=np.r_[g,(q-fit['signal_mean'])/fit['signal_scale']];z=float(x@theta[:-1]+theta[-1])
                if selector==arm:
                    error=abs(z-pred[dataset][i]['scores'][arm+'__agreement']);ferror=float(np.max(abs(q-feat[dataset][i]['variants'][arm]['agreement'])));assert max(error,ferror)<1e-10;maxparity=max(maxparity,error);maxfeature=max(maxfeature,ferror)
                key=prefix+'_'+selector;arrays[key+'_b']=b;arrays[key+'_query']=query
                records.append(dict(dataset=dataset,index=i,video=r['video'],frame=r['frame'],old_id=r['old_id'],label=r['label'],representation=arm,shift_selector=selector,shift_index=winner,shift_pixels=r['offsets'][winner],score=z,accepted=bool(z>0),channels=q,patch_key=key,reference_key=prefix,identity_control=selector==arm,fit_video=fit['video']))
                assert np.array_equal(query,original_query)
        assert hashes==[{a:hashlib.sha256(m[a].tobytes()).hexdigest() for a in ARMS} for m in maps]
verify()
with (OUT/'patches.npz').open('xb') as f:np.savez_compressed(f,**arrays)
save('raw.json',dict(passed=True,records=records,patches_sha256=sha(OUT/'patches.npz'),max_diagonal_score_error=maxparity,max_diagonal_feature_error=maxfeature,new_maps=0,new_fits=0,new_shifts=0,tracker_calls=0))
for r in records:
    if r['dataset']=='screen':print(r['old_id'],r['representation'],r['shift_selector'],round(r['score'],6))
