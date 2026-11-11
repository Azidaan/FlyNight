"""Reuse last-state scores and frozen heads; compute only new joint evidence."""
from common import *
start()
import numpy as np
from tracking.observed_spatial_reader import ObservedSpatialReader
from tracking.full_template_correspondence import relative_template_plan
from tracking.hex_linear_interpolation import sample_linear
from tracking.z13_readout import HexGeometry
from scripts.z13_target_model import neighbor_aug
from tracking.channel_identity_head import decision_function

verify(); assert read(OUT/'maps_complete.json')['passed']
edges = read(OLD/'protocol_v2.json')['edges']; oldfeatures = read(OLD/'features.json')['records']
oldpred = read(OLD/'predictions.json'); models = {(f['video'], f['model']): f for f in oldpred['folds']}
geom = read(ROOT/'runs/z13_local_peak_20261021/geometry.json')
reader = ObservedSpatialReader(geom['points'], geom['eye_size'], [1280,720]); nb = HexGeometry(np.array(geom['points']),geom['eye_size']).neighbors
records = []; head_identity_max = 0.; omitted_last_searches = 0
for i,(e,ready,oldp) in enumerate(zip(edges,oldfeatures,oldpred['records'])):
    assert all(e[n] == ready[n] == oldp[n] for n in ['video','frame','old_id','raw_id'])
    plan = relative_template_plan(reader,e['reference_box'],e['reference_crop'],e['current_box'],e['current_crop'],e['offsets'])
    assert len(plan['cells']) == ready['full_cells']
    zero = np.flatnonzero((np.asarray(e['offsets']) == 0).all(1)); assert len(zero) == 1; zero=int(zero[0])
    lastmaps=[]; firstmaps=[]
    for seq in [e['sequence'][:4],e['sequence']]:
        k=key(e['video'],seq); value=arrays(k); meta=read(OUT/'maps'/(k+'.json')); assert sha(OUT/'maps'/(k+'.npz'))==meta['sha256']
        with np.load(OUT/'maps'/(k+'.npz'),allow_pickle=False) as z: first={n:z[n].copy() for n in z.files}
        def signals(c,r):
            r=r.astype(float)
            return dict(C34=c.astype(float),R134_temporal=r,R134_spatial=neighbor_aug(r[:,:17].T[None],nb)[0].T)
        lastmaps.append(signals(value['C34'],value['R1'])); firstmaps.append(signals(first['C1'],first['R1_mid']))
    variants={}
    for name in ['C34','R134_temporal','R134_spatial']:
        saved=ready['variants'][name]; assert saved['valid']
        a=firstmaps[0][name][plan['cells']]; patches=sample_linear(reader.mesh,firstmaps[1][name],plan['query']).transpose(1,0,2)
        midvectors=[correlations(a,b) for b in patches]
        scores=[None if v is None or saved['scores'][k] is None else (float(v.mean())+saved['scores'][k])/2 for k,v in enumerate(midvectors)]
        valid=[k for k,s in enumerate(scores) if s is not None]; winner=max(valid,key=lambda k:scores[k]) if valid else None
        fold=models[e['video'],name]; geometry=np.array(ready['geometry']); g=(geometry-np.array(models[e['video'],'G7']['normalization']['mean']))/np.array(models[e['video'],'G7']['normalization']['scale'])
        assert e['video'] not in fold['training_videos']
        identity=float(decision_function(fold['fit'],np.r_[g,saved['channels']][None])[0]); head_identity_max=max(head_identity_max,abs(identity-oldp['scores'][name]))
        assert abs(identity-oldp['scores'][name]) < 1e-12
        if winner is None:
            z=oldp['scores']['G7']; vector=None; v2=None
        else:
            if winner==saved['winner']: v2=np.asarray(saved['channels'])
            else:
                patch=sample_linear(reader.mesh,lastmaps[1][name],plan['query'][:,winner])
                v2=correlations(lastmaps[0][name][plan['cells']],patch)
                assert v2 is not None and abs(v2.mean()-saved['scores'][winner]) < 1e-12
            vector=(midvectors[winner]+v2)/2
            assert abs(vector.mean()-scores[winner]) < 1e-12
            z=float(decision_function(fold['fit'],np.r_[g,vector][None])[0])
        omitted_last_searches+=1
        variants[name]=dict(valid=winner is not None,winner=winner,old_winner=saved['winner'],scores=scores,
            first_vectors=midvectors,channels=vector,last_vector_at_joint=v2,
            before_shift=scores[zero],after_shift=None if winner is None else scores[winner],
            last_before_shift=saved['scores'][zero],last_after_shift=saved['scores'][saved['winner']],
            last_decision_score=oldp['scores'][name],joint_decision_score=z,
            decision_changed=bool((z>0)!=(oldp['scores'][name]>0)))
    records.append(dict(video=e['video'],frame=e['frame'],old_id=e['old_id'],raw_id=e['raw_id'],label=e['kind'],
        cells=plan['cells'],query=plan['query'],geometry=ready['geometry'],variants=variants))
    status('comparing',completed=i+1,total=26)
write('predictions.json',dict(passed=True,records=records,frozen_head_identity_max_error=head_identity_max,
    ready_last_searches_reused=omitted_last_searches,training_runs=0,threshold_sweeps=0))
verify();status('comparison_complete',completed=26,total=26)
print('JOINT_COMPARISON_READY',len(records))
