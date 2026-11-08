"""Only absent score grids; existing results retain exact values and provenance."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'runs/flyvis_repairs_20261106/unification_20261107'))
from system_memory_guard import guard_entrypoint
guard_entrypoint(timeout=900)
import numpy as np
from scipy.interpolate import LinearNDInterpolator
from tracking.observed_spatial_reader import ObservedSpatialReader
from tracking.z13_readout import HexGeometry
from scripts.z13_target_model import neighbor_aug
OTHER=[0,1,2,3]+list(range(12,34));SELECT=[0,5,11,16,17,22,28,33]
NAMES=['FlyVis','C34','center8','neighbor8','input8_temporal','R134_temporal','R134_spatial']
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(n,v):
    with (OUT/n).open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def key(r):return r['video'],r['frame'],r['old'],r['new']
def verify(p):
    for path,h in p['sources'].items():assert sha(ROOT/path)==h,path
def arrays(k,p,nb):
    with np.load(ROOT/p['maps'][k]['path'],allow_pickle=False) as z:c8=z['C'].astype(float);r=z['R1'].astype(float) if 'R1' in z.files else None
    if r is None:
        with np.load(OUT/'missing_input_features.npz',allow_pickle=False) as z:r=z[k].astype(float)
        with np.load(ROOT/'runs/singleton_eight_input_20261108/input8_maps.npz',allow_pickle=False) as z:assert np.array_equal(r[:,SELECT],z[k])
    with np.load(OUT/'maps'/(k+'.npz'),allow_pickle=False) as z:extra=z['C_OTHER']
    full=np.empty((721,34));full[:,4:12]=c8;full[:,OTHER]=extra;central=r[:,SELECT[:4]]
    return dict(FlyVis=c8,C34=full,center8=np.concatenate([central,central],1),neighbor8=neighbor_aug(central.T[None],nb)[0].T,input8_temporal=r[:,SELECT],R134_temporal=r,R134_spatial=neighbor_aug(r[:,:17].T[None],nb)[0].T)
def register():
    p=read(OUT/'protocol.json');verify(p);verify(read(OUT/'maps_complete.json'));proof=read(ROOT/'runs/clipped_anchor_readout_20261109/runtime_verification.json');verify(proof)
    files=[Path(__file__),OUT/'protocol.json',OUT/'maps_complete.json',OUT/'readiness.json',OUT/'external_selection_protocol.json',OUT/'external_inventory.json',ROOT/'runs/clipped_anchor_readout_20261109/runtime_verification.json',ROOT/'runs/singleton_support_size_20261108/protocol.json',ROOT/'runs/singleton_support_size_20261108/results.json',ROOT/'runs/channel_balance_attribution_20261109/transfer.json',ROOT/'runs/singleton_matched_support_20261108/scores.json',ROOT/'runs/singleton_neighbor_input_20261108/scores.json',ROOT/'runs/singleton_eight_input_20261108/input8_maps.npz']
    historical=[]
    for path in sorted((ROOT/'runs/clipped_anchor_readout_20261109/events').glob('*.json')):
        r=read(path)
        if r['anchor']['changed'] and r['kind'] in ['own','foreign'] and (r['old'],r['new'])!=(88,144):historical.append(path.relative_to(ROOT).as_posix());files.append(path)
    assert len(historical)==5
    write('comparison_protocol.json',dict(registered=__import__('datetime').datetime.now().isoformat(),new_cases=[key(c) for c in p['cases']],historical_clipped_sources=historical,
        decision='No directly confirmed independent clipped cases in the46-video saved context scan. Report the six newly compared original eligible cases without clipping separately from five already scored clipped development cases. Never pool these into an independent transfer claim. Two newly found both-direct own cases require additional maps and are inventory evidence only, not silently treated as scored.',
        controls='Exactly seven prior representations, four frozen combinations of original/corrected geometry and global/channel normalization. All sources immutable and all ready grids reused. No decision threshold. Clip stratum has4foreign/1own, no-clip5own/1foreign; report sparse class coverage.',
        source_proof='All13 already passed mathematical/runtime tests reused by exact source hashes. Compare new plans to archived geometry independently. Independent SciPy interpolation and normalized scalar products verify ONLY newly computed grids.',
        sources={f.relative_to(ROOT).as_posix():sha(f) for f in files}))
def run():
    cp=read(OUT/'comparison_protocol.json');verify(cp);p=read(OUT/'protocol.json');verify(p);verify(read(OUT/'maps_complete.json'))
    if (OUT/'results.json').exists():raise SystemExit('Completed comparison exists; no repetition.')
    g=read(ROOT/'runs/z13_local_peak_20261021/geometry.json');reader=ObservedSpatialReader(g['points'],g['eye_size'],[1280,720]);nb=HexGeometry(np.array(g['points']),g['eye_size']).neighbors
    seven={key(c):c for c in read(ROOT/'runs/singleton_support_size_20261108/protocol.json')['cases']};old7=read(ROOT/'runs/singleton_support_size_20261108/results.json')['records'];bal7=read(ROOT/'runs/channel_balance_attribution_20261109/transfer.json')['records'];match=read(ROOT/'runs/singleton_matched_support_20261108/scores.json')['pairs'];simple=read(ROOT/'runs/singleton_neighbor_input_20261108/scores.json')['pairs'];rows=[];reused=0;computed=0;maxerr=0.
    for c in p['cases']:
        assert not c['clipped'] and not c['plan_joint']['anchor']['changed'];assert c['plan_old']['cells']==c['plan_joint']['cells'];assert np.array_equal(c['plan_old']['query'],c['plan_joint']['query'])
        if key(c) in seven:
            old=seven[key(c)]['geometry'];assert old['eligible_cells']==c['plan_old']['cells'];points=np.array(g['points'])[old['eligible_cells']];query=(points[:,None]*old['scale0']+old['crop0'][:2]+np.array(old['prediction'])+old['offsets']-old['crop'][:2])/old['scale'];assert np.max(abs(query-np.array(c['plan_old']['query'])))<=1e-12
        a=arrays(c['map_keys'][0],p,nb);b=arrays(c['map_keys'][1],p,nb);variants={}
        for name in NAMES:
            prior=next((r for r in old7 if key(r)==key(c) and r['representation']==name),None);balanced=next((r for r in bal7 if key(r)==key(c) and r['representation']==name),None)
            global_scores=prior['forward_scores'] if prior else None;balanced_scores=balanced['scores'] if balanced else None;provenance={}
            if prior:assert prior['cells']==c['plan_old']['cells'];provenance['old']='runs/singleton_support_size_20261108/results.json'
            if balanced:assert balanced['cells']==c['plan_old']['cells'];provenance['normalization']='runs/channel_balance_attribution_20261109/transfer.json'
            if global_scores is None and c['old']==3 and c['new']==273:
                source=match if name=='FlyVis' else simple;rr=next((r for r in source if r['video']==c['video'] and r['old']==c['old'] and r['new']==c['new']),None)
                if rr is not None and name in rr['variants']:
                    assert rr['cells']==c['plan_old']['cells'];global_scores=rr['variants'][name]['scores'];provenance['old']='runs/singleton_matched_support_20261108/scores.json' if name=='FlyVis' else 'runs/singleton_neighbor_input_20261108/scores.json'
            for norm,values in [('global',global_scores),('channels',balanced_scores)]:
                if values is not None:reused+=1;continue
                computed+=1;result=reader.compare(a[name],b[name],c['plan_old'],normalization=norm);values=result['scores'];idx=c['plan_old']['cells'];template=a[name][idx];patches=LinearNDInterpolator(reader.mesh,b[name])(np.array(c['plan_old']['query'])).transpose(1,0,2)
                # Independent expression, only new grids. No old scores re-evaluated.
                ac=template-template.mean(0);ac[:,np.ptp(template,axis=0)==0]=0;an=np.linalg.norm(ac,axis=0);active=an>0
                independent=[]
                for patch in patches:
                    bc=patch-patch.mean(0);bc[:,np.ptp(patch,axis=0)==0]=0;bn=np.linalg.norm(bc,axis=0)
                    if norm=='global':den=np.linalg.norm(ac)*np.linalg.norm(bc);score=float(np.vdot(ac,bc)/den) if den else None
                    elif not active.any() or not np.any(bn):score=None
                    else:score=float(sum(float(np.dot(ac[:,j],bc[:,j])/(an[j]*bn[j])) if bn[j] else 0. for j in np.flatnonzero(active))/active.sum())
                    independent.append(score)
                assert len(independent)==len(values)
                for x,y in zip(values,independent):
                    assert (x is None)==(y is None)
                    if x is not None:maxerr=max(maxerr,abs(x-y));assert abs(x-y)<1e-9
                if norm=='global':global_scores=values;provenance['old']='new_missing_grid'
                else:balanced_scores=values;provenance['normalization']='new_missing_grid'
            def pack(scores):
                ids=[i for i,v in enumerate(scores) if v is not None];j=max(ids,key=lambda i:scores[i]) if ids else None
                return dict(scores=scores,maximum=scores[j] if j is not None else None,winner=j,fixed=scores[9])
            variants[name]=dict(old=pack(global_scores),normalization=pack(balanced_scores),edge=pack(global_scores),joint=pack(balanced_scores),provenance=provenance)
        rows.append(dict(video=c['video'],frame=c['frame'],old=c['old'],new=c['new'],kind=c['kind'],clipped=False,source_group='other_video_new_reader',old_cells=c['plan_old']['cells'],joint_cells=c['plan_joint']['cells'],direct_current=True,direct_reference=False,variants=variants))
        print('Compared',key(c),'reused',reused,'new',computed,flush=True)
    historical=[]
    for path in cp['historical_clipped_sources']:
        r=read(ROOT/path);variants={}
        for name,v in r['variants'].items():
            def pack(scores):
                ids=[i for i,x in enumerate(scores) if x is not None];j=max(ids,key=lambda i:scores[i]) if ids else None
                return dict(scores=scores,maximum=scores[j] if j is not None else None,winner=j,fixed=scores[9])
            # Old baseline grids and equal-channel grids remain in their original events.
            filename=Path(path).name;oldpath=ROOT/('runs/background_patch_gate_20261108/events' if name in ['FlyVis','center8','neighbor8','input8_temporal'] else 'runs/decoder34_identity_gate_20261108/events')/filename;old=read(oldpath)['variants'][name];balanced=read(ROOT/'runs/channel_balance_attribution_20261109/events'/filename)['variants'][name]
            variants[name]=dict(old=pack(old['correlation_scores']),normalization=pack(balanced['scores']),edge=pack(v['global_scores']),joint=pack(v['balanced_scores']),provenance=dict(all=path))
        historical.append(dict(video=r['video'],frame=r['frame'],old=r['old'],new=r['new'],kind=r['kind'],clipped=True,source_group='development_ready_only',old_cells=r['baseline_cells'],joint_cells=r['geometry']['cells'],source=path,variants=variants))
    verify(cp);verify(p);verify(read(OUT/'maps_complete.json'))
    write('results.json',dict(passed=True,new_records=rows,historical_clipped_records=historical,new_reader_events=len(rows),ready_clipped_events=len(historical),ready_score_grids_reused=reused,new_missing_score_grids=computed,historical_grids_reused=len(historical)*len(NAMES)*4,independent_new_grid_max_error=maxerr,all_ready_maps_unchanged=True,zero_detector_calls=True,zero_new_retinal_views=True,reader_unchanged=True));print('Finished',reused,computed,maxerr,flush=True)
if __name__=='__main__':{'register':register,'run':run}[sys.argv[1]]()
