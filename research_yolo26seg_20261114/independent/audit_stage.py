"""Recompute saved segmentation diagnostics on CPU, without production helpers."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import hashlib,json,sys
import numpy as np
from scalar_metrics import assignment_optimum,coverage,cosine,timing
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUT=HERE.parent
stage=sys.argv[1]
folder=OUT/('stage_'+stage)
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
result=read(folder/'results.json')
protocol=read(OUT/'protocol.json')
manifest_path=OUT/('case2179.json' if stage=='2179' else 'holdout/manifest.json')
manifest=read(manifest_path)
execution=read(folder/'execution_protocol.json')
device=read(folder/'device.json')
assert result['passed'] and result['model_unchanged']
assert sha(folder/'vectors.npz')==result['vectors_sha256']
assert sha(folder/'masks.npz')==result['masks_sha256']
assert sha(manifest_path)==execution['manifest_sha256']
assert sha(OUT/'protocol.json')==execution['protocol_sha256']
assert sha(OUT/'worker.py')==execution['source_sha256']
for name,h in protocol['inputs'].items():assert sha(ROOT/name)==h,name
vectors=np.load(folder/'vectors.npz',allow_pickle=False)
masks=np.load(folder/'masks.npz',allow_pickle=False)
assert set(vectors.files)=={r['key'] for r in result['records']}
events={e['id']:e for e in manifest['events']}
expected=sum(len(e['roles']) for e in manifest['events'])*len(protocol['arms'])*len(protocol['fills'])*len(protocol['levels'])
assert len(vectors.files)==len(result['records'])==expected
assignments=[]
selected_by_role={}
for segment in result['segments']:
    event=events[segment['event']]
    role_lookup={r['role']:r for r in event['roles']}
    rois=[role_lookup[role]['box'] for role in segment['roles']]
    checked=assignment_optimum(rois,segment['boxes'])
    matrix=np.asarray(segment['iou_matrix'],dtype=np.float64)
    assert matrix.shape==checked['iou'].shape and np.allclose(matrix,checked['iou'],rtol=0,atol=1e-12)
    selected=tuple(-1 if v is None else v for v in segment['selected'])
    assert selected in checked['optimal_assignments'],(segment['event'],segment['frame'],selected)
    assert np.all(np.asarray(segment['classes'])==2)
    assert len(segment['boxes'])==len(segment['classes'])==len(segment['scores'])==len(masks[segment['mask_bank_key']])
    assert min(segment['scores'],default=1)>protocol['settings']['conf']
    top=[int(row.argmax()) if row.size and row.max()>0 else None for row in matrix]
    assert top==segment['independent_top1']
    for role,index in zip(segment['roles'],segment['selected']):
        selected_by_role[(event['id'],role)]=index
        saved=masks[event['id']+'_'+role+'_automatic']
        expected_mask=np.zeros_like(saved) if index is None else masks[segment['mask_bank_key']][index]
        assert np.array_equal(saved,expected_mask)
    collisions={str(index):[role for role,best in zip(segment['roles'],top) if best==index] for index in set(top) if index is not None and top.count(index)>1}
    assignments.append(dict(event=event['id'],frame=segment['frame'],roles=segment['roles'],selected=list(selected),objective=checked['objective'],optimal_count=len(checked['optimal_assignments']),top1_collisions=collisions,independently_optimal=True))
pixel=[]
comparisons=[]
parity=[]
for eid,event in events.items():
    source_masks=np.load(ROOT/event['masks'],allow_pickle=False)
    for role in event['roles']:
        label=role['role']
        key=eid+'_'+label
        own=masks[key+'_manual']
        other=masks[key+'_other']
        assert np.array_equal(own,source_masks[role['manual_key']])
        assert np.array_equal(other,source_masks[role['other_manual_key']])
        assert own.dtype==other.dtype==np.bool_ and own.shape==other.shape==(720,1280)
        assert not np.any(own&other)
        automatic=masks[key+'_automatic']
        groups={'other_physical_core':other}
        if stage=='2179':
            for group in ['occluder','background','uncertain']:
                groups[group]=source_masks[str(role['frame'])+'_'+group]
        stats=coverage(automatic,own,groups)
        stats.update(event=eid,role=label,selected=selected_by_role[(eid,label)],missing=selected_by_role[(eid,label)] is None,unclassified_pixels=int(np.count_nonzero(automatic&~(own|other))),covers_both_physical_cores=bool(np.any(automatic&own) and np.any(automatic&other)))
        pixel.append(stats)
        for arm in protocol['arms']:
            for level in protocol['levels']:
                first=vectors[f'{key}_{arm}_imagenet_mean_{level}']
                assert first.shape==(256,) and np.isfinite(first).all()
                assert abs(float(np.linalg.norm(first))-1)<1e-5
        for level in protocol['levels']:
            a=vectors[f'{key}_full_imagenet_mean_{level}']
            b=vectors[f'{key}_full_black_{level}']
            error=float(np.max(np.abs(a.astype(float)-b.astype(float))))
            assert error<=1e-5
            parity.append(dict(event=eid,role=label,level=level,control='full_fill_identity',max_abs_error=error))
        if stage=='2179':
            bank=np.load(ROOT/'research_assignment_weights_2179_20261112/descriptors.npz')
            frame,index=role['slot']
            previous=bank[frame][index]
            current=vectors[f'{key}_full_imagenet_mean_-1']
            error=float(np.max(np.abs(previous.astype(float)-current.astype(float))))
            assert error<=1e-5,('prior_native',label,error)
            parity.append(dict(event=eid,role=label,control='prior_native_full',max_abs_error=error))
            previous_levels=np.load(ROOT/'research_visible_part_2179_20261114/level_features.npz')
            for arm,old_arm in [('full','full'),('manual','red_query_clean_refs')]:
                old_key=f'imagenet_mean_{old_arm}_{label}_0'
                assert old_key in previous_levels.files
                error=float(np.max(np.abs(previous_levels[old_key].astype(float)-vectors[f'{key}_{arm}_imagenet_mean_0'].astype(float))))
                assert error<=1e-5,('prior_level0',label,arm,error)
                parity.append(dict(event=eid,role=label,control='prior_level0_'+arm,max_abs_error=error))
    for arm in protocol['arms']:
        present=all(selected_by_role[(eid,role)] is not None for role in ['target_ref','competitor_ref','query']) if arm=='automatic' else True
        for fill in protocol['fills']:
            for level in protocol['levels']:
                query=vectors[f'{eid}_query_{arm}_{fill}_{level}']
                own=cosine(query,vectors[f'{eid}_target_ref_{arm}_{fill}_{level}'])
                other=cosine(query,vectors[f'{eid}_competitor_ref_{arm}_{fill}_{level}'])
                comparisons.append(dict(event=eid,arm=arm,fill=fill,level=level,target_cosine=own,competitor_cosine=other,margin_target_minus_competitor=own-other,all_required_masks_present=present,target_preferred=bool(present and own>other),scope='Physical target/competitor descriptor comparison; no assertion of public-ID correctness for mixed detector box.'))
times=[]
for eid in events:
    for arm in ['full','manual','automatic']:
        entries=[r for r in result['timing'] if r['event']==eid and r['arm']==arm]
        assert sorted(r['repeat'] for r in entries)==list(range(protocol['repeats']))
        for entry in entries:assert 0<=entry['peak_gpu_allocated_bytes']<=entry['peak_gpu_reserved_bytes']
        stats=timing([entry['seconds']*1000 for entry in entries])
        stats.update(event=eid,arm=arm,peak_gpu_allocated_bytes=max(r['peak_gpu_allocated_bytes'] for r in entries),peak_gpu_reserved_bytes=max(r['peak_gpu_reserved_bytes'] for r in entries),peak_process_rss_bytes=max(r['process_rss_bytes'] for r in entries))
        stats['registered_performance_gate']=bool(stats['median_ms']<=200 and stats['p95_ms']<=200 and stats['peak_gpu_reserved_bytes']<=8*1024**3)
        times.append(stats)
out=dict(passed=True,audited=datetime.now(timezone(timedelta(hours=3))).isoformat(),stage=stage,vector_count=len(vectors.files),timing_count=len(result['timing']),device=device,assignments=assignments,pixel_coverages=pixel,comparisons=comparisons,parity=parity,timing=times,primary_hashes={n:sha(folder/n) for n in ['results.json','vectors.npz','masks.npz','device.json','execution_protocol.json']},limitations=['Partial manual cores are not complete GT contours; unknown pixels are not counted as definite segmentation error.','Automatic similarity when any required mask is missing is retained numerically but rejected as valid identity evidence.','Query timing excludes frame I/O, historical reference recomputation and human annotation; both networks resident; reserved allocator cache can reflect preceding arms.','No full tracker or population-level improvement demonstrated.'])
with (HERE/(stage+'_audit_v2.json')).open('x',encoding='utf8') as stream:json.dump(out,stream,ensure_ascii=False,indent=2,allow_nan=False)
print(json.dumps(dict(passed=True,stage=stage,vectors=len(vectors.files),pixel=pixel,timing=times),ensure_ascii=False))
