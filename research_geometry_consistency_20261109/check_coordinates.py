"""Prerecorded coordinate-only gate on the closed C34 native pilot, no signal search."""
import os,sys,json,hashlib,gzip,subprocess
from pathlib import Path
from datetime import datetime,timezone,timedelta
ROOT=Path(__file__).resolve().parents[1]
OUT=Path(__file__).resolve().parent
NATIVE=ROOT/'runs/full_template_native_20261109'
MSK=timezone(timedelta(hours=3));STOP=datetime(2026,11,9,23,54,tzinfo=MSK)
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'runs/flyvis_repairs_20261106/unification_20261107'))
from system_memory_guard import guard_entrypoint
guard_entrypoint(timeout=min(600,(STOP-datetime.now(MSK)).total_seconds()))
import psutil
psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def save(name,obj):
    with (OUT/name).open('x',encoding='utf8',newline='\n') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
mode=sys.argv[1]
sources=[Path(__file__),ROOT/'tracking/spatial_identity_assignment.py',ROOT/'tracking/full_template_correspondence.py',
    ROOT/'tracking/observed_spatial_reader.py',ROOT/'tracking/hex_linear_interpolation.py',
    NATIVE/'native_runtime.py',NATIVE/'execution_protocol.json',NATIVE/'id_fate.json',
    ROOT/'runs/z13_local_peak_20261021/geometry.json',OUT/'ОСНОВАНИЕ.md']
sources+=sorted((NATIVE/'rollouts/C34').glob('*.events.json.gz'))
if mode=='register':
    assert len(list((NATIVE/'rollouts/C34').glob('*.events.json.gz')))==20
    save('coordinate_protocol.json',dict(registered=datetime.now(MSK).isoformat(),
        head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip(),
        sources={p.relative_to(ROOT).as_posix():sha(p) for p in sources},
        question='Does the geometry admission transformation materially differ from the frozen C34 map reader on its actual scored requests, and is a common visible support available?',
        status='Post-negative-pilot diagnostic, not an independent tracker test or a selected successful reader.',
        action='If supported by coordinate evidence, replace only spatial correspondence by the existing geometry transform; retain original G7features, gates,19physicalshifts,34channels,normalization and all matched input controls. No present permission to declare this a useful fix.',
        math='Fbox=bk+sbox*(x-b0); Fgeom=bk+sg*(x-b0-og); inverse=b0+og+(y-bk)/sg. Difference=(sg-sbox)*(x-b0)-sg*og. Verify inverse and degenerate equality before measured records.',
        support='Reference cells inside old box whose zero-shift Fgeom lies in candidate box AND all19queries lie in current eye hull. Choose once before feature sampling; never intersect candidate box across all19shifts. Same plan for FlyVis and ordinary input controls.',
        unchanged_admission='Keep the archived old support>=3 admission and original7geometryfeatures. New support<3 means geometry fallback, not new cost threshold. Do not relabel unknown pairs.',
        analysis='All213scoredC34records, not selected successes. Attach existing six direct labels only after coordinate computation. No maps, scores, head fits, thresholds, video, assignments or metrics repeated.',
        gate='If transforms coincide within floating-point roundoff everywhere, stop this action. If visible support is unavailable in own refusal cases, a geometry-aligned reader alone cannot repair those cases under current three-cell contract. Otherwise report support and deformation in grid-spacing units before deciding the implementation. No performance acceptance threshold inferred from these20viewedvideos.',
        alternative='Kalman full-size geometry may itself be wrong; training/readout weights may not transfer. Inverse consistency or coordinate displacement is not proof of object identity.',
        prior='Old8channelG_FLYVIS used this inverse geometry with nearest-cell correspondence and inter-track relative costs and failed prior native20video criteria. This proposed gate is specific to the later34channel/full-template/interpolated19shift reader.',
        deadline=STOP.isoformat(),device='CPU',GPU=False))
    print('REGISTERED',len(sources),flush=True);raise SystemExit(0)
assert mode=='run'
p=read(OUT/'coordinate_protocol.json')
for name,digest in p['sources'].items():assert sha(ROOT/name)==digest,name
import numpy as np
from scipy.spatial import Delaunay
from tracking.spatial_identity_assignment import geometric_match,frame_points
from tracking.observed_spatial_reader import ObservedSpatialReader
from tracking.full_template_correspondence import relative_template_plan
geom=read(ROOT/'runs/z13_local_peak_20261021/geometry.json')
reader=ObservedSpatialReader(geom['points'],geom['eye_size'],[1280,720])
axes=np.array([(u,v) for u in range(-2,3) for v in range(-2,3) if max(abs(u),abs(v),abs(u+v))<=2])
offset_axes=axes@np.array([[13.,6.5],[0.,13.]])
def transition(x,b0,bk,scale,offset):return bk+scale*(x-b0-offset)
def inverse(y,b0,bk,scale,offset):return b0+offset+(y-bk)/scale
rng=np.random.default_rng(42)
x=rng.uniform(-100,200,(100,2));b0=np.array([7.,11.]);bk=np.array([31.,19.]);s=np.array([.87,1.13]);o=np.array([3.,8.])
roundtrip=float(abs(inverse(transition(x,b0,bk,s,o),b0,bk,s,o)-x).max());assert roundtrip<1e-12
degenerate=float(abs(transition(x,b0,bk,s,np.zeros(2))-(bk+s*(x-b0))).max());assert degenerate==0
toy=float(transition(np.array([75.,0.]),np.zeros(2),np.zeros(2),np.ones(2),np.zeros(2))[0]);assert toy==75 and .8*75==60
save('coordinate_math.json',dict(passed=True,inverse_max_error=roundtrip,equal_transform_max_error=degenerate,
    toy=dict(old_point=75,box_stretch=60,full_scale=toy),theorem='At least two independent reference displacements imply equality of both affine maps only if sg=sbox and og=0. This is not a measured physical scale.'))
records=[];max_roundtrip=0.
for source in sorted((NATIVE/'rollouts/C34').glob('*.events.json.gz')):
    video=source.name.split('.events')[0]
    with gzip.open(source,'rt',encoding='utf8') as f:events=json.load(f)['events']
    for event in events:
        for row in event['proposals']:
            if row['reason'] not in ('head_accept','head_refuse'):continue
            metadata=event['rows'][row['row']]
            assert metadata['tid']==row['old_id'] and row['reference_frame']==metadata['history'][-1]['frame']
            g=geometric_match(metadata['mean'],metadata['covariance'],metadata['history'],row['current_crop'],row['current_box'],geom)
            assert g['valid']
            old=np.array(row['reference_box']);box=np.array(row['current_box']);rc=np.array(row['reference_crop']);ct=np.array(row['current_crop'])
            sg=np.array(g['scale']);og=np.array(g['offset']);sbox=(box[2:]-box[:2])/(old[2:]-old[:2])
            assert (sg>0).all() and (og>=0).all()
            offsets=offset_axes*np.maximum((rc[2:]-rc[:2])/geom['eye_size'],(ct[2:]-ct[:2])/geom['eye_size'])
            original=relative_template_plan(reader,old,rc,box,ct,offsets)
            assert len(original['cells'])==row['full_cells']
            x=reader.points*(rc[2:]-rc[:2])/reader.eye_size+rc[:2]
            base_box=box[:2]+(x-old[:2])*sbox
            base_geom=transition(x,old[:2],box[:2],sg,og)
            independent=(sg-sbox)*(x-old[:2])-sg*og
            assert np.max(abs((base_geom-base_box)-independent))<1e-10
            inverse_error=float(abs(inverse(base_geom,old[:2],box[:2],sg,og)-x).max())
            max_roundtrip=max(max_roundtrip,inverse_error)
            assert inverse_error<1e-10
            refmask=((x>=old[:2])&(x<=old[2:])).all(1)
            visible=((base_geom>=box[:2])&(base_geom<=box[2:])).all(1)
            trial=base_geom[:,None,:]+offsets[None,:,:]
            query=(trial-ct[:2])/((ct[2:]-ct[:2])/reader.eye_size)
            in_eye=(reader.mesh.find_simplex(query)>=0).all(1)
            cells=np.flatnonzero(refmask&visible&in_eye)
            # Contrast the tempting but previously harmful all-shift box intersection.
            all_box=((trial>=box[:2])&(trial<=box[2:])).all((1,2))
            eroded=np.flatnonzero(refmask&visible&in_eye&all_box)
            delta=base_geom[original['cells']]-base_box[original['cells']]
            spacing=geom['spacing']*(ct[2]-ct[0])/geom['eye_size']
            norm=np.linalg.norm(delta,axis=1)/spacing
            residual=delta-delta.mean(0)
            search_hull=Delaunay(offsets)
            out_search=(search_hull.find_simplex(delta)<0)
            records.append(dict(video=video,frame=event['frame'],old_id=row['old_id'],raw_id=row['raw_id'],
                reference_frame=row['reference_frame'],head_accepted=row['accepted'],logit=row['logit'],
                scale_geom=sg.tolist(),scale_box=sbox.tolist(),offset_reference_pixels=og.tolist(),
                scale_ratio=(sg/sbox).tolist(),geometry_residual=g['residual'],original_cells=len(original['cells']),
                reference_cells=int(refmask.sum()),visible_before_eye_cells=int((refmask&visible).sum()),
                aligned_cells=cells.tolist(),aligned_supported=len(cells)>=3,
                all_shift_box_eroded_cells=len(eroded),shared_cells=len(set(cells)&set(original['cells'])),
                delta_spacing_median=float(np.median(norm)),delta_spacing_max=float(norm.max()),
                delta_pixel_max=float(np.linalg.norm(delta,axis=1).max()),
                residual_after_translation_spacing_rms=float(np.sqrt((residual*residual).sum(1).mean())/spacing),
                cells_displacement_outside19search=int(out_search.sum()),
                current_spacing_pixels=float(spacing),roundtrip_max_error=inverse_error))
    print('COORDINATES',video,len(records),flush=True)
assert len(records)==213,len(records)
# Labels and actual future IDs are attached only after all transform/support choices.
fate=read(NATIVE/'id_fate.json')
lookup={(r['video'],r['frame'],r['old_id'],r['raw_id']):r for r in fate['records'] if r['arm']=='C34'}
for row in records:
    f=lookup[(row['video'],row['frame'],row['old_id'],row['raw_id'])]
    row.update(kind=f['kind'],global_selected=f['global_selected'],old_id_emitted=f['old_id_emitted_on_candidate'])
known=[r for r in records if r['kind']!='unknown'];assert len(known)==6
summary=dict(records=len(records),direct_known=len(known),unknown=len(records)-len(known),
    support_at_least_three=sum(r['aligned_supported'] for r in records),
    nonzero_coordinate_difference=sum(r['delta_pixel_max']>1e-9 for r in records),
    any_cell_outside_old_search=sum(r['cells_displacement_outside19search']>0 for r in records),
    median_of_median_spacing_displacement=float(np.median([r['delta_spacing_median'] for r in records])),
    median_reference_cells=float(np.median([r['reference_cells'] for r in records])),
    median_original_cells=float(np.median([r['original_cells'] for r in records])),
    median_aligned_cells=float(np.median([len(r['aligned_cells']) for r in records])),
    max_roundtrip_error=max_roundtrip)
for name,digest in p['sources'].items():assert sha(ROOT/name)==digest,name
assert 'flyvis' not in sys.modules
save('coordinate_report.json',dict(passed=True,summary=summary,known=known,records=records,
    protocol_sha256=sha(OUT/'coordinate_protocol.json'),no_maps_scores_fits_or_rollouts=True,
    interpretation='Coordinate gate only; nonzero deformation and supported cells cannot establish identity information, calibration or actual old-ID improvement.'))
print('SUMMARY',json.dumps(summary),flush=True)
