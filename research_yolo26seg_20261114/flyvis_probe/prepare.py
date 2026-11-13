"""Register a bounded CPU-only mask readout before looking at predictions."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import hashlib,json,subprocess
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT=Path('E:/Dissertation')
OUT=Path(__file__).resolve().parent
VID='b1ca2e5d-84cf9134'
OLD=ROOT/'research_visible_part_2179_20261114'
for variable in ('TEMP','TMP','TMPDIR'):
    p=OUT/'tmp';p.mkdir(parents=True,exist_ok=True);os.environ[variable]=str(p)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def rel(p):return Path(p).resolve().relative_to(ROOT.resolve()).as_posix()
def write(name,value):
    with (OUT/name).open('x',encoding='utf8',newline='\n') as f:
        json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n')

stems={1159:'909e4901278076d36184690aae1bc505',1164:'84f51eeebf94b68441ebd1891e0f1383',1165:'d49679f47b3bf3e29cfcd479c04b23ec'}
records=[];sources={}
for frame,stem in stems.items():
    meta=ROOT/'runs/domain_matched_native_20261109/features'/VID/(stem+'.C34.json')
    array=meta.with_suffix('.npy');m=read(meta)
    assert m['slot']=='C34' and m['video']==VID and m['sequence'][-1][0]==frame
    assert sha(array)==m['sha256']
    original=ROOT/m['reused_from'];originalmeta=original.with_suffix('.json')
    assert sha(original)==m['reused_source_sha256']==sha(array)
    assert sha(originalmeta)==m['original_metadata_sha256']
    retinal=[]
    for time,crop in m['sequence']:
        key=hashlib.sha256(json.dumps([int(time),list(crop)],separators=(',',':')).encode()).hexdigest()[:32]
        options=[ROOT/n for n in m['retinal_sources'] if Path(n).stem==key]
        assert options,(frame,time,crop,key)
        path=options[0];assert sha(path)==m['retinal_sources'][rel(path)]
        retinal.append(dict(frame=time,crop=crop,path=rel(path),sha256=sha(path)))
        sources[rel(path)]=sha(path)
    for p in (meta,array,original,originalmeta):sources[rel(p)]=sha(p)
    records.append(dict(frame=frame,role='fit' if frame==1159 else 'evaluation',
        map=rel(array),metadata=rel(meta),crop=m['sequence'][-1][1],sequence=m['sequence'],retinal=retinal))
fixed=[ROOT/'AGENTS.md',ROOT/'runs/z13_local_peak_20261021/geometry.json',
    OLD/'mask_review/masks_reviewed.npz',OLD/'mask_review/masks_reviewed.json',
    ROOT/'runs/full_template_native_20261109/native_runtime.py',
    ROOT/'runs/full_template_native_20261109/sparse_views.py',
    ROOT/'tracking/z13_readout.py',ROOT/'tracking/flyvis_activity.py',
    ROOT/'tools/flyvis-runtime/flyvis/datasets/rendering/eye.py',
    ROOT/'scripts/z13_target_model.py',ROOT/'runs/z13_target_center_dev_20261018/C/normalization.json',
    ROOT/'runs/z13_target_center_dev_20261018/R1/normalization.json',Path(__file__)]
for frame in stems:fixed.append(OLD/'frames'/f'{frame}.png')
for p in fixed:sources[rel(p)]=sha(p)
protocol=dict(registered=datetime.now(ZoneInfo('Europe/Moscow')).isoformat(),
    state='ready',question='Can a frozen linear per-cell readout of archived FlyVis activity segment the two visible car cores, compared with exactly matched retinal-history features and raw-color/gray controls?',
    intended_action='Decide whether an object-mask readout from the existing FlyVis representation deserves a subsequent independent-video experiment; no production integration from this local result.',
    fit_frames=[1159],evaluation_frames=[1164,1165],video=VID,records=records,inputs=sources,
    classes=['other_known','far_red_core','near_black_core'],
    labels='Manual conservative cores already fixed before this experiment. Other_known=confident remaining scene OR occluder. Uncertain pixels excluded. For training only, resize known masks with INTER_AREA to391x391, average13x13 with zero exterior, sample721 archived receptor positions; accept a cell only when one known class covers at least90% of its input footprint. No threshold tuning.',
    reader=dict(kind='Class-balanced multiclass ridge regression with unpenalized intercept',regularization=1.0,selection='One fixed setting; no validation search',normalization='Training accepted-cell mean and SD only, zero-SD channels set1',decision='argmax of three scores; no probability threshold',features='No xy coordinates, frame index, crop box, instance IDs, or evaluation labels as input',parameters='3*(D+1) per arm; D34/34/1/1/3, same fitting rule'),
    arms={'FlyVis34':'Archived721x34 normalized frozen neural maps',
          'Retina_history34':'17 fixed EMA time constants geomspace1.5..32 plus17current-minus-EMA from SAME archived retinal sequence',
          'Retina_current1':'Last saved retinal721-vector, exactly the input to core',
          'Raw_gray1':'Current original grayscale, same crop/resize/13x13average/grid',
          'Raw_RGB3':'Current original RGB, same crop/resize/13x13average/grid'},
    geometry='Archived points are integer xy on391 image. Cell input averages13x13 resized pixels centred there. Pixel-centre inverse is crop_origin+(point+.5)*crop_size/391-.5; pixel-area edges are crop_origin+index*crop_size/391. Regenerate BoxEye order without importing FlyVis; test geometry by independent area-overlap mapping of training-label footprints. Prediction rasterization is nearest cell inside convex hull only; outside hull is unobserved, not background.',
    extraction='No new FlyVis forward, GPU import, core training or tracker run. Validate all saved map and selected retinal hashes before reuse. Archived sequence omits frame1160; both FlyVis and history control retain exactly this frozen timing; do not reinterpret it as a continuous raw-frame simulation.',
    evaluation='Predict and save ALL evaluation per-cell scores and raster masks BEFORE opening1164/1165 manual-mask members. Evaluate confusion and per-core IoU over known pixels inside eye support; uncertain/outside excluded. Report counts and positive coverage per frame. Training labels1159 are not an independent test;1164/65 are same-scene, same-object transfer and are not independent generalization.',
    stop='If missing map/retinal provenance, nonfinite vectors, coordinate audit fails, or fewer than3training cells for any class: stop without fitting and report exact reason; no threshold relaxation.',
    acceptance=['Exact hashes/shapes/finite values and no changes to input assets.','No evaluation label or coordinate feature enters fit or prediction.','Both car classes IoU>=0.5 on both held-outframes required for local mask feasibility.','Additional FlyVis benefit requires better mean held-out per-core IoU than dimension-matched retinal-history control on BOTHframes; one-scene diagnostic only.'],
    visualization='Large actual four-panel result: original crop, manual known cores, predictedFlyVis, predictedmatched retinal-history control; train/evaluation frame labels, unknown region and eye support explicit. Additional rawcontrols in numeric results. Never substitute manual masks for predictions.',
    counterargument='Retina history or raw appearance may encode these parts just as well; manual conservative cores and same-object frames can overstate deployable segmentation. Coarse cells may miss thin visible regions.',
    literature=[dict(url='https://www.nature.com/articles/s41586-024-07939-3',role='FlyVis is trained for optical-flow; this motivates testing a new readout but does not establish segmentation ability.'),dict(url='https://openaccess.thecvf.com/content_ICCV_2019/papers/Bhat_Learning_Discriminative_Model_Prediction_for_Tracking_ICCV_2019_paper.pdf',role='Target-background discriminative readout precedent; this probe is simple fixed ridge, not a DiMP reproduction.')],
    bytes_estimate_under=20_000_000,new_core_calls=0,full_tracker_calls=0)
write('protocol.json',protocol)
print('PROTOCOL_READY',rel(OUT/'protocol.json'),sha(OUT/'protocol.json'))
