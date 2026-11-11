"""Check physical-correspondence readiness and replay identity controls on frozen maps.

Bounding boxes are not physical landmarks. Do not substitute a box warp for the
registered physical-detail intervention when the latter is not identifiable.
"""
from common import *
import numpy as np
from tracking.observed_spatial_reader import ObservedSpatialReader
from tracking.full_template_correspondence import relative_template_plan
from tracking.hex_linear_interpolation import sample_linear
from tracking.channel_agreement_correspondence import agreement_channels
p=verify();assert read(OUT/'stage2.json')['passed'];assert read(OUT/'frame_manifest.json')['passed']
ag=read(ROOT/'research_agreement_readout_20261112/protocol.json');pred=read(ROOT/'research_agreement_readout_20261112/predictions.json')
fit=next(f for f in pred['fits'] if f['video']=='FINAL' and f['model']=='C34__agreement');theta=np.array(fit['fit']['theta'])
geom=read(ROOT/'runs/z13_local_peak_20261021/geometry.json');reader=ObservedSpatialReader(geom['points'],geom['eye_size'],[1280,720])
rows=[];maxerr=0.
for r in ag['screen']:
    source=next(s for s in p['selected'] if s['old_id']==r['old_id'] and s['video']==r['video'] and s['frame']==r['frame'])
    maps=[]
    for key in r['keys']:
        item=next(x for x in ag['snapshots'][key]['files'] if x['slot']=='C34')
        assert item['array'] is None;maps.append(np.load(ROOT/item['path'],allow_pickle=False).astype(float))
    plan=relative_template_plan(reader,r['reference_box'],r['reference_crop'],r['current_box'],r['current_crop'],r['offsets'])
    cells=plan['cells'];winner=r['old_variants']['C34']['winner'];query=plan['query'][:,winner]
    a=maps[0][cells];b=sample_linear(reader.mesh,maps[1],query)
    original_hashes=[hashlib.sha256(m.tobytes()).hexdigest() for m in maps]
    identity=maps[1].copy();assert np.array_equal(identity,maps[1]);b2=sample_linear(reader.mesh,identity,query.copy());assert np.array_equal(b,b2)
    def score(bb):
        q=agreement_channels(a,bb,mean_penalty=True)
        g=(np.array(r['geometry'])-fit['normalization']['mean'])/fit['normalization']['scale']
        x=np.r_[g,(q-fit['signal_mean'])/fit['signal_scale']]
        return float(x@theta[:-1]+theta[-1])
    z=score(b);assert score(b2)==z
    expected=next(x for x in pred['screen'] if x['old_id']==r['old_id'])['scores']['C34__agreement'];err=abs(z-expected);assert err<1e-12;maxerr=max(maxerr,err)
    rc=np.array(r['reference_crop']);cc=np.array(r['current_crop']);oldpoints=np.array(reader.points)[cells]*(rc[2:]-rc[:2])/reader.eye_size+rc[:2];newpoints=query*(cc[2:]-cc[:2])/reader.eye_size+cc[:2]
    rg=np.array(source['reference_hits'][0]['box'] if 'box' in source['reference_hits'][0] else next(s for s in read(OUT/'stage1.json')['records'] if s['old_id']==r['old_id'] and s['frame']==r['frame'])['reference_hits'][0]['box'])
    sg=next(s for s in read(OUT/'stage1.json')['records'] if s['old_id']==r['old_id'] and s['frame']==r['frame'])
    cg=np.array(sg['current_hits'][0]['box'])
    inside=lambda x,bb:((x>=bb[:2])&(x<=bb[2:])).all(1)
    oldinside=inside(oldpoints,rg);newinside=inside(newpoints,cg)
    assert original_hashes==[hashlib.sha256(m.tobytes()).hexdigest() for m in maps]
    rows.append(dict(video=r['video'],old_id=r['old_id'],frame=r['frame'],label=r['label'],cells=len(cells),winner=winner,
      original_score=z,identity_control_score=score(b2),stored_score=expected,
      reference_inside_annotated_box=int(oldinside.sum()),current_inside_candidate_GT_box=int(newinside.sum()),both_inside_their_endpoint_GT_boxes=int((oldinside&newinside).sum()),
      reference_pixel_points=oldpoints,current_pixel_points=newpoints,reference_GT_box=rg,current_candidate_GT_box=cg,
      physical_correspondence_ready=False,
      missing_evidence='No independently validated physical-detail correspondences with localization uncertainty; available GT supplies instance bounding boxes only. Low-resolution/occluded source frames do not supply exact dense correspondences.',
      causal_intervention_executed=False,matched_displacement_control_executed=False,
      warning='Inside a box does not prove same physical detail; outside counts do not prove causal score failure. Do not call a GT-box affine warp corrected physical correspondence.'))
verify();save('stage3.json',dict(identity_checks_passed=True,physical_intervention_completed=False,records=rows,max_saved_score_error=maxerr,
 script_sha256=sha(__file__),model='C34__agreement FINAL',new_models=0,new_maps=0,tracker_calls=0,
 decision='Do not run a mislabeled physical-correspondence intervention. Exact visual root causes remain unresolved pending independent point correspondence evidence.'))
journal('Третий этап: исходные кадры пяти случаев просмотрены, неизменяющий контроль на готовых картах воспроизвёл сохранённые оценки. Независимой разметки физических деталей нет; рамки объектов ею не являются. Причинная подстановка соответствий не выполнена, чтобы не выдать приближение за точное доказательство. Источник research_failure_localization_20261112/stage3.json.')
print(json.dumps([{k:r[k] for k in ['old_id','cells','reference_inside_annotated_box','current_inside_candidate_GT_box','both_inside_their_endpoint_GT_boxes','original_score']} for r in rows]))
