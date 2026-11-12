from common import *
import numpy as np
from tracking.spatial_identity_assignment import geometric_match
from tracking.observed_spatial_reader import ObservedSpatialReader
from tracking.full_template_correspondence import relative_template_plan
p=verify();r=next(x for x in read(OUT/'eligibility.json')['records'] if x['new']==801);rec=read(OUT/'reconstruction_v2.json');row=next(x for x in rec['birth_states'] if x['new']==801)['older_states'][0];g=read(ROOT/'runs/z13_local_peak_20261021/geometry.json');native=read(ROOT/'runs/domain_matched_native_20261109/protocol.json');v=r['video'];rid=r['new_raw_id']
save('downstream_protocol.json',dict(registered=now(),posthoc=True,question='Would the single candidate29726 also fail later geometry/support/gap gates? The original count gate remains unchanged; this only inspects predicates that were not executed.',script_sha256=sha(__file__),no_head=True,no_maps=True,no_runtime_change=True))
with np.load(ROOT/native['videos'][v]['detections'],allow_pickle=False) as z:b=z['xyxy'][rid]
geometry=geometric_match(row['mean'],row['covariance'],r['history'],r['crop'],b,g);gap=r['frame']-r['history'][-1]['frame'];reader=ObservedSpatialReader(g['points'],g['eye_size'],[1280,720]);ref=r['history'][-1];rc=np.array(ref['crop']);ct=np.array(r['crop']);axes=np.array([(u,v) for u in range(-2,3) for v in range(-2,3) if max(abs(u),abs(v),abs(u+v))<=2]);offsets=(axes@np.array([[13.,6.5],[0.,13.]]))*np.maximum((rc[2:]-rc[:2])/g['eye_size'],(ct[2:]-ct[:2])/g['eye_size']);plan=relative_template_plan(reader,ref['box'],rc,b,ct,offsets,clip_to_candidate=True)
save('downstream.json',dict(passed=True,video=v,frame=r['frame'],old=773,new=801,geometry=geometry,reference_gap=gap,old_support_cells=len(plan['cells']),unique_reference_frames=len({x['frame'] for x in r['history']}),geometry_gate=geometry['valid'],gap_gate=gap in [5,6],support_gate=len(plan['cells'])>=3,scope='Unevaluated original predicates inspected conditionally. No head score or assignment outcome inferred.'))
print('DOWNSTREAM',geometry['valid'],gap,len(plan['cells']))
