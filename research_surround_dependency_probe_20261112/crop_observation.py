from common import *
import numpy as np
p=verify();rows=[]
for c in p['cases']:
    transitions=[]
    for j in range(1,len(c['sequence'])):
        f0,b0=c['sequence'][j-1];f1,b1=c['sequence'][j];a=np.array(b0,float);b=np.array(b1,float)
        transitions.append(dict(from_frame=f0,to_frame=f1,from_crop=b0,to_crop=b1,center_displacement_image_pixels=((b[:2]+b[2:]-a[:2]-a[2:])/2).tolist(),size_ratio=((b[2:]-b[:2])/(a[2:]-a[:2])).tolist(),crop_changed=bool(np.any(a!=b))))
    rows.append(dict(old_id=c['case']['old_id'],transitions=transitions,reference_last_transition=transitions[c['endpoints'][0]['length']-2],current_last_transition=transitions[-1]))
source='runs/full_template_native_20261109/native_runtime.py';archived=read(ROOT/'runs/domain_matched_native_20261109/execution_protocol.json')['sources'];actual=sha(ROOT/source)
save('crop_observation.json',dict(created=now(),scope='Read-only observation of archived crop coordinates, not an intervention or measured source of identity errors. Image-crop changes do not measure object motion.',source=source,current_sha256=actual,archived_sha256=archived.get(source),archived_source_match=archived.get(source)==actual,records=rows,interpretation='Input crops may translate and rescale across observations. Native FeatureBank replays retinal arrays by cell index without a world-coordinate warp before recurrence. Output-map box alignment is downstream. Whether this caused own232 rejection is untested.'))
print([(r['old_id'],r['reference_last_transition']['crop_changed'],r['current_last_transition']['crop_changed']) for r in rows])
