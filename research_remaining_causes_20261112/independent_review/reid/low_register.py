"""Register all saved raw-car candidates, preserving the previous diagnostic set."""
from common import *
import numpy as np
old=verify();oldr=read(OUT/'result.json');cases=[]
for c in old['cases']:
    cc=dict(c);entry=old['videos'][c['video']]
    with np.load(ROOT/entry['detections']) as z:
        ids=np.flatnonzero((z['frame_idx']==c['current_frame'])&(z['cls']==2));objs=objects(c['video'],c['current_frame'])
        cc['candidates']=[dict(raw_id=int(i),box=z['xyxy'][i].astype(float).tolist(),confidence=float(z['conf'][i]),cls=int(z['cls'][i]),hits=hits(z['xyxy'][i],objs),previous_diagnostic_ge_point1=bool(z['conf'][i]>=.1),native_first=bool(z['conf'][i]>=.25),native_second=bool((z['conf'][i]>.1)&(z['conf'][i]<.25))) for i in ids]
        assert set(r['raw_id'] for r in c['candidates'])<=set(map(int,ids))
    cases.append(cc)
names=[str((OUT/n).relative_to(ROOT)).replace('\\','/') for n in ['protocol.json','result.json','descriptors.npz','common.py','low_register.py','low_worker.py','low_audit.py']]
names+=['runs/domain_matched_native_20261109/protocol.json','runs/domain_matched_native_20261109/runtime_adapter.py','runs/full_template_native_20261109/native_runtime.py','tracking/offline_association.py','.venv/Lib/site-packages/ultralytics/trackers/byte_tracker.py']
save('low_protocol.json',dict(registered=now(),inputs={n:sha(ROOT/n) for n in names},original_protocol_sha256=sha(OUT/'protocol.json'),cases=cases,videos=old['videos'],environment=old['environment'],deadline=old['deadline'],batch_size=16,expected_raw_frame_hashes=old['expected_raw_frame_hashes'],method='Allsavedrawcarclass2detections at all5currentframes, no additional confidence filter and no operational threshold change. Reuse5original frozen reference descriptors, recompute5currentframefeaturemaps in oneGPUowner, embed allboxes16perbatch. Reproduce77previouscandidate descriptors/scores before newinterpretation. Keep prior conf>=.1 diagnostic set, actual native_first conf>=.25 and native_second .1<conf<.25 as three distinct flags. No new threshold, fit, tracker, or appearanceconditionselection.',acceptance=['Originalprotocol/sources/results/descriptors unchanged.','Allrawcurrentcarrows retained. Previous77descriptors match within1e-6 and oldscores reproduce within1e-6.','Independent CPUscalarcosines/ranks/identityavailability verified; unmatchedrows retained as unknown, not automatically foreign.'],limits='Posthoc followup motivated by raw574 ownGTdetections below original native lowgate. No threshold deployment, fulltracker, rejectionpolicy, HOTA, or physicalsurfacecorrespondence tested. Native firstpool is stricter than old conf>=.1 diagnostic candidate set.'))
print('LOW_PROTOCOL_PREPARED',sum(len(c['candidates']) for c in cases),sha(OUT/'low_protocol.json'))
