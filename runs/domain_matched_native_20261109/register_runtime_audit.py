"""Pin an independent journal audit without altering the native passport."""
from evaluation_common import *
os.environ['CUDA_VISIBLE_DEVICES']=''
files=[Path(__file__),OUT/'audit_runtime.py',OUT/'evaluation_common.py',OUT/'protocol.json',OUT/'execution_protocol.json',OUT/'evaluation_protocol.json',OUT/'deployment.json',OUT/'zero_complete.json',
    ROOT/'research_domain_matched_reader_20261109/audit.json',ROOT/'research_domain_matched_reader_20261109/deployment.json',ROOT/'tracking/channel_identity_head.py',ROOT/'tracking/spatial_identity_assignment.py']
for path in (ROOT/'.venv/Lib/site-packages/ultralytics/trackers/utils').rglob('*.py'):files.append(path)
write('audit_protocol.json',dict(registered=datetime.now(MSK).isoformat(),sources={p.relative_to(ROOT).as_posix():sha(p) for p in files},
    scope='Independently recompute saved geometry,linear scores and full global assignments on the100newnativevideos. Do not refit,extractmaps,rerun tracking,change thresholds or repeat oldfullaudits. Check both geometry-only and signal arms. Record unexpected float32 delivery cases without modifying the run.'))
print('RUNTIME_AUDIT_REGISTERED',flush=True)
