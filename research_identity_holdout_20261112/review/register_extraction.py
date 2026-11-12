"""Prepare shared A descriptor extraction protocol; no inference."""
import json,hashlib
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent;SHARED=OUT.parent/'shared';SHARED.mkdir(exist_ok=True)
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
plan=read(OUT/'A_descriptor_plan_v2.json');runtime=read(OUT/'A_runtime_manifest_v2.json');assert plan['runtime_manifest_sha256']==sha(OUT/'A_runtime_manifest_v2.json');assert len(runtime['events'])==54
inputs=dict(plan['source_sha256'])
for n in ['review/register_extraction.py','review/extract_A.py','review/audit_A_bank.py','review/A_descriptor_plan_v2.json','review/A_runtime_manifest_v2.json','review/anchored_readiness.json','review/proposed_selection_protocol_v2.json']:
 p=OUT.parent/n;inputs[p.relative_to(ROOT).as_posix()]=sha(p)
for n in ['weights/masa/masa_r50.pth','tracking/masa_r50.py','tracking/gpu_lock.py','research_remaining_causes_20261112/independent_review/reid/result.json','runs/flyvis_repairs_20261106/unification_20261107/system_memory_guard.py']:inputs[n]=sha(ROOT/n)
protocol=dict(registered=datetime.now(ZoneInfo('Europe/Moscow')).isoformat(),inputs=inputs,plan='research_identity_holdout_20261112/review/A_descriptor_plan_v2.json',runtime='research_identity_holdout_20261112/review/A_runtime_manifest_v2.json',original_state_source='research_remaining_causes_20261112/independent_review/reid/result.json',output='research_identity_holdout_20261112/shared/descriptors_A.npz',batch_size=16,deadline='2026-11-13T07:35:00+03:00',frames=len(plan['frames']),rois=sum(len(f['boxes']) for f in plan['frames'].values()),scope='ALL54strictAevents/all1211pastrowreferences+4916currentrawcars via388deduplicatedframes/5901uniqueRoIs. ALLnewcontexts, no reuseoldrepairvideoembeddings, noGT/readout/policyoutcomes. Stableframeledgerorderbatch16 fixedBEFOREforward; no silentboxclipping.',acceptance=['SamefrozenMASAstatehashasold, eval/inference only.','OneGPUowner canonicalmutex/systemreserve/deadline; sequentialdecodeonce/video.','DuplicatefirstnewROIexact,CPU/GPUpreprocess exact; shapesfiniteunitnorm.','Allsourcehashunchangedbeforeafter; saveexactrawpixelhash perframe. IndependentCPUbankauditverifiesallframekeys/slots/shapes/strictpastreferenceandpixelhashes withoutneuralforward.'],new_fits=0,tracker_calls=0,B_reserved_read=False)
with (SHARED/'extraction_protocol.json').open('x',encoding='utf8') as f:json.dump(protocol,f,ensure_ascii=False,indent=2,allow_nan=False)
print('A_EXTRACTION_PREPARED',sha(SHARED/'extraction_protocol.json'))
