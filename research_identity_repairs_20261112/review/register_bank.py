"""Metadata-only registration of exact reused targets and new row references."""
import json,hashlib,sys
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
snap=read(OUT/'assignment_snapshots.json');plan=read(OUT/'reference_extraction_plan.json');old=read(ROOT/'research_remaining_causes_20261112/independent_review/reid/result.json');low=read(ROOT/'research_remaining_causes_20261112/independent_review/reid/low_result.json');assert sha(OUT/'assignment_snapshots.json')==plan['snapshot_sha256'];targets=[]
for s in snap['snapshots']:
 targetrow=next(r for r in s['rows'] if r['tid']==s['target_tid']);assert targetrow['reference']['slot']==s['reference_slot'];slot=s['reference_slot'];oldslot=old['slots'][str(s['old_case'])]['reference'];frame=plan['frames'][slot[0]];assert oldslot[0]==slot[0]
 oldframe=next(f for f in old['frames'] if f['key']==oldslot[0]);assert frame['boxes'][slot[1]]==oldframe['source_boxes'][oldslot[1]]
 targets.append(dict(old_case=s['old_case'],target_slot=slot,source_slot=oldslot))
for key,frame in plan['frames'].items():
 for usages in frame['usages']:
  for use in usages:
   s=next(s for s in snap['snapshots'] if s['old_case']==use['old_case']);assert frame['frame']<s['frame']
names=['assignment_snapshots.json','reference_extraction_plan.json','register_bank.py','extract_bank.py','audit_bank.py'];inputs={str((OUT/n).relative_to(ROOT)).replace('\\','/'):sha(OUT/n) for n in names};inputs.update(snap['sources'])
for n in ['runs/flyvis_repairs_20261106/unification_20261107/system_memory_guard.py']:inputs[n]=sha(ROOT/n)
current=[dict(old_case=r['old_id'],key=str(r['old_id']),raw_ids=[q['raw_id'] for q in r['candidates']]) for r in low['records']]
p=dict(registered=datetime.now(ZoneInfo('Europe/Moscow')).isoformat(),inputs=inputs,plan='research_identity_repairs_20261112/review/reference_extraction_plan.json',target_reuse=targets,original_descriptor_path='research_remaining_causes_20261112/independent_review/reid/descriptors.npz',original_result_path='research_remaining_causes_20261112/independent_review/reid/result.json',current_descriptor_path='research_remaining_causes_20261112/independent_review/reid/low_descriptors.npz',current_ledger=current,frames=plan['frames'],deadline='2026-11-13T07:35:00+03:00',batch_size=16,method='Exactlastobservedbox+frame forALL107savedrowreferences,37uniquevideo-frames. Reuse5existingtargetvectors EXACT; encodeonlymissingrefs, one sharedimmutable backbone perneededfullframe in singleGPUowner. NewcompetitorRoIs fixedframeledgerorder batch16; no silentclipping. Reuseexisting222currentdescriptors, never inferGT intoextractor.',acceptance=['Modelstate hashes equaloldMASA; sourcefiles unchangedbefore/after.','Everyusage strictpast; everytargetvector exactlyequal sourcearray, shape/finite/unitnorm all107.','DuplicatedfirstnewROI exactlyequal; CPU/GPUpreprocessingexact; sequentialdecodeandpixelhashledgers.','Independent CPUbankaudit verifiesallslots/shapes/hash/targetreuse and rawframehashes, no independentneuralforward claim.'],state='prepared_waiting_parent_go')
with (OUT/'bank_protocol.json').open('x',encoding='utf8') as f:json.dump(p,f,ensure_ascii=False,indent=2,allow_nan=False)
print('BANK_PROTOCOL_PREPARED',len(plan['frames']),sum(len(f['boxes']) for f in plan['frames'].values()),sha(OUT/'bank_protocol.json'))
