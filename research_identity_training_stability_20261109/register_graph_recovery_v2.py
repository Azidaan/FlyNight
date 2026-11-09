"""Recover post-repair provenance without repeating the already-applied edits."""
import json,hashlib
from pathlib import Path
from datetime import datetime,timezone,timedelta
ROOT=Path(__file__).resolve().parents[1];OUT=Path(__file__).resolve().parent
def digest(b):return hashlib.sha256(b).hexdigest()
def sha(p):return digest(p.read_bytes())
tests=json.loads((OUT/'final_catalog_tests_receipt.json').read_text(encoding='utf8'));changed={}
path=ROOT/'experiments/useful_literature.json';current=path.read_bytes();data=json.loads(current);entry=next(x for x in data['sources'] if x['id']=='koh2017_training_influence')
assert entry.pop('doi')=='не указан на официальной странице PMLR'
prior=(json.dumps(data,ensure_ascii=False,indent=2)+'\n').encode();name=path.relative_to(ROOT).as_posix();assert digest(prior)==tests['stable_sources'][name]
changed[name]=dict(before=digest(prior),after=digest(current),change='Only omitted doi field added, verified by reconstructing exact prior bytes.')
path=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';current=path.read_bytes();nl=b'\r\n' if b'\r\n' in current else b'\n';marker='## Журнал изменений'.encode()+nl+nl
prefix,tail=current.split(marker,1);end=tail.index(nl+nl+b'**');entry=tail[:end]
assert 'Исправлено обязательное поле записи литературы перед сборкой графа'.encode() in entry.split(nl,1)[0]
prior=prefix+marker+tail[end+2*len(nl):];name=path.relative_to(ROOT).as_posix();assert digest(prior)==tests['stable_sources'][name]
changed[name]=dict(before=digest(prior),after=digest(current),change='One factual infrastructure entry prepended; exact prior bytes reconstructed and checked.')
for name,h in tests['stable_sources'].items():assert sha(ROOT/name)==(changed[name]['after'] if name in changed else h),name
sources={n:sha(OUT/n) for n in ['comparison.json','audit.json','recording.json','РЕЗУЛЬТАТ.md','final_catalog_tests_receipt.json','final_catalog_tests.log']}
with (OUT/'graph_repair_attempt.json').open('x',encoding='utf8') as f:json.dump(dict(error='First metadata repair completed its two intended edits, then provenance check used Windows backslashes against POSIX receipt keys and failed. A subsequent graph-only worker failed because the protocol had not yet been written. Neither operation is reported as passed; edits are not repeated.',repair_script_sha256=sha(OUT/'repair_literature_schema.py'),failed_followup_guard='0b796755bede41bc91652141d7cc564e',current_changes=changed),f,ensure_ascii=False,indent=2)
with (OUT/'graph_recovery_protocol.json').open('x',encoding='utf8') as f:json.dump(dict(registered=datetime.now(timezone(timedelta(hours=3))).isoformat(),changed=changed,scientific_sources=sources,tests_reused=58,failure_guard='0042932f54414430b91299af5b65d890',scope='Only source-schema repair and protected graph reconstruction. Exact old bytes reconstructed to prove the two applied edits. Reuse unchanged catalog tests; rerun catalog validator. No fits/core/video/scientific results repeated.'),f,ensure_ascii=False,indent=2)
print('GRAPH_RECOVERY_V2_PROVENANCE_VERIFIED',flush=True)
