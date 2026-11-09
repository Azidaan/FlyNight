"""Produce an inspectable corrected analyser; retain the failed original source."""
from pathlib import Path
folder=Path(__file__).resolve().parent
text=(folder/'id_fate.py').read_text(encoding='utf8')
def replace(old,new):
    global text
    assert text.count(old)==1,old
    text=text.replace(old,new)
replace("start('cpu',timeout=2400)","start('cpu',timeout=2400)\nfrom status_write_recovery import stable_status\nstatus=stable_status\nfrom repair_public_provenance import repair\nverify(read(OUT/'fate_recovery_protocol.json'))")
replace('all_records=[];summary={};sources={};future_cache={}',
    'all_records=[];summary={};sources={};future_cache={};provenance_audits=[]')
replace("        assignments=collections.defaultdict(list)\n        for item in data['assignments']:assignments[(item['frame'],item['tid'])].append(item)",
    "        repaired,repair_counts,repair_witnesses=repair(raw,flat,data['assignments'])\n        provenance_audits.append(dict(video=video,arm=arm,counts=repair_counts,witnesses=repair_witnesses))\n        assignments=collections.defaultdict(list)\n        for item in repaired:assignments[(item['frame'],item['tid'])].append(item)")
replace("old_id_emitted_on_candidate=any(x['emitted'] and x['raw_id']==rid for x in current),",
    "old_id_emitted_on_candidate=(True if any(x['emitted'] and x['raw_id']==rid for x in current) else (None if any(x['emitted'] and x['raw_id'] is None for x in current) else False)),")
replace("if len(reference)!=1 or not reference[0]['emitted']:",
    "if len(reference)!=1 or not reference[0]['emitted'] or reference[0]['raw_id'] is None:")
replace("'reference_not_a_unique_published_observation'","'reference_not_a_unique_published_observation_with_verified_raw'")
replace("int(record['old_id_emitted_on_candidate'])","int(record['old_id_emitted_on_candidate'] is True)")
replace("decision=='refused' and not record['old_id_emitted_on_candidate']:","decision=='refused' and record['old_id_emitted_on_candidate'] is False:")
replace("verify(phase)\nwrite('id_fate.json'", "verify(phase);verify(read(OUT/'fate_recovery_protocol.json'))\nwrite('id_fate.json'")
replace("evaluation_sha256=sha(OUT/'evaluation_protocol.json'),",
    "evaluation_sha256=sha(OUT/'evaluation_protocol.json'),provenance_audits=provenance_audits,\n    recovery_sha256=sha(OUT/'fate_recovery_protocol.json'),original_metrics_unchanged=True,")
dest=folder/'id_fate_v2.py'
with dest.open('xb') as f:f.write(text.encode('utf8'))
print('Corrected analyser saved; original immutable source retained')
