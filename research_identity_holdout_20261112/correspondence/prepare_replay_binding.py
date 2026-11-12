"""Bind completed actual-core artifacts to the next parent-controlled CPU stage."""
import json,hashlib
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()
p=read(OUT/'protocol.json');core=read(OUT/'core_result.json');assert core['passed']
for n,h in p['inputs'].items():assert sha(ROOT/n)==h,n
needed={c['endpoints'][0]['key'] for c in p['cases']};records=core['records'];assert {r['key'] for r in records}==needed
assert len(records)==2*len(needed) and core['new_core_passes']==2*len(needed)
derived={str((OUT/'core_result.json').relative_to(ROOT)).replace('\\','/'):sha(OUT/'core_result.json')}
for record in records:
    path=ROOT/record['path'];assert sha(path)==record['sha256'];arr=np.load(path,mmap_mode='r',allow_pickle=False);assert arr.shape==(721,34) and arr.dtype==np.float32 and np.isfinite(arr).all();derived[record['path']]=record['sha256']
    if record['mode']=='index':assert record['original_parity']<=1e-4
    else:assert record['mode']=='terminal_hold'
for key in needed:assert sorted(r['mode'] for r in records if r['key']==key)==['index','terminal_hold']
out=dict(prepared=datetime.now(ZoneInfo('Europe/Moscow')).isoformat(),approved=False,authorization='Ready for parent GO only; this preparation computes no Census/head/appearance outcomes.',protocol_path='research_identity_holdout_20261112/correspondence/protocol.json',protocol_sha256=sha(OUT/'protocol.json'),preparation_script_sha256=sha(Path(__file__)),completed_reference_contexts=len(needed),completed_core_passes=core['new_core_passes'],core_inputs=derived,entrypoints=dict(replay=['.venv/Scripts/python.exe','research_identity_holdout_20261112/correspondence/replay_worker.py'],audit=['.venv/Scripts/python.exe','research_identity_holdout_20261112/correspondence/audit.py']),parent_GO_required='Create replay_GO.json approved:true binding this protocol_sha256 and core_result hash, then enqueue CPU replay and independent audit. No standalone GPU call.')
with (OUT/'replay_binding.json').open('xb') as f:f.write((json.dumps(out,ensure_ascii=False,indent=2).rstrip()+'\n').encode('utf8'))
print('REPLAY_BINDING_READY',len(needed),core['new_core_passes'])
