"""Make the already-registered CPU replay concrete; does not approve or launch."""
import json,hashlib
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()
p=read(OUT/'protocol.json');binding=read(OUT/'replay_binding.json');assert sha(OUT/'protocol.json')==binding['protocol_sha256'];inputs=dict(p['inputs']);inputs.update(binding['core_inputs'])
for path in [OUT/'protocol.json',OUT/'replay_binding.json',OUT/'prepare_cpu_queue.py']:
    inputs[str(path.relative_to(ROOT)).replace('\\','/')]=sha(path)
for n,h in inputs.items():assert sha(ROOT/n)==h,n
queue=dict(prepared=datetime.now(ZoneInfo('Europe/Moscow')).isoformat(),approval='PENDING: replay_GO.json must be explicitly provided by authorized coordinator. This file creates no GO and starts no process.',source_protocol_sha256=binding['protocol_sha256'],inputs=inputs,runner_sha256=sha(OUT.parent/'run_queue.py'),jobs=[dict(entry='research_identity_holdout_20261112/correspondence/replay_worker.py',timeout=1800),dict(entry='research_identity_holdout_20261112/correspondence/audit.py',timeout=1800)],deadline=p['deadline'],scope='Only original registered A CPU readout and independent audit, no new variants/GPU/core passes; B unopened.')
with (OUT/'ready_cpu_queue.json').open('xb') as f:f.write((json.dumps(queue,ensure_ascii=False,indent=2).rstrip()+'\n').encode('utf8'))
print('CPU_QUEUE_READY_NOT_APPROVED',len(inputs))
