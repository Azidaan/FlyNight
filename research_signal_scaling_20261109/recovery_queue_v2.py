"""Correct only the Windows text encoding before the recovery spawns a worker."""
import json,hashlib,psutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=Path(__file__).resolve().parent
source=OUT/'recovery_queue.py';code=source.read_text(encoding='utf8')
assert code.count(".read_text())['passed']")==1
code=code.replace(".read_text())['passed']",".read_text(encoding='utf8'))['passed']")
code=code.replace('recovery_status.json','recovery_v2_status.json').replace('recovery_executor.json','recovery_v2_executor.json').replace("'recovery_'+name","'recovery_v2_'+name")
if not (OUT/'recovery_v2_protocol.json').exists():
    old=json.loads((OUT/'recovery_executor.json').read_text(encoding='utf8'))
    try:assert psutil.Process(old['pid']).create_time()!=old['birth']
    except psutil.NoSuchProcess:pass
    assert not list(OUT.glob('recovery_*.started.json')),'A worker may have been launched'
    with (OUT/'recovery_v2_protocol.json').open('x',encoding='utf8') as f:json.dump(dict(correction='Explicit UTF8 when reading the existing preflight receipt. Prior queue stopped before any child spawn; numerical recovery and scientific inputs unchanged.',sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [source,Path(__file__),OUT/'recovery_launch.err.log',OUT/'recovery_protocol.json']}),f,indent=2)
exec(compile(code,str(source),'exec'))
