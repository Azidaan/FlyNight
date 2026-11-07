"""Повышенная точность эталона, без ослабления допуска и без повтора готовогоGPU."""
from pathlib import Path
import json
import sys
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'runs/flyvis_y1_latency_20261104'
previous = json.loads((OUT / 'recovery2/chain_result.json').read_text())
assert not previous['complete'] and previous['failed_stage'] == 'reference'
items = [(int(pid), birth) for pid, birth in previous['all_owned'].items()]
m = json.loads((OUT / 'recovery2/controller_launch.json').read_text());items.append((m['pid'],m['created']))
items += [(m['pid'],m['created']) for m in json.loads((OUT / 'recovery2/agent_launch_verified.json').read_text())['launches']]
for pid,birth in items:
    try:assert abs(psutil.Process(pid).create_time()-birth)>.01,pid
    except psutil.NoSuchProcess:pass
folder=OUT/'recovery3';folder.mkdir(exist_ok=False)
source=(ROOT/'scripts/control_flyvis_y1_latency_recovery1_20261104.py').read_text(encoding='utf-8')
source=source.replace('recovery1/','recovery3/').replace("OUT / 'recovery1'","OUT / 'recovery3'")
old="('reference', 'scripts/flyvis_y1_latency_20261104.py', ['--stage', 'reference'])"
assert old in source
source=source.replace(old,"('reference', 'scripts/flyvis_y1_cpu_reference_fp64_20261104.py', [])")
(folder/'controller_source.py').write_text(source,encoding='utf-8')
(folder/'previous_failure_verified.json').write_text(json.dumps(dict(old_processes_ended=items,numeric_tolerance_unchanged=1e-5,no_gpu_retry_started=True),indent=2)+'\n',encoding='utf-8')
exec(compile(source,str(folder/'controller_source.py'),'exec'),dict(__name__='__main__',__file__=str(ROOT/'scripts/control_flyvis_y1_latency_recovery1_20261104.py')))
