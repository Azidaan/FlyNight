"""Исправить только CPU-загрузку; повтор GPU ещё не запускался в recovery1."""
from pathlib import Path
import json
import sys
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'runs/flyvis_y1_latency_20261104'
previous = json.loads((OUT / 'recovery1/chain_result.json').read_text())
assert not previous['complete'] and previous['failed_stage'] == 'reference'
items = [(int(pid), birth) for pid, birth in previous['all_owned'].items()]
item = json.loads((OUT / 'recovery1/controller_launch.json').read_text());items.append((item['pid'], item['created']))
items += [(item['pid'], item['created']) for item in json.loads((OUT / 'recovery1/agent_launch_verified.json').read_text())['launches']]
for pid, birth in items:
    try:assert abs(psutil.Process(pid).create_time() - birth) > .01, pid
    except psutil.NoSuchProcess:pass
source = (ROOT / 'scripts/control_flyvis_y1_latency_recovery1_20261104.py').read_text(encoding='utf-8')
source = source.replace('recovery1/', 'recovery2/').replace("OUT / 'recovery1'", "OUT / 'recovery2'")
old = "('reference', 'scripts/flyvis_y1_latency_20261104.py', ['--stage', 'reference'])"
new = "('reference', 'scripts/flyvis_y1_cpu_reference_recovery1_20261104.py', [])"
assert old in source
source = source.replace(old, new)
folder = OUT / 'recovery2';folder.mkdir(exist_ok=False)
(folder / 'controller_source.py').write_text(source, encoding='utf-8')
(folder / 'previous_cpu_failure_verified.json').write_text(json.dumps(dict(owned_ended=items, previous_code=previous['stages'][0]['exit_code'],
    no_gpu_retry_started=True), indent=2) + '\n', encoding='utf-8')
exec(compile(source, str(folder / 'controller_source.py'), 'exec'), dict(__name__='__main__', __file__=str(ROOT / 'scripts/control_flyvis_y1_latency_recovery1_20261104.py')))
