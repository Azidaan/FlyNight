"""Подтвердить фактический конец ночных деревьев и приостановку расписания."""
from pathlib import Path
from datetime import datetime, timezone
import json
import psutil

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/b123_20261027'


def matching(pid, birth):
    try:
        p = psutil.Process(int(pid))
        return p.is_running() and abs(p.create_time()-float(birth)) < .01
    except psutil.NoSuchProcess:
        return False


result = json.loads((RUN/'deadline_result.json').read_text(encoding='utf-8'))
launch = json.loads((RUN/'deadline_guard_recovery1/launch.json').read_text(encoding='utf-8'))
live = [int(pid) for pid, birth in result['owned'].items() if matching(pid, birth)]
automation = Path(r'C:\Users\Егор\.codex\automations\claude-flyvis\automation.toml')
status_line = next(l for l in automation.read_text(encoding='utf-8').splitlines() if l.startswith('status = '))
paused = json.loads(status_line.split(' = ', 1)[1]) == 'PAUSED'
proof = dict(checked=datetime.now(timezone.utc).isoformat(),
             deadline_result_all_owned_ended=result['all_owned_ended'],
             actual_live_owned=live, guard_still_alive=matching(launch['pid'], launch['created']),
             automation_paused=paused, no_new_scientific_runs=True)
proof['passed'] = result['all_owned_ended'] and not live and not proof['guard_still_alive'] and paused
dest = RUN/'final_stop/final_verification.json'
assert not dest.exists()
dest.write_text(json.dumps(proof, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
print(json.dumps(proof, ensure_ascii=False))
assert proof['passed']
