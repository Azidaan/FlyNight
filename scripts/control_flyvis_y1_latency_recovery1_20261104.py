"""Один перезапуск после ожидания CUDA; готовые восемь ответов не повторяются."""
from pathlib import Path
from datetime import datetime, timezone
import json
import os
import subprocess
import sys
import time
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.flyvis_y1_latency_20261104 import OUT, write, protocol, sha


def assert_ended(items):
    for pid, birth in items:
        try:assert abs(psutil.Process(int(pid)).create_time() - birth) > .01, pid
        except psutil.NoSuchProcess:pass


def main():
    p = protocol();deadline = datetime.fromisoformat(p['hard_stop']).timestamp()
    old = json.loads((OUT / 'chain_result.json').read_text())
    assert not old['complete'] and old['failed_stage'] == 'infer'
    assert json.loads((OUT / 'prepare_controller_result.json').read_text())['exit_code'] == 0
    ended = [(int(pid), birth) for pid, birth in old['all_owned'].items()]
    for name in ['controller_launch.json']:
        m = json.loads((OUT / name).read_text());ended.append((m['pid'], m['created']))
    ended += [(m['pid'], m['created']) for m in json.loads((OUT / 'agent_launch_verified.json').read_text())['launches']]
    assert_ended(ended)
    stop = json.loads((OUT / 'infer_external_stop.json').read_text())
    for name, expected in stop['completed_output_markers'].items():assert sha(OUT / 'outputs' / name) == expected
    me = psutil.Process();me.nice(psutil.IDLE_PRIORITY_CLASS)
    write('recovery1/controller_launch.json', dict(pid=me.pid, created=me.create_time()))
    write('recovery1/authorization.json', dict(scope='Technical recovery of the same frozen first Y1 diagnostic; one GPU retry only',
          reused_complete_cases=len(stop['completed_output_markers']), old_processes_verified_ended=ended,
          old_chain_result_sha256=sha(OUT / 'chain_result.json'), old_infer_result_sha256=sha(OUT / 'infer_controller_result.json'),
          frozen_protocol_sha256=sha(OUT / 'protocol.json'), source_sha256=sha(Path(__file__))))
    stages = [('reference', 'scripts/flyvis_y1_latency_20261104.py', ['--stage', 'reference']),
              ('infer', 'scripts/flyvis_y1_latency_20261104.py', ['--stage', 'infer']),
              ('audit', 'scripts/audit_flyvis_y1_latency_20261104.py', [])]
    reports = [];all_owned = {}
    for stage, script, extra in stages:
        owned = {};reason = None;env = dict(os.environ)
        if stage != 'infer':env['CUDA_VISIBLE_DEVICES'] = ''
        logs = OUT / 'recovery1';logs.mkdir(exist_ok=True)
        with (logs / (stage + '_stdout.log')).open('w', encoding='utf-8') as stdout, (logs / (stage + '_stderr.log')).open('w', encoding='utf-8') as stderr:
            child = subprocess.Popen([sys.executable, '-u', str(ROOT / script), *extra], cwd=ROOT, env=env,
                     stdout=stdout, stderr=stderr, creationflags=subprocess.CREATE_NO_WINDOW)
            root = psutil.Process(child.pid);root.nice(psutil.IDLE_PRIORITY_CLASS);owned[root.pid] = root.create_time()
            write('recovery1/' + stage + '_launch.json', dict(pid=root.pid, created=root.create_time()))
            started = time.time()
            while child.poll() is None:
                try:
                    for q in root.children(recursive=True):owned[q.pid] = q.create_time()
                except psutil.NoSuchProcess:pass
                newest = max([started, *[q.stat().st_mtime for q in [OUT / 'status.json', logs / (stage + '_stdout.log')] if q.exists()]])
                if time.time() >= deadline or time.time() - newest > 600:
                    reason = 'hard_stop' if time.time() >= deadline else 'watchdog_600s'
                    for pid, birth in reversed(list(owned.items())):
                        try:
                            q = psutil.Process(pid)
                            if abs(q.create_time() - birth) < .01:q.kill()
                        except psutil.NoSuchProcess:pass
                    break
                time.sleep(.5)
            child.wait(timeout=20)
        assert_ended(list(owned.items()))
        result = dict(stage=stage, exit_code=child.returncode, stop_reason=reason, owned=owned,
                      all_owned_ended=True, ended_utc=datetime.now(timezone.utc).isoformat())
        write('recovery1/' + stage + '_controller_result.json', result)
        reports.append(result);all_owned.update(owned)
        if child.returncode or reason:
            write('recovery1/chain_result.json', dict(complete=False, failed_stage=stage, stages=reports, all_owned=all_owned))
            return 1
    write('recovery1/chain_result.json', dict(complete=True, stages=reports, all_owned=all_owned, all_owned_ended=True))
    return 0


if __name__ == '__main__':raise SystemExit(main())
