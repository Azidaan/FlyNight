"""Одна скрытая цепочка Я1 с независимой остановкой и учётом PID + birth."""
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
from scripts.flyvis_y1_latency_20261104 import OUT, write, protocol


def main():
    p = protocol()
    deadline = datetime.fromisoformat(p['hard_stop']).timestamp()
    me = psutil.Process();me.nice(psutil.IDLE_PRIORITY_CLASS)
    write('controller_launch.json', dict(pid=me.pid, created=me.create_time(), hard_stop=p['hard_stop']))
    stages = [('prepare', 'scripts/flyvis_y1_latency_20261104.py', ['--stage', 'prepare']),
              ('infer', 'scripts/flyvis_y1_latency_20261104.py', ['--stage', 'infer']),
              ('reference', 'scripts/flyvis_y1_latency_20261104.py', ['--stage', 'reference']),
              ('audit', 'scripts/audit_flyvis_y1_latency_20261104.py', [])]
    all_owned, reports = {}, []
    for stage, script, extra in stages:
        report_file = OUT / (stage + '_controller_result.json')
        if report_file.exists():
            previous = json.loads(report_file.read_text())
            assert previous['exit_code'] == 0 and previous['all_owned_ended']
            for pid, birth in previous['owned'].items():
                try:assert abs(psutil.Process(int(pid)).create_time() - birth) > .01
                except psutil.NoSuchProcess:pass
            reports.append(previous);continue
        assert time.time() < deadline
        owned = {};reason = None
        env = dict(os.environ)
        if stage != 'infer':env['CUDA_VISIBLE_DEVICES'] = ''
        with (OUT / (stage + '_stdout.log')).open('w', encoding='utf-8') as stdout, (OUT / (stage + '_stderr.log')).open('w', encoding='utf-8') as stderr:
            child = subprocess.Popen([sys.executable, '-u', str(ROOT / script), *extra], cwd=ROOT,
                     stdout=stdout, stderr=stderr, env=env, creationflags=subprocess.CREATE_NO_WINDOW)
            proc = psutil.Process(child.pid);proc.nice(psutil.IDLE_PRIORITY_CLASS)
            owned[proc.pid] = proc.create_time()
            write(stage + '_launch.json', dict(pid=proc.pid, created=proc.create_time()))
            started = time.time()
            while child.poll() is None:
                try:
                    for q in proc.children(recursive=True):owned[q.pid] = q.create_time()
                except psutil.NoSuchProcess:pass
                newest = max([started, *[q.stat().st_mtime for q in [OUT / 'status.json', OUT / (stage + '_stdout.log')] if q.exists()]])
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
        remaining = []
        for pid, birth in owned.items():
            try:
                q = psutil.Process(pid)
                if abs(q.create_time() - birth) < .01 and q.is_running():remaining.append(pid)
            except psutil.NoSuchProcess:pass
        result = dict(stage=stage, exit_code=child.returncode, stop_reason=reason, owned=owned,
                      remaining=remaining, all_owned_ended=not remaining, ended_utc=datetime.now(timezone.utc).isoformat())
        write(stage + '_controller_result.json', result)
        all_owned.update(owned);reports.append(result)
        if child.returncode or reason or remaining:
            write('chain_result.json', dict(complete=False, failed_stage=stage, stages=reports, all_owned=all_owned))
            return 1
    write('chain_result.json', dict(complete=True, stages=reports, all_owned=all_owned,
          all_owned_ended=all(r['all_owned_ended'] for r in reports), ended_utc=datetime.now(timezone.utc).isoformat()))
    return 0


if __name__ == '__main__':raise SystemExit(main())
