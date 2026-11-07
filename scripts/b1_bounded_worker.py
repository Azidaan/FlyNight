"""Независимый ограничитель Б1 с идентичностью PID и дочернего Python."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time
import psutil

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', required=True)
    parser.add_argument('--script', required=True)
    parser.add_argument('--prefix', required=True)
    args = parser.parse_args()
    folder = (ROOT / args.folder).resolve()
    folder.relative_to(ROOT.resolve())
    protocol = json.loads((folder / 'protocol.json').read_text(encoding='utf-8-sig'))
    deadline = datetime.fromisoformat(protocol['deadline']).timestamp()
    assert time.time() < deadline
    owned = {}
    def record(name, data):
        (folder / (args.prefix + '_' + name + '.json')).write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
    record('status', dict(state='starting', deadline=protocol['deadline']))
    with (folder / (args.prefix + '.stdout.log')).open('w', encoding='utf-8') as stdout, (folder / (args.prefix + '.stderr.log')).open('w', encoding='utf-8') as stderr:
        process = subprocess.Popen([sys.executable, '-u', str(ROOT / args.script), '--run'], cwd=ROOT,
                                   stdout=stdout, stderr=stderr, creationflags=subprocess.CREATE_NO_WINDOW)
        root = psutil.Process(process.pid)
        root.nice(psutil.IDLE_PRIORITY_CLASS)
        owned[root.pid] = root.create_time()
        timed_out = False
        record('launch', dict(pid=root.pid, created=root.create_time(), deadline=protocol['deadline']))
        while process.poll() is None:
            try:
                for child in root.children(recursive=True):
                    owned[child.pid] = child.create_time()
            except psutil.NoSuchProcess:
                pass
            if time.time() >= deadline - 2:
                timed_out = True
                break
            time.sleep(.5)
        for pid, created in reversed(list(owned.items())):
            try:
                child = psutil.Process(pid)
                if abs(child.create_time() - created) < .01 and child.is_running():
                    child.kill()
            except psutil.NoSuchProcess:
                pass
        process.wait(timeout=5)
        time.sleep(.2)
        alive = []
        for pid, created in owned.items():
            try:
                child = psutil.Process(pid)
                if abs(child.create_time() - created) < .01 and child.is_running():
                    alive.append(pid)
            except psutil.NoSuchProcess:
                pass
        result = dict(exit_code=process.returncode, timed_out=timed_out, all_owned_ended=not alive,
                      owned=owned, remaining_owned=alive, updated=datetime.now(timezone.utc).isoformat())
        record('controller', result)
        print(json.dumps(result))
        return 0 if process.returncode == 0 and not timed_out and not alive else 1


if __name__ == '__main__':
    raise SystemExit(main())
