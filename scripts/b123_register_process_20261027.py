"""Регистрация только что запущенного своего контроллера ночной серии."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import psutil

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/b123_20261027'


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--pid', type=int, required=True)
    parser.add_argument('--folder', required=True); args = parser.parse_args()
    folder = (ROOT / args.folder).resolve(); folder.relative_to(RUN.resolve())
    process = psutil.Process(args.pid)
    cmd = ' '.join(process.cmdline())
    assert 'b1_bounded_worker.py' in cmd or 'b123_' in cmd, cmd
    path = RUN / 'process_registry.json'
    raw = path.read_bytes(); data = json.loads(raw)
    record = dict(pid=process.pid, created=process.create_time(), folder=folder.relative_to(ROOT).as_posix(),
                  registered=datetime.now(timezone.utc).isoformat())
    assert not any(x['pid'] == record['pid'] and x['created'] == record['created'] for x in data['processes'])
    data['processes'].append(record)
    assert path.read_bytes() == raw
    temp = path.with_suffix('.next'); temp.write_bytes((json.dumps(data, indent=2)+'\n').encode()); temp.replace(path)
    print(json.dumps(record))


if __name__ == '__main__': main()
