"""Независимый срок ночной серии: только зарегистрированные PID+время создания."""
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import psutil

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/b123_20261027'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write(path, data):
    temp = path.with_suffix('.next')
    temp.write_bytes((json.dumps(data, ensure_ascii=False, indent=2) + '\n').encode())
    temp.replace(path)


def matching(pid, created):
    try:
        p = psutil.Process(int(pid))
        return p if p.is_running() and abs(p.create_time() - float(created)) < .01 else None
    except psutil.NoSuchProcess:
        return None


def main():
    auth = read(RUN / 'authorization.json')
    save_at = datetime.fromisoformat(auth['save_at']).timestamp()
    stop_at = datetime.fromisoformat(auth['controller_stop']).timestamp()
    owned = {}; sent = set()
    while time.time() < stop_at:
        registry = read(RUN / 'process_registry.json')
        for entry in registry['processes']:
            pid, created = entry['pid'], entry['created']
            owned[int(pid)] = created
            p = matching(pid, created)
            if p:
                try:
                    for child in p.children(recursive=True): owned[child.pid] = child.create_time()
                except psutil.NoSuchProcess:
                    pass
            if time.time() >= save_at and entry.get('folder') not in sent:
                folder = (ROOT / entry['folder']).resolve(); folder.relative_to(RUN.resolve())
                (folder / 'stop.request').write_text('Author deadline: save and end.\n', encoding='utf-8')
                sent.add(entry['folder'])
        write(RUN / 'deadline_status.json', dict(state='watching', owned=owned, saved_folders=sorted(sent),
              updated=datetime.now(timezone.utc).isoformat(), stop_at=auth['controller_stop']))
        time.sleep(1)
    for pid, created in reversed(list(owned.items())):
        p = matching(pid, created)
        if p:
            try: p.kill()
            except psutil.NoSuchProcess: pass
    until = min(datetime.fromisoformat(auth['hard_stop']).timestamp() - 2, time.time() + 15)
    while time.time() < until and any(matching(pid, created) for pid, created in owned.items()): time.sleep(.2)
    alive = [pid for pid, created in owned.items() if matching(pid, created)]
    write(RUN / 'deadline_result.json', dict(all_owned_ended=not alive, remaining=alive, owned=owned,
          updated=datetime.now(timezone.utc).isoformat(), author_hard_stop=auth['hard_stop']))


if __name__ == '__main__':
    main()
