"""Append one pinned engineering record and build a private publication overlay."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import re
import subprocess
import os
from contextlib import ExitStack

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'runs/flyvis_repairs_20261106'
OUT = RUN / 'boundary_ledger_overlay_v1'
PARENT = '2c3cb2644f66eedfa70c7820ae29070bf2738794'
PLAN = 'dissertation/ЕДИНЫЙ_ПЛАН.md'
CAT = 'experiments/project_history.json'
GLOSS = 'experiments/metrics_glossary.json'

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def git(*args):
    return subprocess.check_output(['git', '--no-optional-locks', *args], cwd=ROOT)

def append_card(raw, card):
    text = raw.decode('utf-8')
    data = json.loads(text)
    assert card['id'] not in {c['id'] for c in data['experiments']}
    start = re.search(r'"experiments"\s*:\s*\[', text).end() - 1
    _, end = json.JSONDecoder().raw_decode(text, start)
    offset = len(text[:end - 1].rstrip())
    addition = ',\n' + '\n'.join('    ' + line for line in json.dumps(card, ensure_ascii=False, indent=2).splitlines())
    result = (text[:offset] + addition + text[offset:]).encode('utf-8')
    assert json.loads(result) == dict(data, experiments=[*data['experiments'], card])
    assert (result.decode()[:offset] + result.decode()[offset+len(addition):]).encode() == raw
    return result

def append_journal(raw, entry):
    marker = '## Журнал изменений'.encode('utf-8')
    assert raw.count(marker) == 1 and entry.startswith(b'\n\n**') and entry.endswith(b'\n')
    assert entry not in raw
    result = raw.replace(marker, marker + entry, 1)
    assert result.replace(marker + entry, marker, 1) == raw
    return result

def open_exclusive(path):
    """Hold a Windows no-sharing handle during the short compare-and-write phase."""
    import ctypes
    from ctypes import wintypes
    import msvcrt
    assert os.name == 'nt'
    api = ctypes.WinDLL('kernel32', use_last_error=True)
    api.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                               wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    api.CreateFileW.restype = wintypes.HANDLE
    api.CloseHandle.argtypes = [wintypes.HANDLE]
    api.CloseHandle.restype = wintypes.BOOL
    handle = api.CreateFileW(str(path.resolve()), 0x80000000 | 0x40000000, 0, None, 3, 0x80, None)
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        fd = msvcrt.open_osfhandle(handle, os.O_RDWR | os.O_BINARY)
    except BaseException:
        api.CloseHandle(handle)
        raise
    return os.fdopen(fd, 'r+b')

def main():
    assert not OUT.exists(), 'Use a new immutable overlay directory'
    card_raw = (RUN / 'boundary_card.json').read_bytes()
    card = json.loads(card_raw)
    entry = (RUN / 'boundary_journal.txt').read_bytes()
    assert card['id'] == 'flyvis_system_boundary_addendum_20261107' and card['metrics'] == []
    assert card['state'] == 'infrastructure'
    for item in card['evidence']:
        assert (ROOT / item['ref']).is_file(), item['ref']
    before = {name: (ROOT / name).read_bytes() for name in (PLAN, CAT)}
    base = {name: git('cat-file', 'blob', PARENT + ':' + name) for name in (PLAN, CAT, GLOSS)}
    after = {PLAN: append_journal(base[PLAN], entry), CAT: append_card(base[CAT], card)}
    work_after = {PLAN: append_journal(before[PLAN], entry), CAT: append_card(before[CAT], card)}
    OUT.mkdir()
    rows = []
    for name in (PLAN, CAT):
        target = OUT / 'after' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(after[name])
        original = OUT / 'baseline' / name
        original.parent.mkdir(parents=True, exist_ok=True)
        original.write_bytes(base[name])
        rows.append(dict(path=name, baseline_sha256=sha(base[name]), after_sha256=sha(after[name]),
                         staged_candidate=target.relative_to(ROOT).as_posix()))
    original = OUT / 'baseline' / GLOSS
    original.parent.mkdir(parents=True, exist_ok=True)
    original.write_bytes(base[GLOSS])
    manifest = dict(base_commit=PARENT, files=rows, selected_card_ids=[card['id']],
                    prior_cards_unchanged=True, prior_plan_bytes_preserved=True,
                    card_sha256=sha(card_raw), journal_sha256=sha(entry))
    (OUT / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    backup = RUN / 'private_ledger_backups' / 'boundary_20261107'
    backup.mkdir(parents=True, exist_ok=False)
    for name, raw in before.items():
        (backup / Path(name).name).write_bytes(raw)
    with ExitStack() as stack:
        handles = {name: stack.enter_context(open_exclusive(ROOT/name)) for name in sorted(before)}
        assert all(handles[name].read() == raw for name, raw in before.items()), 'Concurrent ledger edit'
        for name in (CAT, PLAN):
            stream = handles[name]
            stream.seek(0)
            stream.write(work_after[name])
            stream.truncate()
            stream.flush()
            os.fsync(stream.fileno())
        confirmed_after = {}
        for name, stream in handles.items():
            stream.seek(0)
            confirmed = stream.read()
            assert confirmed == work_after[name]
            confirmed_after[name] = sha(confirmed)
    receipt = dict(at=datetime.now(timezone.utc).isoformat(), old_content_preserved=True,
                   before={name: sha(raw) for name, raw in before.items()},
                   after=confirmed_after,
                   publication_parent=PARENT, shared_git_written=False,
                   exclusive_windows_handles_during_compare_and_write=True,
                   multi_file_power_failure_transaction=False)
    (RUN / 'boundary_record_receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(json.dumps(receipt))

if __name__ == '__main__':
    main()
