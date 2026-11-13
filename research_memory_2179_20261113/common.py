import os, sys, json, hashlib
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
PREV = ROOT / 'research_assignment_weights_2179_20261112'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['OMP_NUM_THREADS'] = os.environ['MKL_NUM_THREADS'] = '2'
for key, rel in [('TEMP', '.cache/tmp'), ('TMP', '.cache/tmp'), ('TMPDIR', '.cache/tmp'), ('TORCH_HOME', '.cache/torch'), ('CUDA_CACHE_PATH', '.cache/cuda')]:
    p = ROOT / rel
    p.mkdir(parents=True, exist_ok=True)
    os.environ[key] = str(p)
sys.path.insert(0, str(ROOT))
def read(p): return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''): h.update(b)
    return h.hexdigest()
def now(): return datetime.now(ZoneInfo('Europe/Moscow')).isoformat()
def save(n, v):
    p = OUT / n
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('xb') as f: f.write((json.dumps(v, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode())
def journal(message):
    p = ROOT / 'dissertation/ЕДИНЫЙ_ПЛАН.md'
    raw = p.read_bytes(); text = raw.decode('utf8'); nl = '\r\n' if '\r\n' in text else '\n'
    mark = '## Журнал изменений' + nl
    addition = '\n**' + datetime.now(ZoneInfo('Europe/Moscow')).strftime('%d.%m.%Y, %H:%M MSK') + '** — ' + message + '\n\n'
    assert text.count(mark) == 1 and p.read_bytes() == raw
    p.write_bytes(text.replace(mark, mark + addition.replace('\n', nl), 1).encode())
    with (OUT / 'journal_additions.txt').open('ab') as f: f.write(addition.encode())
