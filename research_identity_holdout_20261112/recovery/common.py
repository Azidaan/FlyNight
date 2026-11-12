"""I/O and provenance only; no policy, annotation, or numeric semantics."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = os.environ['MKL_NUM_THREADS'] = '2'
import json, hashlib
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent

def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def now():
    return datetime.now(ZoneInfo('Europe/Moscow')).isoformat()

def write(name, value):
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf8', newline='\n') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).rstrip() + '\n')

def verify():
    protocol = read(OUT / 'protocol.json')
    binding = read(OUT / 'descriptor_binding.json')
    assert binding['descriptor_file'] in binding['inputs']
    for source in [protocol['inputs'], binding['inputs']]:
        for path, digest in source.items():
            assert sha(ROOT / path) == digest, path
    for path, digest in protocol['sources'].items():
        assert sha(OUT / path) == digest, path
    return protocol, binding

def status(stage, **kwargs):
    value = dict(updated=now(), stage=stage, **kwargs)
    (OUT / 'status.json').write_bytes((json.dumps(value, ensure_ascii=False, indent=2).rstrip() + '\n').encode('utf8'))
