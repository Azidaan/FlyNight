"""Те же условия до расчёта; атомарная запись вместо отказавшего открытия каталога."""
from pathlib import Path
import hashlib

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/flyvis_y1_latency_20261104'
source = OUT / 'registration_failure1/source.py'
text = source.read_text(encoding='utf-8')
assert hashlib.sha256(source.read_bytes()).hexdigest() == hashlib.sha256((ROOT / 'scripts/register_flyvis_y1_latency_20261104.py').read_bytes()).hexdigest()
assert not (OUT / 'inference_completed.json').exists()
text = text.replace('OUT.mkdir(exist_ok=False)', 'OUT.mkdir(exist_ok=True)', 1)
text = text.replace("history_path.write_text(json.dumps(history, ensure_ascii=False, indent=2) + '\\n', encoding='utf-8')",
                    "atomic_write(history_path, json.dumps(history, ensure_ascii=False, indent=2) + '\\n')", 1)
text = text.replace("glossary_path.write_text(json.dumps(glossary, ensure_ascii=False, indent=2) + '\\n', encoding='utf-8')",
                    "atomic_write(glossary_path, json.dumps(glossary, ensure_ascii=False, indent=2) + '\\n')", 1)


def atomic_write(path, contents):
    temp = path.with_suffix('.y1_registration.next')
    temp.write_bytes(contents.encode('utf-8'))
    temp.replace(path)


namespace = dict(__name__='__main__', __file__=str(ROOT / 'scripts/register_flyvis_y1_latency_20261104.py'), atomic_write=atomic_write)
exec(compile(text, str(__file__), 'exec'), namespace)
