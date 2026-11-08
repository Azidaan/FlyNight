"""Use a non-stdlib module name; inspect.py shadows Python's inspect module."""
from pathlib import Path
import hashlib,json
here=Path(__file__).resolve().parent
source=here/'initial_inspect_source.txt'
expected=json.loads((here/'before.json').read_text(encoding='utf-8-sig'))['Hash'].lower()
assert hashlib.sha256(source.read_bytes()).hexdigest()==expected
exec(compile(source.read_text(encoding='utf-8'),str(source),'exec'),{'__file__':str(Path(__file__).resolve()),'__name__':'__main__'})
