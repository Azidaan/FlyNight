"""Correct the last-axis mask for a batch of 19 shifts; no change to geometry."""
from pathlib import Path
import hashlib,json
here=Path(__file__).resolve().parent
source=here/'initial_inspect_source.txt'
expected=json.loads((here/'before.json').read_text(encoding='utf-8-sig'))['Hash'].lower()
assert hashlib.sha256(source.read_bytes()).hexdigest()==expected
code=source.read_text(encoding='utf-8')
old='def in_box(q,b):return ((q>=b[:2])&(q<=b[2:])).all(1)'
assert code.count(old)==1
code=code.replace(old,old.replace('.all(1)','.all(-1)'))
exec(compile(code,str(source),'exec'),{'__file__':str(Path(__file__).resolve()),'__name__':'__main__'})
