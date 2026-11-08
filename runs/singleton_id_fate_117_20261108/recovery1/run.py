"""Preserve two inconsistent pre-event identity histories as unknown."""
import sys,hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent;SOURCE=HERE.parent/'analyze.py'
f=json.loads((HERE.parent/'frozen.json').read_text(encoding='utf-8'))
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==f['files'][SOURCE.relative_to(SOURCE.parents[2]).as_posix()]
code=SOURCE.read_text(encoding='utf-8')
old="assert (target is not None)==bool(anchors)"
assert code.count(old)==1
code=code.replace(old,"assert (target is not None)==(bool(anchors) and len({i for fr,i in anchors})==1)")
code=code.replace("anchor=max((fr for fr,i in anchors),default=None)","anchor=max((fr for fr,i in anchors),default=None) if target is not None else None")
for filename in ('events.json','summary.json'):
    code=code.replace("P/'"+filename+"'","P/'recovery1'/ '"+filename+"'")
exec(compile(code,str(SOURCE),'exec'),{'__file__':str(SOURCE),'__name__':'__main__'})
