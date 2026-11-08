"""Remove identical duplicate observations of the same frame before speed division."""
from pathlib import Path
import hashlib,json
here=Path(__file__).resolve().parent
source=here.parent/'compare.py'
record=json.loads((here.parent/'before.json').read_text(encoding='utf-8-sig'))
assert hashlib.sha256(source.read_bytes()).hexdigest()==next(x['Hash'].lower() for x in record if x['Path']==str(source))
code=source.read_text(encoding='utf-8')
def replace(old,new):
    global code
    assert code.count(old)==1
    code=code.replace(old,new)
replace('ROOT=OUT.parents[1]',"ROOT=OUT.parents[1]\nOUT=OUT/'recovery1'")
replace("h=h[-4:]; last=", """h=h[-4:]
    assert [a['frame'] for a in h]==sorted(a['frame'] for a in h)
    assert all(a['box']==b['box'] for a in h for b in h if a['frame']==b['frame'])
    h=list({a['frame']:a for a in h}.values())
    last=""")
replace("protocol_sha256=sha(OUT/'protocol.md')","protocol_sha256=sha(OUT.parent/'protocol.md'), recovery_sha256=sha(OUT/'launch.py')")
exec(compile(code,str(source),'exec'),{'__file__':str(source),'__name__':'__main__'})
