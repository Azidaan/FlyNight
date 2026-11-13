import json
from pathlib import Path
d=Path(__file__).resolve().parent
p=d/'alignment_check.py';q=d/'alignment_check_v2.py'
s=p.read_text(encoding='utf8')
s=s.replace("sources=legacy['source_hashes']","sources=read(ROOT/'runs/full_template_native_20261109/execution_protocol.json')['sources']")
assert not q.exists()
q.write_text(s,encoding='utf8')
with (d/'alignment_v1_failure.json').open('x',encoding='utf8') as f:
    json.dump(dict(stage='Before alignment protocol and any decode',error='KeyError tracking/retina.py',
        cause='Legacy protocol hashes four launch scripts; imported retina source hash is in full_template map-producer execution protocol',
        recovery='v2 reads exact retina hash from map-producer execution protocol; original source preserved',
        scientific_changes=False),f,indent=2)
