"""Complete immutable read-set fingerprint before executing frozen comparison."""
import hashlib,json,runpy,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
cp=read(OUT/'comparison_protocol.json');paths=[Path(__file__),OUT/'comparison_protocol.json',OUT/'external_inventory.py']
for path in cp['historical_clipped_sources']:
    name=Path(path).name
    for folder in ['background_patch_gate_20261108','decoder34_identity_gate_20261108','channel_balance_attribution_20261109']:
        p=ROOT/f'runs/{folder}/events/{name}';paths.append(p)
        m=read(ROOT/f'runs/{folder}/results.json')['files']
        expected=m.get(name,m.get('events/'+name,m.get(p.relative_to(ROOT).as_posix())));assert expected==sha(p),(p,expected)
paths += [ROOT/p for p in read(OUT/'maps_complete.json')['sources']]
protocol=dict(sources={p.relative_to(ROOT).as_posix():sha(p) for p in paths},purpose='Capture complete remaining ready-map/grid read set before any new score, preserving all frozen source files.')
fingerprint=OUT/'comparison_read_set.json'
if fingerprint.exists():
    old=read(fingerprint);assert old==protocol
else:
    with fingerprint.open('x',encoding='utf8') as f:json.dump(protocol,f,ensure_ascii=False,indent=2)
sys.argv=[str(OUT/'compare.py'),'run'];runpy.run_path(str(OUT/'compare.py'),run_name='__main__')
