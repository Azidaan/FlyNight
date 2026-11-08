"""Check the provenance of reused inventory facts and maps, without recomputing labels."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
p=read(OUT/'protocol.json')
inv=read(ROOT/'runs/singleton_car_map_coverage_20261108/inventory.json')
old={Path(k).as_posix():v for k,v in inv['sources'].items()}
checked={}
for path,expected in p['sources'].items():
    if path in old:
        assert expected==old[path],path
        assert sha(ROOT/path)==old[path],path
        checked[path]=old[path]
maps={}
for case in p['cases']:
    for key,path in case['ready_maps'].items():
        manifest=ROOT/path
        receipt=manifest.parent.parent/'features_complete.json'
        assert receipt.relative_to(ROOT).as_posix() in old
        assert sha(receipt)==old[receipt.relative_to(ROOT).as_posix()]
        expected=read(receipt)['files'][key]
        assert sha(ROOT/path)==expected
        maps[path]=expected
assert len(maps)==6
with (OUT/'inventory_provenance.json').open('x',encoding='utf8') as f:
    json.dump(dict(passed=True,verified_inventory_sources=checked,reused_map_hashes=maps),f,ensure_ascii=False,indent=2)
print('Verified',len(checked),'inventory sources and',len(maps),'ready maps')
