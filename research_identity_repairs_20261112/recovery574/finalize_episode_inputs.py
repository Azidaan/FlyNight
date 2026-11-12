import ast, hashlib, json
from pathlib import Path
OUT = Path(__file__).resolve().parent
for name in ['worker_v4.py', 'audit_v4.py']:
    path = OUT / name
    source = path.read_text(encoding='utf8')
    marker = "for name, digest in episode_addendum['new_sources'].items():\n    assert sha(OUT / name) == digest, name"
    assert source.count(marker) == 1
    source = source.replace(marker, marker + "\nfor name, digest in episode_addendum['additional_inputs'].items():\n    assert sha(ROOT / name) == digest, name", 1)
    ast.parse(source)
    path.write_bytes((source.rstrip() + '\n').encode('utf8'))
protocol_path = OUT / 'episode_addendum_v4.json'
protocol = json.loads(protocol_path.read_text(encoding='utf8'))
protocol['new_sources'] = {name: hashlib.sha256((OUT / name).read_bytes()).hexdigest() for name in list(protocol['new_sources']) + ['finalize_episode_inputs.py']}
protocol_path.write_bytes((json.dumps(protocol, ensure_ascii=False, indent=2).rstrip() + '\n').encode('utf8'))
print(hashlib.sha256(protocol_path.read_bytes()).hexdigest())
