import ast, hashlib, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

source = (OUT / 'worker.py').read_text(encoding='utf8')
source = source.replace("        anchor = clocks[str(frame)]", "        if str(frame) not in clocks:\n            # No exact evaluation annotation: retain unknown, never guess a clock.\n            return []\n        anchor = clocks[str(frame)]")
source = source.replace("protocol, binding = verify()", "protocol, binding = verify()\naddendum = read(OUT / 'worker_addendum_v2.json')\nfor name, digest in addendum['new_sources'].items():\n    assert sha(OUT / name) == digest, name", 1)
source = '\n'.join(line.rstrip() for line in source.splitlines()).rstrip() + '\n'
ast.parse(source)
with (OUT / 'worker_v2.py').open('xb') as stream:
    stream.write(source.encode('utf8'))
addendum = dict(original_protocol_sha256=sha(OUT / 'proposal_protocol.json'), reason='Pre-execution metadata audit found arbitrary competing-reference rawframes lack exact GT clock anchors. Pure decision policy needs no labels. Evaluation now keeps unavailable reference annotations unknown instead of crashing or guessing a nearby GT frame; primary noforeign/nounknown engineering criterion remains unchanged.', repair_outcomes_computed=False, original_sources_preserved=True,
                new_sources={'worker_v2.py': sha(OUT / 'worker_v2.py')}, policy_changed=False, gate_changed=False)
with (OUT / 'worker_addendum_v2.json').open('x', encoding='utf8', newline='\n') as stream:
    stream.write(json.dumps(addendum, ensure_ascii=False, indent=2).rstrip() + '\n')
print(sha(OUT / 'worker_addendum_v2.json'))
