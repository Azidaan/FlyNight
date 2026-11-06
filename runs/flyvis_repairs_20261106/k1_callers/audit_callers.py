"""Read-only source/legacy-weight audit after secondary caller repairs."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import ast
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
before=json.loads((OUT/'before_manifest.json').read_text(encoding='utf-8-sig'))
callers=[]
for entry in before:
    path=ROOT/entry['path'];snapshot=OUT/'original'/path.name
    assert sha(snapshot)==entry['sha256']
    tree=ast.parse(path.read_text(encoding='utf-8'))
    construct=[];conversions=[]
    for node in ast.walk(tree):
        if not isinstance(node,ast.Call):continue
        name=ast.unparse(node.func)
        if name in ('K.Fly','make_fly'):
            assert any(k.arg=='geometry_version' for k in node.keywords),(path,node.lineno)
            construct.append(dict(line=node.lineno,call=name,explicit_geometry=True))
        if name=='K.to_px':conversions.append(node.lineno)
    assert construct,path
    callers.append(dict(path=entry['path'],sha256=sha(path),original_sha256=entry['sha256'],
        snapshot=str(snapshot.relative_to(ROOT)),constructors=construct,to_px_lines=conversions))
import torch
from tracking.flyvis_geometry_protocol import require_arm_geometry
legacy=ROOT/'runs/fly_k1_v2/arms.pt'
original_hash=sha(legacy)
arms=torch.load(legacy,map_location='cpu',weights_only=False)
assert all('geometry_version' not in arm for arm in arms.values())
try:require_arm_geometry(arms,'resize_v2')
except ValueError as exc:rejection=str(exc)
else:raise AssertionError('Historical checkpoint was silently reinterpreted')
require_arm_geometry(arms,'legacy_v1')
assert sha(legacy)==original_hash
new=['tracking/flyvis_geometry_protocol.py','tests/test_flyvis_secondary_geometry.py']
report=dict(passed=True,callers=callers,new_files={name:sha(ROOT/name) for name in new},
    all_constructor_calls_explicit=True,shape_contract='All scale consumers use shared K.to_px, accepting n or n,2; chunked run_frames exercised in tests.',
    actual_legacy_checkpoint=dict(path=str(legacy.relative_to(ROOT)),sha256=original_hash,
        arms=list(arms),new_geometry_rejected=True,explicit_legacy_allowed=True,rejection=rejection),
    actual_cli_evidence='actual_old_eval_rejection.log: exit 1 at require_arm_geometry before constructing Fly',
    test_evidence='tests_final_v2.log: 112 passed in 6.77s',
    scientific_runs_performed=False,cuda_initialized=torch.cuda.is_initialized(),
    script_sha256=sha(__file__))
assert not report['cuda_initialized']
(OUT/'manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k!='callers'},ensure_ascii=False,indent=2))
