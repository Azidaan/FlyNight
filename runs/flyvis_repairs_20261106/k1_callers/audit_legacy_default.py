"""Final secondary callers audit after retracting the resize default."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
before=json.loads((OUT/'before_manifest.json').read_text(encoding='utf-8-sig'))
callers=[]
for entry in before:
    path=ROOT/entry['path'];snapshot=OUT/'original'/path.name
    assert sha(snapshot)==entry['sha256']
    tree=ast.parse(path.read_text(encoding='utf-8'))
    constructors=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Call) and ast.unparse(node.func) in ('K.Fly','make_fly'):
            assert any(k.arg=='geometry_version' for k in node.keywords)
            constructors.append(dict(line=node.lineno,call=ast.unparse(node.func),explicit_geometry=True))
    assert constructors
    callers.append(dict(path=entry['path'],sha256=sha(path),original_sha256=entry['sha256'],constructors=constructors))
for row in json.loads((OUT/'pre_pixel_centers/manifest.json').read_text(encoding='utf-8')):
    assert sha(OUT/'pre_pixel_centers'/Path(row['path']).name)==row['sha256']
import torch
from tracking.flyvis_geometry_protocol import require_arm_geometry
from tracking.flyvis_motion import GEOMETRY_VERSION,GEOMETRY_VERSIONS
assert GEOMETRY_VERSION=='legacy_v1'
assert set(GEOMETRY_VERSIONS)=={'legacy_v1','pixel_centers_v2','resize_v2'}
legacy=ROOT/'runs/fly_k1_v2/arms.pt';original_hash=sha(legacy)
arms=torch.load(legacy,map_location='cpu',weights_only=False)
assert all('geometry_version' not in value for value in arms.values())
rejections={}
for version in ['pixel_centers_v2','resize_v2']:
    try:require_arm_geometry(arms,version)
    except ValueError as e:rejections[version]=str(e)
    else:raise AssertionError('Historical weights silently reinterpreted')
require_arm_geometry(arms,'legacy_v1')
cli=[]
for label,extra in [('explicit_pixel_centers',['--geometry-version','pixel_centers_v2']),('research_with_fixed_gain',['--geometry-version','resize_v2'])]:
    result=subprocess.run([sys.executable,str(ROOT/'scripts/fly_k1_v2.py'),'--eval-only',*extra],
        cwd=ROOT,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=30)
    (OUT/f'actual_{label}_rejection.log').write_text(result.stdout+result.stderr,encoding='utf-8')
    assert result.returncode!=0
    expected='saved geometry legacy_v1' if label=='explicit_pixel_centers' else 'historical fixed gain'
    assert expected in result.stderr
    cli.append(dict(case=label,exit_code=result.returncode,evidence=f'actual_{label}_rejection.log',expected_failure=expected))
assert sha(legacy)==original_hash and not torch.cuda.is_initialized()
report=dict(passed=True,default_geometry=GEOMETRY_VERSION,versions=list(GEOMETRY_VERSIONS),callers=callers,
    new_files={name:sha(ROOT/name) for name in ['tracking/flyvis_geometry_protocol.py','tests/test_flyvis_secondary_geometry.py']},
    actual_legacy_weights=dict(path=legacy.relative_to(ROOT).as_posix(),sha256=original_hash,rejections=rejections,explicit_legacy_allowed=True),
    actual_cli=cli,
    resize_unit_gain_entrypoints=['fly_axes_probe','fly_dynamic_norm_probe','fly_reference_probe','fly_scale_probe','fly_stage_probe','build_fly_lcn_motion','fly_noise_probe','fly_speed_response'],
    resize_rejected_historical_gain_entrypoints=['fly_finetune_k1_synthetic','fly_k1_v2','fly_wide_field'],
    test_evidence='tests_legacy_default_final.log: 62 passed in 5.88s',
    correction='Earlier REPORT.txt described resize_v2 as default. Current default is legacy_v1; see FINAL_LEGACY_DEFAULT.txt and prior additive corrections.',
    scientific_runs_performed=False,cuda_initialized=False,script_sha256=sha(__file__))
(OUT/'manifest_legacy_default.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k!='callers'},ensure_ascii=False,indent=2))
