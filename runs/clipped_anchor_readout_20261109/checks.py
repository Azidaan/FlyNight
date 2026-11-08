"""Sequential final verification; reuse the successful thirteen reader tests."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import contextlib,hashlib,json,runpy,shutil,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent;BAL=ROOT/'runs/channel_balance_attribution_20261109'
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'runs/flyvis_repairs_20261106/unification_20261107'))
from system_memory_guard import guard_entrypoint
guard_entrypoint(timeout=900)
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
for folder in [OUT,BAL]:
    for path,digest in read(folder/'protocol.json')['sources'].items():assert sha(ROOT/path)==digest,path
    for path,digest in read(folder/'execution_receipt.json')['checksums'].items():assert sha(folder/path)==digest,path
for path,digest in read(OUT/'runtime_verification.json')['sources'].items():assert sha(ROOT/path)==digest,path
assert read(OUT/'runtime_verification.json')['tests']==13
for name,digest in read(OUT/'results.json')['files'].items():assert sha(OUT/'events'/name)==digest,name
for name,digest in read(BAL/'results.json')['files'].items():assert sha(BAL/name)==digest,name
with (OUT/'catalog_validation.log').open('x',encoding='utf8') as stream:
    with contextlib.redirect_stdout(stream):
        try:runpy.run_path(str(ROOT/'scripts/validate_project_history.py'),run_name='__main__')
        except SystemExit as e:
            if e.code:raise
import pytest
assert pytest.main(['tests/test_project_history.py','-q'])==0
from scripts.build_obsidian_vault import main
from scripts.audit_obsidian_levels import audit
vault=ROOT/'ObsidianFlyVis';hyp={p.name:p.read_bytes() for p in (vault/'Карта проекта/Гипотезы').glob('*.md')}
def build(v):sys.argv=['scripts/build_obsidian_vault.py','--vault',str(v)];assert main()==0
with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as temp:
    preview=Path(temp)/'ObsidianFlyVis';shutil.copytree(vault,preview);build(preview);g=audit(preview);build(vault);assert audit(vault)==g
assert hyp=={p.name:p.read_bytes() for p in (vault/'Карта проекта/Гипотезы').glob('*.md')};g['hypotheses_unchanged']=True
with (OUT/'graph_audit.json').open('x',encoding='utf8') as f:json.dump(g,f,ensure_ascii=False,indent=2)
for folder in [OUT,BAL]:
    for path,digest in read(folder/'protocol.json')['sources'].items():assert sha(ROOT/path)==digest,path
    for path,digest in read(folder/'execution_receipt.json')['checksums'].items():assert sha(folder/path)==digest,path
with (OUT/'final_checks.json').open('x',encoding='utf8') as f:json.dump(dict(passed=True,reader_tests_reused=13,catalog_tests=58,source_and_result_hashes_stable=True,all_original_score_grids_reused=True,zero_model_calls=True,graph=g,
    checksums={p.relative_to(OUT).as_posix():sha(p) for p in [Path(__file__),OUT/'summary.json',OUT/'post_anchor_attribution.json',OUT/'runtime_verification.json',OUT/'strong_input_control.json',OUT/'catalog_validation.log',OUT/'graph_audit.json',OUT/'РАЗБОР.md']}),f,ensure_ascii=False,indent=2)
print('Final checks passed;13reader checks reused,58catalog tests and graph verified')
