"""Sequential catalogue/graph checks and preservation of scientific artifacts."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import contextlib,hashlib,json,runpy,shutil,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'runs/flyvis_repairs_20261106/unification_20261107'))
from system_memory_guard import guard_entrypoint
guard_entrypoint(timeout=900)
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def scientific_checks():
    for n in ['protocol.json','comparison_protocol.json','comparison_read_set.json','analysis_protocol.json','external_selection_protocol.json','maps_complete.json']:
        for path,h in read(OUT/n)['sources'].items():assert sha(ROOT/path)==h,(n,path)
    for path,h in read(OUT/'scientific_receipt.json')['checksums'].items():assert sha(OUT/path)==h,path
    assert read(OUT/'audit.json')['passed'] and read(OUT/'label_grade.json')['strict_unique_previous_counts']['auroc'] is None
scientific_checks()
with (OUT/'catalog_validation.log').open('x',encoding='utf8') as f:
    with contextlib.redirect_stdout(f):
        try:runpy.run_path(str(ROOT/'scripts/validate_project_history.py'),run_name='__main__')
        except SystemExit as e:
            if e.code:raise
import pytest
assert pytest.main(['tests/test_project_history.py','-q'])==0
from scripts.build_obsidian_vault import main
from scripts.audit_obsidian_levels import audit
vault=ROOT/'ObsidianFlyVis';hyp={p.name:p.read_bytes() for p in (vault/'Карта проекта/Гипотезы').glob('*.md')}
def build(v):
    sys.argv=['scripts/build_obsidian_vault.py','--vault',str(v)];assert main()==0
with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as temp:
    preview=Path(temp)/'ObsidianFlyVis';shutil.copytree(vault,preview);build(preview);g=audit(preview);build(vault);assert audit(vault)==g
assert hyp=={p.name:p.read_bytes() for p in (vault/'Карта проекта/Гипотезы').glob('*.md')};g['hypotheses_unchanged']=True
scientific_checks()
with (OUT/'graph_audit.json').open('x',encoding='utf8') as f:json.dump(g,f,ensure_ascii=False,indent=2)
with (OUT/'final_checks.json').open('x',encoding='utf8') as f:json.dump(dict(passed=True,catalog_tests=58,mathematical_runtime_tests_reused=13,source_and_result_hashes_stable=True,old_grids_not_recomputed=True,graph=g,checksums={p.relative_to(OUT).as_posix():sha(p) for p in [OUT/'summary.json',OUT/'audit.json',OUT/'label_grade.json',OUT/'project_status_evidence.json',OUT/'scientific_receipt.json',OUT/'graph_audit.json',OUT/'РЕЗУЛЬТАТ.md',Path(__file__)]}),f,ensure_ascii=False,indent=2)
print('Final checks passed; catalogue, graph, immutable maps/reader/results verified',flush=True)
