import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'runs/flyvis_repairs_20261106/unification_20261107'))
from system_memory_guard import guard_entrypoint
guard_entrypoint(timeout=600)
import pytest
assert pytest.main(['tests/test_hex_background_filter.py','tests/test_project_history.py','-q'])==0
from scripts.build_obsidian_vault import main
from scripts.audit_obsidian_levels import audit

def build(vault):
    sys.argv=['scripts/build_obsidian_vault.py','--vault',str(vault)]
    assert main()==0

vault=ROOT/'ObsidianFlyVis'
hypotheses={p.name:p.read_bytes() for p in (vault/'Карта проекта/Гипотезы').glob('*.md')}
with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as temporary:
    preview=Path(temporary)/'ObsidianFlyVis'
    shutil.copytree(vault,preview)
    build(preview)
    result=audit(preview)
    build(vault)
    assert audit(vault)==result
assert hypotheses=={p.name:p.read_bytes() for p in (vault/'Карта проекта/Гипотезы').glob('*.md')}
result['hypotheses_unchanged']=True
(OUT/'graph_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
