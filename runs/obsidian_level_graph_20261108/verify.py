"""Preview, repeat, deploy with the normal generator and compare its output."""
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'runs/flyvis_repairs_20261106/unification_20261107'))
from system_memory_guard import guard_entrypoint
guard_entrypoint(timeout=420)
from scripts.build_obsidian_vault import main
from scripts.audit_obsidian_levels import audit

def build(vault):
    args=sys.argv
    try:
        sys.argv=['scripts/build_obsidian_vault.py','--vault',str(vault)]
        assert main()==0
    finally:sys.argv=args

def hashes(vault):
    return {p.relative_to(vault).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
            for p in vault.rglob('*') if p.is_file() and
            ('.obsidian' not in p.relative_to(vault).parts or p.name=='graph.json')}

vault=ROOT/'ObsidianFlyVis'
before=hashes(vault)
with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as tmp:
    preview=Path(tmp)/'ObsidianFlyVis'
    shutil.copytree(vault,preview)
    build(preview)
    result=audit(preview)
    snapshot=hashes(preview)
    build(preview)
    assert snapshot==hashes(preview)
    settings={'.graph-visible.json','.obsidian/graph.json'}
    assert all(snapshot.get(p)==h for p,h in before.items() if p not in settings and not p.startswith('Карта проекта/'))
    assert before==hashes(vault),'Vault changed during preview'
    build(vault)
    assert audit(vault)==result and hashes(vault)==snapshot
result.update(preview_equals_live=True,repeated_build_identical=True,
              existing_files_preserved=sum(p not in settings and not p.startswith('Карта проекта/') for p in before),original_notes_removed=0)
(HERE/'audit.json').write_bytes((json.dumps(result,ensure_ascii=False,indent=2)+'\n').encode())
