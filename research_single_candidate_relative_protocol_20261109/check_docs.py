"""Documentation-only graph checks, hard-ended by06:40; no scientific run."""
import sys,json,shutil,tempfile,hashlib
from pathlib import Path
from datetime import datetime,timedelta,timezone
ROOT=Path(__file__).resolve().parents[1];OUT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'runs/flyvis_repairs_20261106/unification_20261107'))
from system_memory_guard import guard_entrypoint
remaining=(datetime(2026,11,9,6,40,tzinfo=timezone(timedelta(hours=3)))-datetime.now(timezone(timedelta(hours=3)))).total_seconds();assert remaining>0;guard_entrypoint(timeout=min(150,remaining))
from scripts.build_obsidian_vault import main
from scripts.audit_obsidian_levels import audit
vault=ROOT/'ObsidianFlyVis';hyp={p.name:p.read_bytes() for p in (vault/'Карта проекта/Гипотезы').glob('*.md')};catalog=(ROOT/'experiments/project_history.json').read_bytes()
def build(v):sys.argv=['scripts/build_obsidian_vault.py','--vault',str(v)];assert main()==0
with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as tmp:
    copy=Path(tmp)/'ObsidianFlyVis';shutil.copytree(vault,copy);build(copy);result=audit(copy);build(vault);assert audit(vault)==result
assert hyp=={p.name:p.read_bytes() for p in (vault/'Карта проекта/Гипотезы').glob('*.md')};assert catalog==(ROOT/'experiments/project_history.json').read_bytes();evidence=json.loads((OUT/'evidence.json').read_text(encoding='utf8'));assert hashlib.sha256((ROOT/evidence['source']).read_bytes()).hexdigest()==evidence['source_sha256']
(OUT/'doc_checks.json').write_text(json.dumps(dict(passed=True,graph=result,hypotheses_unchanged=True,catalogue_unchanged=True,scientific_runs=0),ensure_ascii=False,indent=2)+'\n',encoding='utf8');print('Draft documentation and graph verified')
