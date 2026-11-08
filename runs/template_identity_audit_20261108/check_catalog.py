"""Run the required catalog validation once under the system reserve guard."""
import contextlib,runpy,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'runs/flyvis_repairs_20261106/unification_20261107'))
from system_memory_guard import guard_entrypoint
guard_entrypoint(timeout=600)
with (OUT/'catalog_validation.log').open('x',encoding='utf8') as stream:
    with contextlib.redirect_stdout(stream):
        try:runpy.run_path(str(ROOT/'scripts/validate_project_history.py'),run_name='__main__')
        except SystemExit as exc:
            if exc.code:raise
print('Catalog validation completed')
