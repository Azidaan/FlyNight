"""Bind the diagnosed import recovery before resuming any scientific work."""
import json,hashlib,sysconfig
from pathlib import Path
from datetime import datetime,timezone,timedelta
ROOT=Path(__file__).resolve().parents[1];OUT=Path(__file__).resolve().parent
failed=ROOT/'runs/flyvis_repairs_20261106/unification_20261107/uncapped_checks/2e62bf53d9f6423485d946ca95add11d'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
sources=[OUT/n for n in ['safe_phase.py','queue_recovery.py','compute.py','audit.py','record_result.py','check_records.py','execution_protocol.json','queue.py','inputs_complete.json']]
sources += [failed/'memory_guard.json',failed/'memory_guard.worker.log']
stdlib=Path(sysconfig.get_path('stdlib'))/'queue.py'
assert not (OUT/'input_math_witness.json').exists() and not (OUT/'maps_complete.json').exists()
report=dict(registered=datetime.now(timezone(timedelta(hours=3))).isoformat(),sources={p.relative_to(ROOT).as_posix():sha(p) for p in sources},
    standard_queue_path=str(stdlib),standard_queue_sha256=sha(stdlib),
    failure='Torch dataloader imported queue; Python resolved the experiment local queue.py, which executed the orchestration script again and failed on existing logs. Direct traceback proves the module path. No core/map work had begun.',
    recovery='Keep all frozen sources, failed queue/guard/logs and ready52inputs. Safe bootstrap removes only the experiment directory from import search, verifies and preloads standard queue, then executes the unchanged bound phase with its correct __file__. Bootstrap target survives guard worker reentry via a task-specific environment variable.',
    scientific_invariants='No changed data,weights,reader,model,threshold,fold or criterion. Existing ready snapshots reused. Resume only failed restore stage, then original registered sequence.',goal_achieved=False)
with (OUT/'import_recovery_protocol.json').open('x',encoding='utf8') as f:json.dump(report,f,ensure_ascii=False,indent=2);f.write('\n')
print('IMPORT_RECOVERY_REGISTERED',flush=True)
