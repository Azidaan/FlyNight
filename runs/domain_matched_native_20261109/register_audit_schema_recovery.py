"""Preserve the failed audit and verify its exact schema-only correction."""
from evaluation_common import *
import psutil,ast
failed=ROOT/'runs/flyvis_repairs_20261106/unification_20261107/uncapped_checks/5d3f1c5c4b834e5cb5c663c7b87983f5';report=read(failed/'memory_guard.json')
assert report['returncode']==1 and report['reason'] is None and not (OUT/'runtime_audit.json').exists()
for pid,birth in report['owned'].items():
    try:assert psutil.Process(int(pid)).create_time()!=birth
    except psutil.NoSuchProcess:pass
value=read(OUT/'deployment.json')['independent_gradient_max'];assert isinstance(value,dict) and set(value)=={'G7','C34','R134_spatial','R134_temporal'} and all(v<1e-7 for v in value.values())
verify(read(OUT/'audit_protocol.json'))
write('audit_schema_recovery_protocol.json',dict(registered=datetime.now(MSK).isoformat(),
    failure='The new deployment stores the independently computed final gradient residual per arm as a dictionary, while the audit treated it as a scalar. Failure occurred before any event matrix or classifier check. No scientific output invalidated.',
    change='AST replaces only the scalar comparison with all per-arm residuals<1e-7. Original audit, sources, thresholds and scientific results preserved.',
    sources={p.relative_to(ROOT).as_posix():sha(p) for p in [Path(__file__),OUT/'run_runtime_audit_v2.py',OUT/'audit_runtime.py',OUT/'audit_protocol.json',OUT/'deployment.json',failed/'memory_guard.json',failed/'memory_guard.worker.log']},
    independent_gradients=value,no_fit=True,no_tracker_or_core_repeated=True))
print('AUDIT_SCHEMA_RECOVERY_REGISTERED',value,flush=True)
