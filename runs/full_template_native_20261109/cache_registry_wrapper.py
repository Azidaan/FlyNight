"""Keep status recovery; add only validated cache-receipt registry coverage."""
import sys,runpy
import common
target=sys.argv[1];arguments=sys.argv[2:]
marker=arguments.pop() if arguments and arguments[-1].startswith('--audit-memory-worker=') else None
allowed={'native_run.py','register_evaluation.py','deployment_audit.py','evaluate_native.py','native_statistics.py','id_fate.py',
    'register_boundary.py','boundary_audit.py'}
assert target in allowed
mode=arguments[0] if target=='native_run.py' else None
common.start('gpu' if mode=='C34' else 'cpu',timeout=3600 if mode=='C34' else 7200)
assert marker is not None
phase=common.read(common.OUT/'cache_registry_protocol.json');common.verify(phase)
from status_write_recovery import stable_status
common.status=stable_status
if target=='native_run.py':
    from cache_registry_recovery import install
    install()
sys.argv=[str(common.OUT/target),*arguments,marker]
runpy.run_path(str(common.OUT/target),run_name='__main__')
