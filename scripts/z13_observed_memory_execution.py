"""Service adapter: use the project's shared GPU mutex; frozen science unchanged.

The adapter is also the controller's worker entry point, so the replacement is
applied in every child. The original registered scientific source is preserved.
"""
from pathlib import Path
import hashlib
import json

service=Path(__file__).resolve()
root=service.parents[1]
run=root/'runs/z13_observed_memory_20261021'
receipt=json.loads((run/'execution_protocol.json').read_text(encoding='utf-8'))
source=service.with_name('z13_observed_memory_20261021.py')
assert hashlib.sha256(source.read_bytes()).hexdigest()==receipt['original_sha256']
assert hashlib.sha256(service.read_bytes()).hexdigest()==receipt['service_sha256']
code=source.read_text(encoding='utf-8')
old="GpuLock('FlyVis_z13_observed_memory_single_gpu')"
new="GpuLock('flyvis_fovea_gpu')"
assert code.count(old)==1
exec(compile(code.replace(old,new),str(source),'exec'),{'__name__':'__main__','__file__':str(service)})
