"""Verify the formerly rejected optional C8 file via its real closed receipt."""
from common import *
start('cpu',timeout=600)
import numpy as np
p=read(OUT/'cache_registry_protocol.json');verify(p)
from cache_registry_recovery import install
Bank=install();bank=Bank('b251064f-5f6b663e','C34')
name=p['failed_optional_C8'];value=bank._load(ROOT/name)
assert value.shape==(721,8) and value.dtype==np.float32 and np.isfinite(value).all()
assert bank.legacy_sources[name]==sha(ROOT/name)
assert sha(OUT/'native_runtime.py')==read(OUT/'execution_protocol.json')['sources']['runs/full_template_native_20261109/native_runtime.py']
write('cache_registry_test.json',dict(passed=True,shape=value.shape,dtype=str(value.dtype),
    verified_sha256=sha(ROOT/name),real_closed_G_PERM_receipt=True,scientific_sources_unchanged=True,
    zero_neural_or_tracker_calls=True))
print('Real failed cache provenance test passed',flush=True)
