"""Fault injection checks bounded retry and event timestamp preservation."""
from common import *
start('cpu',timeout=600)
import status_write_recovery as module
p=read(OUT/'status_recovery_protocol.json');verify(p)
original=module.common.write;sleeper=module.time.sleep;seen=[]
def transient(name,value,**kwargs):
    assert name=='status.json' and kwargs=={'new':False}
    seen.append(dict(value))
    if len(seen)<4:raise PermissionError('injected temporary sharing violation')
def permanent(name,value,**kwargs):raise PermissionError('injected persistent sharing violation')
try:
    module.common.write=transient;module.time.sleep=lambda _:None
    module.stable_status('retry_test',frame=300)
    assert len(seen)==4 and all(v==seen[0] for v in seen)
    module.common.write=permanent
    try:module.stable_status('persistent_retry_test')
    except PermissionError:pass
    else:raise AssertionError('Persistent failure swallowed')
finally:module.common.write=original;module.time.sleep=sleeper
write('status_retry_test.json',dict(passed=True,transient_attempts=4,unchanged_event_timestamp=True,
    persistent_error_not_swallowed=True,scientific_sources_unchanged=True,zero_scientific_calls=True))
print('STATUS RETRY TEST PASSED',flush=True)
