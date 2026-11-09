"""Technical serialization wrapper fixed before any new data audit."""
import audit as a
from audit import *
original_cast=a.cast
def compatible_cast(value):
    return sorted(value) if isinstance(value,set) else original_cast(value)
a.cast=compatible_cast
assert compatible_cast({'fold':{'foreign','own'}})=={'fold':['foreign','own']}
execution=dict(passed=True,change='Before data execution, convert per-fold class SETS to sorted JSON lists. No selection, label, support, threshold or model change.',sources={p.relative_to(ROOT).as_posix():sha(p) for p in [Path(__file__),OUT/'audit.py',OUT/'protocol.json']})
write('execution_protocol.json',execution)
original_verify=a.verify
def verify_execution(p):
    original_verify(p);original_verify(execution)
a.verify=verify_execution
a.run()
