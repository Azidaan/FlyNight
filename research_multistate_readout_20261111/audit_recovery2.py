"""Correct evaluator-only float32 neighbor averaging; keep strict tolerance."""
from common import *
import numpy as np
from scripts.z13_target_model import neighbor_aug
from tracking.z13_readout import HexGeometry

verify()
source=OUT/'audit.py'
assert sha(source)==read(OUT/'protocol.json')['sources'][source.relative_to(ROOT).as_posix()]
text=source.read_text(encoding='utf8')
old="def shape(c,r):return dict(C34=c.astype(float),R134_temporal=r.astype(float),R134_spatial=neighbor_aug(r[:,:17].T[None],nb)[0].T.astype(float))"
new="def shape(c,r):\n            r=r.astype(float)\n            return dict(C34=c.astype(float),R134_temporal=r,R134_spatial=neighbor_aug(r[:,:17].T[None],nb)[0].T)"
assert text.count(old)==1
text=text.replace(old,new)
# Add a precise diagnosis before the unchanged strict independent test.
needle="assert maxerr<1e-10,(maxerr,'independent interpolation/correlation disagreement')"
replacement="""write('audit_recovery2_diagnosis.json',dict(original_audit_max_error=2.472616353332846e-08,
    corrected_audit_max_error=maxerr,unchanged_tolerance=1e-10,
    cause='Independent ordinary spatial branch averaged float32 endpoint maps; comparator casts all maps to float64 BEFORE neighborhood averaging. Midpoint branch was already float64.',
    original_audit_sha256=sha(OUT/'audit.py'),recovery_source_sha256=sha(OUT/'audit_recovery2.py'),
    predictions_unchanged_sha256=sha(OUT/'predictions.json'),maps_complete_unchanged_sha256=sha(OUT/'maps_complete.json')))
"""+needle
assert text.count(needle)==1
exec(compile(text.replace(needle,replacement),str(source),'exec'),dict(__name__='__main__',__file__=str(source)))
