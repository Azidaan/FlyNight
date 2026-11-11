"""Evaluator corrections only: float64 preprocessing and exact rate formula."""
from pathlib import Path
import importlib.util
import sys
folder=Path(__file__).resolve().parent
sys.path=[p for p in sys.path if Path(p or '.').resolve()!=folder]
sys.path.insert(0,str(folder.parent))
spec=importlib.util.spec_from_file_location('common',folder/'common.py')
common=importlib.util.module_from_spec(spec);sys.modules['common']=common;spec.loader.exec_module(common)
source=folder/'audit.py';p=common.read(folder/'protocol.json')
assert common.sha(source)==p['sources'][source.relative_to(folder.parent).as_posix()]
text=source.read_text(encoding='utf8')
old="def shape(c,r):return dict(C34=c.astype(float),R134_temporal=r.astype(float),R134_spatial=neighbor_aug(r[:,:17].T[None],nb)[0].T.astype(float))"
new="def shape(c,r):\n            r=r.astype(float)\n            return dict(C34=c.astype(float),R134_temporal=r,R134_spatial=neighbor_aug(r[:,:17].T[None],nb)[0].T)"
assert text.count(old)==1;text=text.replace(old,new)
old="ba=.5*(own/y.sum()+1-foreign/(~y).sum())"
new="ba=.5*(own/y.sum()+((~y).sum()-foreign)/(~y).sum())"
assert text.count(old)==1;text=text.replace(old,new)
needle="assert maxerr<1e-10,(maxerr,'independent interpolation/correlation disagreement')"
replacement="""write('audit_recovery3_diagnosis.json',dict(original_correlation_error=2.472616353332846e-08,
    corrected_correlation_error=maxerr,unchanged_correlation_tolerance=1e-10,
    interpolation_fix='Cast ordinary endpoint maps to float64 before neighborhood averaging, matching prescribed comparator preprocessing.',
    rate_fix='Compute specificity directly from exact true-negative count, as in metric definition. Old algebraically rearranged expression differed by one float64 rounding unit; integer counts already matched exactly.',
    original_C34_rate_difference=1.1102230246251565e-16,
    original_audit_sha256=sha(OUT/'audit.py'),wrapper_sha256=sha(OUT/'launch_audit_recovery3.py'),
    predictions_sha256=sha(OUT/'predictions.json'),maps_complete_sha256=sha(OUT/'maps_complete.json')))
"""+needle
assert text.count(needle)==1
exec(compile(text.replace(needle,replacement),str(source),'exec'),dict(__name__='__main__',__file__=str(source)))
