"""Materialize a frozen alpha-parameterized independent reviewer before registration."""
import ast
from pathlib import Path
root=Path(__file__).resolve().parents[1];source=(root/'research_bbox_singleton_pilot_20261108/audit.py').read_text(encoding='utf-8');tree=ast.parse(source);node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='replay');body=ast.get_source_segment(source,node)
assert body.count("corrected_cost=e['cost']-.2")==1
body=body.replace('def replay(native,data,cfg,ledger):','def replay(native,data,cfg,ledger,alpha):').replace("corrected_cost=e['cost']-.2","corrected_cost=e['cost']-alpha")
path=Path(__file__).parent/'audit_replay.py'
with path.open('x',encoding='utf-8') as f:f.write('"""Independent canonical-ID and SciPy LAP reconstruction; alpha is explicit."""\nfrom collections import Counter\nfrom research_bbox_singleton_pilot_20261108.audit import independently_proposed\n\n\n'+body+'\n')
