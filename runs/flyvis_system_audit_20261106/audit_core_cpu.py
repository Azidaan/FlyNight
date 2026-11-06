"""Audit harness revision: explicit CPU aliases, corrected MotionField attribute.

Original failed harness and its log are retained. No project implementation edits.
"""
from pathlib import Path
p=Path(__file__).with_name('audit_core.py')
source=p.read_text(encoding='utf-8')
source=source.replace("fv = activate_flyvis()", "fv = activate_flyvis()\nimport importlib\nfor module in (fv, importlib.import_module('flyvis.network.initialization'), importlib.import_module('flyvis.task.decoder')):\n    module.device = torch.device('cpu')")
source=source.replace('.vectors.copy()', '.vectors_xy.copy()')
exec(compile(source,str(p),'exec'))
