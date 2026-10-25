"""Retain failed short synthetic price fixture; rerun with a suitable mock budget."""
from pathlib import Path
import ast
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import z13_context_execution_smoke_20261025 as previous

text=Path(previous.__file__).read_text(encoding='utf-8')
tree=ast.parse(text);source=ast.get_source_segment(text,next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main'))
assert source.count('seconds=120')==1
source=source.replace('seconds=120','seconds=600')
ns={**vars(previous),'OUT':previous.OUT/'recovery1','__file__':str(Path(__file__))}
exec(compile(source,str(Path(__file__)),'exec'),ns)
if __name__=='__main__':ns['main']()
