"""Isolate frozen stage imports from the local queue.py filename."""
from pathlib import Path
import importlib.util
import sys

folder=Path(__file__).resolve().parent
assert sys.argv[1] in ('recover.py','compare.py','audit.py')
target=folder/sys.argv[1]
sys.path=[p for p in sys.path if Path(p or '.').resolve()!=folder]
sys.path.insert(0,str(folder.parent))
spec=importlib.util.spec_from_file_location('common',folder/'common.py')
module=importlib.util.module_from_spec(spec);sys.modules['common']=module;spec.loader.exec_module(module)
# Keep argv[0] pointing to this wrapper so the independent memory supervisor
# re-enters the same import isolation rather than the shadowing script folder.
exec(compile(target.read_bytes(),str(target),'exec'),dict(__name__='__main__',__file__=str(target)))
