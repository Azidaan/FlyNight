"""Isolated wrapper for the corrected independent evaluator only."""
from pathlib import Path
import importlib.util
import sys
folder=Path(__file__).resolve().parent
sys.path=[p for p in sys.path if Path(p or '.').resolve()!=folder]
sys.path.insert(0,str(folder.parent))
spec=importlib.util.spec_from_file_location('common',folder/'common.py')
module=importlib.util.module_from_spec(spec);sys.modules['common']=module;spec.loader.exec_module(module)
target=folder/'audit_recovery2.py'
exec(compile(target.read_bytes(),str(target),'exec'),dict(__name__='__main__',__file__=str(target)))
