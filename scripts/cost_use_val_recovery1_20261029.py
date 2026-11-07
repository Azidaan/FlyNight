"""Same frozen cost experiment; require the repaired preparation controller."""
from pathlib import Path
import sys,inspect
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import cost_use_val_20261029 as original

def main():
    source=inspect.getsource(original.main)
    old="source=inspect.getsource(engine.main).replace('memory_write_val','cost_use_val')"
    assert source.count(old)==1
    source=source.replace(old,old+"\n    source=source.replace(\"RUN/'cost_use_val_preparation/controller_result.json'\",\"RUN/'cost_use_val_preparation_recovery1/controller_result.json'\")")
    ns=dict(vars(original));exec(compile(source,__file__,'exec'),ns);ns['main']()

if __name__=='__main__':main()
