"""A stable argv[0] for the same guarded module before and after re-entry."""
import os,sys,runpy,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_singleton_free_recovery2_20261108.bootstrap import install,KEY


def main():
    install();module=os.environ[KEY]
    assert module.startswith('research_singleton_free_20261108.') and module.count('.')==1
    manifest=json.loads((ROOT/'runs/singleton_free_20261108/recovery2_phase.json').read_text(encoding='utf-8'))
    for name,expected in manifest['sources'].items():
        actual=hashlib.sha256((ROOT/name).read_bytes()).hexdigest();assert actual==expected,name
    runpy.run_module(module,run_name='__main__',alter_sys=False)


if __name__=='__main__':main()
