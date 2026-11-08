"""Isolate frozen research scripts from their queue.py stdlib name collision."""
import os,sys,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
TARGET_DIR=ROOT/'research_singleton_free_20261108'
ENTRY=Path(__file__).with_name('entry.py')
KEY='SINGLETON_RECOVERY2_TARGET_MODULE'
ORIGINAL_POPEN=subprocess.Popen


def install():
    # Both the initial child and the memory guardian's re-entered child use
    # this directory as argv[0], where there is no queue.py.
    sys.path[:]=[p for p in sys.path if Path(p or os.getcwd()).resolve()!=TARGET_DIR.resolve()]
    if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
    import queue
    assert Path(queue.__file__).resolve().parent!=TARGET_DIR.resolve()
    assert hasattr(queue,'Queue') and hasattr(queue,'SimpleQueue')
    if getattr(subprocess.Popen,'singleton_recovery2',False):return
    class IsolatedPopen(ORIGINAL_POPEN):
        singleton_recovery2=True
        def __init__(self,args,*positional,**kwargs):
            if isinstance(args,(list,tuple)) and len(args)>=3:
                target=Path(args[2])
                if not target.is_absolute():target=Path(kwargs.get('cwd') or os.getcwd())/target
                target=target.resolve()
                if target.parent==TARGET_DIR.resolve() and target.suffix=='.py':
                    environment=dict(os.environ if kwargs.get('env') is None else kwargs['env'])
                    environment[KEY]='research_singleton_free_20261108.'+target.stem
                    kwargs['env']=environment;args=[*args[:2],str(ENTRY),*args[3:]]
            super().__init__(args,*positional,**kwargs)
    subprocess.Popen=IsolatedPopen
