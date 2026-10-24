"""Unchanged result audit/recording, bound to the separate recovery directory."""
from pathlib import Path
import inspect
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import z13_finish_template_swap_20261025 as original
from scripts.z13_template_swap_recovery1_20261025 import RUN,OLD,read,write,now


def main():
    ns={**vars(original),'RUN':RUN}
    for name in ('independent_audit','record','main'):
        exec(compile(inspect.getsource(getattr(original,name)),str(Path(__file__)),'exec'),ns)
    source=inspect.getsource(original.publish)
    prefix="names=['tracking/z13_template_swap.py'"
    assert source.count(prefix)==1
    extra=['scripts/z13_template_swap_recovery1_20261025.py','scripts/z13_finish_template_swap_recovery1_20261025.py',
        'scripts/z13_template_swap_console_utf8.ps1','scripts/z13_template_swap_recovery1_console.ps1']
    extra += [x.relative_to(ROOT).as_posix() for x in OLD.glob('*.json')]
    source=source.replace(prefix,"names=RECOVERY_FILES+['tracking/z13_template_swap.py'")
    ns['RECOVERY_FILES']=extra
    exec(compile(source,str(Path(__file__)),'exec'),ns)
    ns['main']()


if __name__=='__main__':
    try:main()
    except Exception as exc:
        write(RUN/'monitor_failure.json',dict(at=now().isoformat(),error=repr(exc)));raise
