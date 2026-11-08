"""Wait for the registered gate, then execute the complete pilot sequentially."""
import sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_native_confirm_full_20261108.common import *


def main():
    import psutil
    root=read(RUN/'started.json')
    while not (RUN/'queue_result.json').exists():
        if datetime.now(MSK)>=datetime(2026,11,8,13,55,tzinfo=MSK):return
        try:p=psutil.Process(root['pid']);assert abs(p.create_time()-root['created'])<.01
        except psutil.NoSuchProcess:raise RuntimeError('The risk-gate queue ended before its independent audit; preserve and recover that error.')
        time.sleep(5)
    from research_native_confirm_full_20261108.register import main as register
    if not register():return
    from research_native_confirm_risk_20261108.controller import run
    for name in ['catalog_check','native_trace','features','rollouts','evaluate','additional','statistics','audit']:run('full_'+name,f'research_native_confirm_full_20261108/{name}.py')
    assert read(FULL/'independent_audit.json')['passed'];write(FULL/'queue_result.json',dict(passed=True,comparison_sha256=sha(FULL/'comparison_report.json'),independent_audit_sha256=sha(FULL/'independent_audit.json')),new=True);status('native_confirm_full_pilot_complete')


if __name__=='__main__':main()
