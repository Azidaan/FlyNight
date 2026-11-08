import sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_bbox_confirmation_inventory_20261108.common import *


def main():
    began=time.monotonic()
    while not (RUN/'deadline_registered.json').exists() and time.monotonic()-began<10:time.sleep(.1)
    assert (RUN/'deadline_registered.json').exists();phase()
    # Reuse the guarded sequential launcher while directing only service paths.
    import research_native_confirm_risk_20261108.controller as launcher
    launcher.RUN=RUN;launcher.status=status
    for name in ['catalog_check','postcheck','export']:launcher.run(name,f'research_bbox_confirmation_inventory_20261108/{name}.py')
    write(RUN/'queue_result.json',dict(passed=True,inventory_sha256=sha(RUN/'inventory.json')),new=True);status('bbox_inventory_finished')


if __name__=='__main__':main()
