"""Bounded retry only for service progress writes; scientific writes unchanged."""
import time
from research_identity_assignment_run_20261108.common import write,datetime,MSK


def bounded_write(path,value,attempts=20,delay=.05):
    for attempt in range(attempts):
        try:write(path,value);return attempt+1
        except PermissionError:
            if attempt+1==attempts:raise
            time.sleep(delay)


def for_run(folder):
    def status(stage,**values):
        # Freeze the source timestamp once; retries must not fake progress.
        value=dict(stage=stage,updated=datetime.now(MSK).isoformat(),**values)
        return bounded_write(folder/'status.json',value)
    return status
