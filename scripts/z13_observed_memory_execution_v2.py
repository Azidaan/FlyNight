"""Service recovery: shared GPU queue and retry atomic saves on Windows.

Original scientific source, first service version and failure logs are retained.
Only filesystem replacement, queue name, log names and deadline cap are adapted.
"""
from pathlib import Path
import hashlib
import json
import time


def replace_with_retry(source,target):
    for attempt in range(40):
        try:
            source.replace(target)
            return
        except PermissionError:
            if attempt==39:raise
            time.sleep(.025)


def main():
    service=Path(__file__).resolve();root=service.parents[1]
    run=root/'runs/z13_observed_memory_20261021'
    receipt=json.loads((run/'recovery1_protocol.json').read_text(encoding='utf-8'))
    source=service.with_name('z13_observed_memory_20261021.py')
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    assert digest(source)==receipt['original_sha256']
    assert digest(service)==receipt['service_sha256']
    original=json.loads((run/'full_initial_controller.json').read_text(encoding='utf-8'))
    code=source.read_text(encoding='utf-8')
    changes={
        "GpuLock('FlyVis_z13_observed_memory_single_gpu')":"GpuLock('flyvis_fovea_gpu')",
        'q.replace(p)':'_replace_with_retry(q,p)',
        "marker=RUN/(phase+'_controller_result.json')":"marker=RUN/(phase+'_recovery1_controller_result.json')",
        "phase+'.stdout.log'":"phase+'_recovery1.stdout.log'",
        "phase+'.stderr.log'":"phase+'_recovery1.stderr.log'",
        "hard=min(now()+timedelta(seconds=budget+15),schedule()-timedelta(seconds=10))":"hard=min(now()+timedelta(seconds=budget+15),schedule()-timedelta(seconds=10),datetime.fromisoformat(_recovery_window['hard_stop_at']))",
    }
    for before,after in changes.items():
        assert code.count(before)==1,before
        code=code.replace(before,after)
    exec(compile(code,str(source),'exec'),{'__name__':'__main__','__file__':str(service),
         '_replace_with_retry':replace_with_retry,'_recovery_window':original})

if __name__=='__main__':main()
