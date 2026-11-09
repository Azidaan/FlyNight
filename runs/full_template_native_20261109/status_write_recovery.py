"""Bounded retry ONLY for progress writes; scientific sources stay immutable."""
import sys,time,json,runpy
from datetime import datetime
import common


def stable_status(stage,**kwargs):
    # Keep the time of the original event, never refresh it merely on retry.
    value=dict(stage=stage,updated=datetime.now(common.MSK).isoformat(),**kwargs)
    for attempt in range(30):
        try:
            common.write('status.json',value,new=False);return
        except PermissionError as exc:
            with (common.OUT/'progress_write_retries.jsonl').open('a',encoding='utf8') as f:
                f.write(json.dumps(dict(stage=stage,original_updated=value['updated'],attempt=attempt+1,
                    error=str(exc),retry_time=datetime.now(common.MSK).isoformat()))+'\n')
            if attempt==29:raise
            time.sleep(.1)


if __name__=='__main__':
    target=sys.argv[1];arguments=sys.argv[2:]
    marker=arguments.pop() if arguments and arguments[-1].startswith('--audit-memory-worker=') else None
    allowed={'native_run.py','register_evaluation.py','deployment_audit.py','evaluate_native.py','native_statistics.py','id_fate.py'}
    assert target in allowed
    mode=arguments[0] if target=='native_run.py' else None
    common.start('gpu' if mode=='C34' else 'cpu',timeout=3600 if mode=='C34' else 7200)
    assert marker is not None,'Worker must remain under its verified supervisor'
    p=common.read(common.OUT/'status_recovery_protocol.json');common.verify(p)
    common.status=stable_status
    # The unchanged target revalidates the SAME supervisor marker, with no
    # second scientific worker and no weakening of its memory/deadline guard.
    sys.argv=[str(common.OUT/target),*arguments,marker]
    runpy.run_path(str(common.OUT/target),run_name='__main__')
