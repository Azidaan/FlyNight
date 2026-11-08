"""Finish the authorized conditional comparison with independent verification."""
import sys,time,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_identity_assignment_run_20261108.common import *


def main():
    while not (RUN/'comparison_queue_result.json').exists():
        if datetime.now(MSK)>=datetime(2026,11,9,7,35,tzinfo=MSK):raise TimeoutError('MSK deadline')
        time.sleep(2)
    prior=read(RUN/'comparison_queue_result.json')
    if not prior['passed']:
        write(RUN/'audit_queue_result.json',dict(passed=False,stage='comparison',prior=prior),new=True);return
    time.sleep(2)
    base=['research_identity_assignment_run_20261108/common.py','research_identity_assignment_run_20261108/queue_audit.py',
        'runs/identity_assignment_20261108/comparison_report.json','runs/identity_assignment_20261108/selection.json',
        'runs/identity_assignment_20261108/scores_complete.json','runs/identity_assignment_20261108/export_complete.json',
        'tracking/spatial_identity_assignment.py']
    for stage,script in [('audit','audit_result.py'),('diagnostic','diagnose.py')]:
        sources=base+['research_identity_assignment_run_20261108/'+script]
        if stage=='diagnostic':sources.append('runs/identity_assignment_20261108/independent_audit.json')
        write(RUN/(stage+'_phase.json'),dict(registered=datetime.now(MSK).isoformat(),sources={n:sha(ROOT/n) for n in sources}),new=True)
        with (RUN/(stage+'_launcher.log')).open('x',encoding='utf-8') as out,(RUN/(stage+'_launcher.err')).open('x',encoding='utf-8') as err:
            result=subprocess.run([sys.executable,'-u','research_identity_assignment_run_20261108/'+script],cwd=ROOT,stdout=out,stderr=err)
        if result.returncode!=0:
            write(RUN/'audit_queue_result.json',dict(passed=False,stage=stage,returncode=result.returncode),new=True);return
        time.sleep(2)
    write(RUN/'audit_queue_result.json',dict(passed=True,audit_sha256=sha(RUN/'independent_audit.json'),
        mechanism_sha256=sha(RUN/'mechanism_report.json')),new=True)
    status('audited_conditional_comparison_complete',full_tracker_gate=read(RUN/'comparison_report.json')['full_tracker_gate'])


if __name__=='__main__':main()
