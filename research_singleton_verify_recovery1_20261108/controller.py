"""Summary-only repair, independent current audit, then parent-result audit."""
import sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_singleton_verify_20261108.common import *


def main():
    begun=time.monotonic()
    while not (RUN/'summary_recovery1_deadline_registered.json').exists() and time.monotonic()-begun<10:time.sleep(.1)
    assert (RUN/'summary_recovery1_deadline_registered.json').exists();phase('summary_recovery1_phase.json')
    from research_singleton_verify_20261108.queue_data import launch
    launch('summary_recovery1','../research_singleton_verify_recovery1_20261108/aggregate.py')
    source=read(RUN/'comparison_phase.json')['sources'].copy();source.update(read(RUN/'summary_recovery1_phase.json')['sources'])
    for path in [ROOT/'research_singleton_verify_20261108/audit.py',ROOT/'research_singleton_free_20261108/audit.py',RUN/'comparison_report.json',RUN/'selection.json',RUN/'scores_complete.json']:
        source[path.relative_to(ROOT).as_posix()]=sha(path)
    write(RUN/'audit_phase.json',dict(registered=datetime.now(MSK).isoformat(),sources=source),new=True)
    launch('independent_audit','audit.py');assert read(RUN/'independent_audit.json')['passed']
    write(RUN/'summary_recovery1_result.json',dict(passed=True,audit_sha256=sha(RUN/'independent_audit.json'),full_gate=read(RUN/'comparison_report.json')['full_tracker_gate']),new=True)
    # The former observer has stopped on the preserved summary error.
    sources={}
    for name in ['full_precision_execution_phase.json','full_statistics_phase.json','full_additional_phase.json']:
        for n,h in read(BASE/name)['sources'].items():assert sha(ROOT/n)==h;sources[n]=h
    for path in [ROOT/'research_identity_assignment_run_20261108/audit_full_result.py',ROOT/'research_singleton_free_20261108/full_audit.py',
        ROOT/'research_singleton_verify_20261108/common.py',BASE/'full_precision/comparison_report.json',BASE/'full_precision/additional_report.json',BASE/'full_precision/analyses_complete.json']:
        sources[path.relative_to(ROOT).as_posix()]=sha(path)
    write(BASE/'parent_full_audit_phase.json',dict(registered=datetime.now(MSK).isoformat(),sources=sources,read_only=True),new=True)
    launch('parent_full_audit','../research_identity_assignment_run_20261108/audit_full_result.py')
    assert read(BASE/'full_precision/independent_audit.json')['passed']


if __name__=='__main__':main()
