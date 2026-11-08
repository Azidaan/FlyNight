"""Preserve failed serialization and resume after the completed coverage calculation."""
from research_identity_assignment_run_20261108.common import *
from research_identity_assignment_run_20261108.full_retina import FULL


def main():
    import subprocess
    failed=RUN/'coverage_followup.json';partial=RUN/'coverage_followup.failed_partial.json'
    assert failed.exists() and not partial.exists();failed.rename(partial)
    from research_identity_assignment_run_20261108.coverage_summary_recovery1 import main as recover
    recover();assert read(RUN/'coverage_followup_recovery1.json')['passed']
    folder=ROOT/'research_identity_assignment_run_20261108';old=(folder/'full_master.py').read_text(encoding='utf-8')
    line="if not (RUN/'coverage_followup.json').exists():child('coverage','all','coverage_followup.py',[],obs)"
    assert old.count(line)==1
    new=old.replace(line,"assert read(RUN/'coverage_followup_recovery1.json')['passed']")
    new=new.replace("'full_execution_phase.json'","'full_execution_recovery1_phase.json'")
    with (folder/'full_master_recovery1.py').open('x',encoding='utf-8',newline='\n') as f:f.write(new)
    p=read(RUN/'full_execution_phase.json');p['registered']=datetime.now(MSK).isoformat()
    names=['research_identity_assignment_run_20261108/'+n for n in ['coverage_summary_recovery1.py','prepare_full_recovery1.py','full_master_recovery1.py']]
    names+=['runs/identity_assignment_20261108/coverage_rows.json','runs/identity_assignment_20261108/coverage_followup_recovery1.json']
    p['sources'].update({n:sha(ROOT/n) for n in names})
    p['recovery']=dict(reason='Summary serialization only; reuse all completed coverage rows, no geometric or neural recalculation. Full video rollout has not started.',
        failed_parent_guard='963cc0db7f7d4e59a2b999d533fbe99a',failed_child_log_sha256=sha(FULL/'children/coverage_all.err'),
        failed_partial_sha256=sha(partial),previous_phase_sha256=sha(RUN/'full_execution_phase.json'),scientific_parameters_changed=False)
    write(RUN/'full_execution_recovery1_phase.json',p,new=True)


if __name__=='__main__':main()
