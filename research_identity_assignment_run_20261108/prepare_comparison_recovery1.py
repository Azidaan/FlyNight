"""Version the Windows manifest-key recovery without changing scientific conditions."""
from research_identity_assignment_run_20261108.common import *


def main():
    folder=ROOT/'research_identity_assignment_run_20261108'
    old=(folder/'simulate.py').read_text(encoding='utf-8')
    a="self.export['sources'][raw.relative_to(ROOT).as_posix()]"
    b="{Path(n).as_posix():h for n,h in self.export['sources'].items()}[raw.relative_to(ROOT).as_posix()]"
    assert old.count(a)==1
    with (folder/'simulate_recovery1.py').open('x',encoding='utf-8',newline='\n') as f:f.write(old.replace(a,b))
    old=(folder/'compare.py').read_text(encoding='utf-8')
    a='from research_identity_assignment_run_20261108.simulate import'
    assert old.count(a)==1
    new=old.replace(a,'from research_identity_assignment_run_20261108.simulate_recovery1 import').replace("'comparison_phase.json'","'comparison_recovery1_phase.json'")
    with (folder/'compare_recovery1.py').open('x',encoding='utf-8',newline='\n') as f:f.write(new)
    p=read(RUN/'comparison_phase.json');p['registered']=datetime.now(MSK).isoformat()
    names=['research_identity_assignment_run_20261108/simulate_recovery1.py',
        'research_identity_assignment_run_20261108/compare_recovery1.py',
        'research_identity_assignment_run_20261108/prepare_comparison_recovery1.py',
        'research_identity_assignment_run_20261108/resume_comparison_recovery1.py']
    p['sources'].update({n:sha(ROOT/n) for n in names})
    p['recovery']=dict(reason='Normalize Windows backslash source-manifest keys before hash lookup. Values, labels, costs, grids and selection rules unchanged.',
        original_phase_sha256=sha(RUN/'comparison_phase.json'),
        failed_guard_report='runs/flyvis_repairs_20261106/unification_20261107/uncapped_checks/78aad7b39fa347eb976e9ca42642c69d/memory_guard.json',
        failed_log_sha256=sha(ROOT/'runs/flyvis_repairs_20261106/unification_20261107/uncapped_checks/78aad7b39fa347eb976e9ca42642c69d/memory_guard.worker.log'),
        completed_scientific_stages_reused=['zero_parity','export_complete','input_complete','features_complete','scores_complete'],
        comparison_outputs_before_failure=0)
    assert not (RUN/'selection.json').exists() and not (RUN/'restored_zero_parity.json').exists()
    write(RUN/'comparison_recovery1_phase.json',p,new=True)


if __name__=='__main__':main()
