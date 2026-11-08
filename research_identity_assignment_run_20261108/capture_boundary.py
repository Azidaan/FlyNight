"""Record the shared worktree and protected notes before result publication."""
import subprocess
from research_identity_assignment_run_20261108.common import *


def main():
    assert subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()=='codex/system-integrity-20261106-e8f674ce'
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==read(RUN/'execution_protocol.json')['base_commit']
    prior=read(ROOT/'runs/spatial_identity_maps_20261107/graph_protected_before_recovery1.json')
    protected={name:sha(ROOT/'ObsidianFlyVis'/name) for name in prior}
    assert len(protected)==25
    write(RUN/'graph_protected_before.json',protected,new=True)
    files=sorted(p.relative_to(ROOT/'ObsidianFlyVis').as_posix() for p in (ROOT/'ObsidianFlyVis').rglob('*') if p.is_file())
    write(RUN/'graph_all_files_before.json',files,new=True)
    plan=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';text=plan.read_text(encoding='utf-8')
    assert '**07.11.2026, 20:43 MSK**' in text and read(RUN/'registration_journal.json')['entry'] in text
    write(RUN/'working_tree_boundary.json',dict(captured=datetime.now(MSK).isoformat(),
        preexisting_tracker_plan_sha256=sha(ROOT/'dissertation/План целей по Трекеру.md'),
        current_chronology_sha256=sha(plan),foreign_journal_entry_to_preserve='07.11.2026, 20:43 MSK',
        own_journal_entries=['runs/identity_assignment_20261108/registration_journal.json'],
        staging_policy='Stage only task files and own appended chronology entries; preserve unrelated plan edits and all earlier files.'),new=True)


if __name__=='__main__':main()
