"""Publish only verified replay findings and the bounded native continuation."""
from pathlib import Path
import subprocess,sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_candidate_reserve_20261021 import read,write,now
W=ROOT/'runs/z13_weekend_20261024';RUN=ROOT/'runs/z13_val20_matched_20261024'


def main():
    e=read(W/'native_transition_checks.json');assert e['catalog_exit']==e['tests_exit']==0
    assert '58 passed' in (W/'native_transition_tests.log').read_text(encoding='utf-8')
    assert read(W/'val20_replay_pending_author_decision.json')['status']=='pending'
    names=['AGENTS.md','experiments/project_history.json','dissertation/ЕДИНЫЙ_ПЛАН.md']+['scripts/'+n for n in ('z13_val20_replay_diagnostic_20261024.py','z13_val20_native_masa_20261024.py','z13_val20_native_masa_console.ps1','z13_val20_native_masa_recovery1_20261024.py','z13_record_native_transition_20261024.py','z13_publish_native_transition_20261024.py')]
    for stage in (RUN/'common',RUN/'common_replay_diagnostic',RUN/'native_masa',RUN/'native_masa_recovery1'):
        for n in ('protocol.json','controller.json','controller_result.json','launch.json','completion.json','report.json','previous_difference.json','source_and_feature_audit.json','late_child_completion_audit.json'):
            if (stage/n).is_file():names.append((stage/n).relative_to(ROOT).as_posix())
        for f in (stage/'jobs').glob('*.json'):names.append(f.relative_to(ROOT).as_posix())
    for n in ('HANDOFF.md','val20_common_failure.json','val20_native_masa_failure.json','val20_replay_pending_author_decision.json','native_transition_checks.json','native_transition_vault.json'):names.append((W/n).relative_to(ROOT).as_posix())
    protected={n.replace(chr(92),'/') for n in read(W/'vault_preparation.json')['protected_existing_edits']};deleted=set()
    rows=subprocess.check_output(['git','status','--porcelain','-z','--untracked-files=all','--','ObsidianFlyVis'],cwd=ROOT).decode().split(chr(0))
    for row in rows:
        if row:
            n=row[3:].replace(chr(92),'/')
            if n not in protected:
                if (ROOT/n).is_file():names.append(n)
                elif 'D' in row[:2]:names.append(n);deleted.add(n)
    names=sorted(set(names));assert all((ROOT/n).is_file() or n in deleted for n in names);assert not set(names)&protected and '.gitignore' not in names and not (ROOT/'.git/index.lock').exists()
    staged=set(subprocess.check_output(['git','diff','--cached','--name-only','-z'],cwd=ROOT).decode().split(chr(0)))-{''};assert staged<=set(names)
    for start in range(0,len(names),25):subprocess.run(['git','add','-f','--',*names[start:start+25]],cwd=ROOT,check=True,capture_output=True)
    subprocess.run(['git','diff','--cached','--check'],cwd=ROOT,check=True,capture_output=True)
    subprocess.run(['git','commit','--quiet','-m','Preserve strict FlyVis replay failure and continue matched native MASA baseline','-m','Co-authored-by: Codex <noreply@openai.com>'],cwd=ROOT,check=True,capture_output=True)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip();print(json.dumps(dict(commit=commit,files=len(names))),flush=True)
    subprocess.run(['git','push','origin','main'],cwd=ROOT,check=True);write(W/'native_transition_publication.json',dict(at=now().isoformat(),commit=commit,push_completed=True,paths=names))


if __name__=='__main__':main()
