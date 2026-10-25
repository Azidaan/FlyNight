"""Publish the measured math gate and bounded real-pilot preparation, exact files."""
from pathlib import Path
import sys
import subprocess
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_candidate_reserve_20261021 import read,write,sha,now
from scripts.z13_publish_native20_final_20261024 import foreign_unchanged,graph_rows
MATH=ROOT/'runs/z13_math_priority_20261025'
RUN=ROOT/'runs/z13_box_supervision_followup_20261025'


def main():
    assert not (MATH/'publication.json').exists()
    assert read(MATH/'validation.json')['tests_passed']==63
    assert read(MATH/'report.json')['complete']
    foreign=foreign_unchanged()
    for n,h in read(MATH/'protocol.json')['sources'].items():assert sha(ROOT/n)==h
    for n,h in read(RUN/'protocol.json')['sha256'].items():assert sha(ROOT/n)==h
    assert read(MATH/'vault_start.json')['official_generator_completed']
    assert not (ROOT/'.git/index.lock').exists()
    git=lambda *a:subprocess.check_output(['git',*a],cwd=ROOT).decode('utf8').strip()
    assert git('branch','--show-current')=='main'
    assert git('remote','get-url','origin')=='https://gitlab.com/aziteam-group/drbdiss.git'
    assert not git('diff','--cached','--name-only')
    names=['scripts/z13_math_priority_20261025.py','scripts/z13_box_supervision_followup_20261025.py',
        Path(__file__).relative_to(ROOT).as_posix(),'tracking/z13_action_feasibility.py','tests/test_z13_action_feasibility.py',
        'research_flyvis_deep_2026-10-25/ПРИОРИТЕТ_И_МАТЕМАТИКА.md',
        'research_flyvis_deep_2026-10-25/publication.json',
        'experiments/project_history.json','dissertation/ЕДИНЫЙ_ПЛАН.md','runs/z13_weekend_20261024/HANDOFF.md']
    names += [p.relative_to(ROOT).as_posix() for p in MATH.glob('*') if p.suffix in ('.json','.md') and p.name!='publication.json']
    names += [(RUN/n).relative_to(ROOT).as_posix() for n in ('protocol.json','launch.json','controller.json','cache/protocol.json','cache/launch.json','cache/controller.json','cache/price.json')]
    names += [n for _,n in graph_rows() if n not in foreign and (ROOT/n).is_file()]
    names=sorted(set(names));assert not set(names)&foreign and '.gitignore' not in names
    for n in names:assert (ROOT/n).is_file(),n
    for i in range(0,len(names),20):subprocess.run(['git','-c','core.autocrlf=input','add','-f','--',*names[i:i+20]],cwd=ROOT,check=True,capture_output=True)
    staged=set(git('diff','--cached','--name-only','-z').split('\0'))-{''};assert staged and staged<=set(names)
    subprocess.run(['git','diff','--cached','--check'],cwd=ROOT,check=True)
    subprocess.run(['git','commit','-m','Bound action reserves before the matched full-box readout pilot','-m','Co-authored-by: Codex <noreply@openai.com>'],cwd=ROOT,check=True)
    commit=git('rev-parse','HEAD');subprocess.run(['git','push','origin','main'],cwd=ROOT,check=True)
    foreign_unchanged();write(MATH/'publication.json',dict(at=now().isoformat(),commit=commit,push_completed=True,paths=names))
    print(commit)


if __name__=='__main__':main()
