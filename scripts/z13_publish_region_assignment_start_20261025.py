"""Publish exact prepared sources and audited launch, preserving foreign edits."""
from pathlib import Path
import sys
import subprocess
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_candidate_reserve_20261021 import read,write,sha,now
from scripts.z13_localization_labels_20261023 import verify_sources
from scripts.z13_publish_native20_final_20261024 import foreign_unchanged,graph_rows
RUN=ROOT/'runs/z13_region_assignment_20261025'


def main():
    assert not (RUN/'start_publication.json').exists()
    verify_sources(read(RUN/'protocol.json'))
    assert read(RUN/'prepare_tests_result.json')['exit_code']==0
    assert read(RUN/'roi_preflight.json')['all_choices_equal_independent_gate']
    validation=(RUN/'prepare_validation.log').read_bytes()
    assert 'OK:' in validation.decode('utf-16' if validation[:2]==b'\xff\xfe' else 'utf-8-sig')
    foreign=foreign_unchanged();receipt=RUN/'start_owned_graph.json'
    write(receipt,{n:sha(ROOT/n) for _,n in graph_rows() if (ROOT/n).is_file() and n not in foreign})
    subprocess.run([str(ROOT/'.venv/Scripts/python.exe'),'scripts/z13_build_vault_preserving_edits.py','--report',str(RUN/'vault_start.json'),'--refresh-owned',str(receipt)],cwd=ROOT,check=True,capture_output=True)
    names=['tracking/z13_region_assignment.py','tests/test_z13_region_assignment.py','scripts/z13_region_assignment_20261025.py','scripts/z13_region_assignment_console.ps1',
        'scripts/z13_region_assignment_preflight_20261025.py','scripts/z13_publish_region_assignment_start_20261025.py','experiments/project_history.json','dissertation/ЕДИНЫЙ_ПЛАН.md','runs/z13_weekend_20261024/HANDOFF.md']
    names += [(RUN/n).relative_to(ROOT).as_posix() for n in ('protocol.json','launch.json','pipeline.json','roi_preflight.json','prepare_tests_result.json','start_owned_graph.json','vault_start.json','labels/protocol.json','labels/launch.json','labels/controller.json')]
    names += [n for _,n in graph_rows() if n not in foreign and (ROOT/n).is_file()]
    names=sorted(set(names));assert not set(names)&foreign and '.gitignore' not in names
    staged=set(subprocess.check_output(['git','diff','--cached','--name-only','-z'],cwd=ROOT).decode().split('\0'))-{''};assert staged<=set(names)
    assert not (ROOT/'.git/index.lock').exists()
    assert subprocess.check_output(['git','branch','--show-current'],cwd=ROOT).decode().strip()=='main'
    assert subprocess.check_output(['git','remote','get-url','origin'],cwd=ROOT).decode().strip()=='https://gitlab.com/aziteam-group/drbdiss.git'
    for i in range(0,len(names),25):subprocess.run(['git','add','-f','--',*names[i:i+25]],cwd=ROOT,check=True,capture_output=True)
    subprocess.run(['git','diff','--cached','--check'],cwd=ROOT,check=True,capture_output=True)
    subprocess.run(['git','commit','--quiet','-m','Prepare matched learned region readout with bounded local pipeline','-m','Co-authored-by: Codex <codex@openai.com>'],cwd=ROOT,check=True,capture_output=True)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip();subprocess.run(['git','push','origin','main'],cwd=ROOT,check=True)
    foreign_unchanged();write(RUN/'start_publication.json',dict(at=now().isoformat(),commit=commit,push_completed=True,paths=names))


if __name__=='__main__':main()
