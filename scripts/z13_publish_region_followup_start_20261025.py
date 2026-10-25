"""Publish only our completed audit and prepared balanced-readout follow-up."""
from pathlib import Path
import sys
import subprocess
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_candidate_reserve_20261021 import read,write,sha,now
from scripts.z13_publish_native20_final_20261024 import foreign_unchanged,graph_rows
RUN=ROOT/'runs/z13_region_balanced_20261025'


def main():
    assert not (RUN/'start_publication.json').exists()
    assert '59 passed' in (RUN/'prepare_recovery1_tests.log').read_text(encoding='utf-8')
    assert 'OK:' in (RUN/'prepare_recovery1_validation.log').read_text(encoding='utf-8')
    assert read(ROOT/'runs/z13_region_assignment_20261025/recovery2/publication.json')['push_completed']
    audit=ROOT/'runs/z13_region_fit_audit_20261025';assert read(audit/'report.json')['complete']
    e=read(audit/'controller_result.json');assert e['exit_code']==0 and e['all_owned_ended']
    write(RUN/'prepare_verified.json',dict(at=now().isoformat(),validator_exit0=True,tests59_passed=True,
       tests_sha256=sha(RUN/'prepare_recovery1_tests.log'),scientific_protocol_sha256=sha(RUN/'protocol.json'),
       metadata_repair_only=True,confirmed_results_review='Previous unweighted regional primary intervals all cross0; no new confirmed result added. P8 remains partial.'))
    foreign=foreign_unchanged();receipt=RUN/'start_owned_graph.json'
    write(receipt,{n:sha(ROOT/n) for _,n in graph_rows() if (ROOT/n).is_file() and n not in foreign})
    subprocess.run([str(ROOT/'.venv/Scripts/python.exe'),'scripts/z13_build_vault_preserving_edits.py','--report',str(RUN/'vault_start.json'),'--refresh-owned',str(receipt)],cwd=ROOT,check=True,capture_output=True)
    names=[Path(__file__).relative_to(ROOT).as_posix(),'scripts/z13_region_fit_audit_20261025.py','scripts/z13_region_fit_audit_console.ps1',
       'scripts/z13_region_balanced_20261025.py','scripts/z13_region_balanced_console.ps1','scripts/z13_region_balanced_evidence_repair_20261025.py',
       'tracking/z13_region_balanced_loss.py','tests/test_z13_region_balanced_loss.py','experiments/project_history.json','dissertation/ЕДИНЫЙ_ПЛАН.md']
    names += [p.relative_to(ROOT).as_posix() for p in audit.glob('*.json')]
    names += [(RUN/n).relative_to(ROOT).as_posix() for n in ('protocol.json','training/protocol.json','metadata_repair.json','prepare_verified.json','launch.json','start_owned_graph.json','vault_start.json')]
    names += [n for _,n in graph_rows() if n not in foreign and (ROOT/n).is_file()]
    names=sorted(set(names));assert not set(names)&foreign and '.gitignore' not in names
    staged=set(subprocess.check_output(['git','diff','--cached','--name-only','-z'],cwd=ROOT).decode().split('\0'))-{''};assert staged<=set(names)
    assert not (ROOT/'.git/index.lock').exists()
    assert subprocess.check_output(['git','remote','get-url','origin'],cwd=ROOT).decode().strip()=='https://gitlab.com/aziteam-group/drbdiss.git'
    assert subprocess.check_output(['git','branch','--show-current'],cwd=ROOT).decode().strip()=='main'
    for i in range(0,len(names),25):subprocess.run(['git','add','-f','--',*names[i:i+25]],cwd=ROOT,check=True,capture_output=True)
    subprocess.run(['git','diff','--cached','--check'],cwd=ROOT,check=True)
    subprocess.run(['git','commit','--quiet','-m','Register class-balanced regional readout after fit-error audit','-m','Co-authored-by: Codex <codex@openai.com>'],cwd=ROOT,check=True)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip();subprocess.run(['git','push','origin','main'],cwd=ROOT,check=True)
    foreign_unchanged();write(RUN/'start_publication.json',dict(at=now().isoformat(),commit=commit,push_completed=True,paths=names))


if __name__=='__main__':main()
