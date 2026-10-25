"""Publish only the verified full-box prototype and its architecture milestone."""
from pathlib import Path
import sys
import subprocess
import os
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_candidate_reserve_20261021 import read,write,sha,now
from scripts.z13_publish_native20_final_20261024 import foreign_unchanged,graph_rows
RUN=ROOT/'runs/z13_box_proposal_20261025'


def main():
    assert not (RUN/'publication.json').exists()
    assert read(RUN/'finish_recorded.json')['complete']
    assert read(RUN/'architecture_milestone/review.json')['overview_and_three_readable_branch_crops_viewed']
    foreign=foreign_unchanged()
    for name,args in [('catalog',['scripts/validate_project_history.py']),('tests',['-m','pytest','tests/test_project_history.py','tests/test_z13_proposal_geometry.py','tests/test_z13_box_proposal.py','-q'])]:
        with (RUN/('final_'+name+'.log')).open('w',encoding='utf8') as out:
            subprocess.run([str(ROOT/'.venv/Scripts/python.exe'),*args],cwd=ROOT,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONIOENCODING':'utf-8'},stdout=out,stderr=subprocess.STDOUT,check=True)
    receipt=RUN/'owned_graph.json';write(receipt,{n:sha(ROOT/n) for _,n in graph_rows() if n not in foreign and (ROOT/n).is_file()})
    subprocess.run([str(ROOT/'.venv/Scripts/python.exe'),'scripts/z13_build_vault_preserving_edits.py','--report',str(RUN/'vault_final.json'),'--refresh-owned',str(receipt)],cwd=ROOT,check=True,capture_output=True)
    foreign_unchanged()
    names=['scripts/z13_box_proposal_20261025.py','tracking/z13_box_proposal.py','tests/test_z13_box_proposal.py',
        'scripts/z13_record_box_proposal_20261025.py','scripts/z13_box_proposal_summary_20261025.py',Path(__file__).relative_to(ROOT).as_posix(),
        'scripts/build_flyvis_architecture.py','scripts/flyvis_architecture_layout.py','experiments/project_history.json',
        'dissertation/ЕДИНЫЙ_ПЛАН.md','runs/z13_weekend_20261024/HANDOFF.md','runs/z13_proposal_geometry_20261025/publication.json']
    names += [p.relative_to(ROOT).as_posix() for p in RUN.rglob('*.json') if p.name!='publication.json']
    names += [(ROOT/'dissertation/architecture'/n).relative_to(ROOT).as_posix() for n in (
        'FlyVis_architecture.drawio','FlyVis_architecture.svg','FlyVis_architecture.png',
        'FlyVis_architecture_preview.png','FlyVis_architecture_detail.png','FlyVis_architecture_sources.json','validation.json')]
    names += [n for _,n in graph_rows() if n not in foreign and (ROOT/n).is_file()]
    names=sorted(set(names));assert not set(names)&foreign and '.gitignore' not in names
    assert not (ROOT/'.git/index.lock').exists()
    staged=lambda:set(subprocess.check_output(['git','diff','--cached','--name-only','-z'],cwd=ROOT).decode().split('\0'))-{''}
    assert not staged()
    assert subprocess.check_output(['git','remote','get-url','origin'],cwd=ROOT).decode().strip()=='https://gitlab.com/aziteam-group/drbdiss.git'
    for i in range(0,len(names),20):subprocess.run(['git','-c','core.autocrlf=input','add','-f','--',*names[i:i+20]],cwd=ROOT,check=True,capture_output=True)
    assert staged()==set(names)
    subprocess.run(['git','diff','--cached','--check'],cwd=ROOT,check=True,capture_output=True)
    subprocess.run(['git','commit','-m','Evaluate matched FlyVis center versus complete-box proposal output','-m','Co-authored-by: Codex <noreply@openai.com>'],cwd=ROOT,check=True,capture_output=True)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip()
    subprocess.run(['git','push','origin','main'],cwd=ROOT,check=True)
    foreign_unchanged();write(RUN/'publication.json',dict(at=now().isoformat(),commit=commit,push_completed=True,paths=names));print(commit)


if __name__=='__main__':main()
