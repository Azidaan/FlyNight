"""Verified exact-path publication of the proposal geometry gate and service recovery."""
from pathlib import Path
import os
import sys
import subprocess
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.z13_candidate_reserve_20261021 import read,write,sha,now
from scripts.z13_publish_native20_final_20261024 import foreign_unchanged,graph_rows
RUN=ROOT/'runs/z13_proposal_geometry_20261025'


def main():
    assert not (RUN/'publication.json').exists()
    assert read(RUN/'recovery1/finish_recorded.json')['complete']
    foreign=foreign_unchanged()
    for name,args in [('catalog',['scripts/validate_project_history.py']),
                      ('tests',['-m','pytest','tests/test_project_history.py','tests/test_z13_proposal_geometry.py','-q'])]:
        with (RUN/('final_'+name+'.log')).open('w',encoding='utf8') as out:
            subprocess.run([str(ROOT/'.venv/Scripts/python.exe'),*args],cwd=ROOT,env={**os.environ,'PYTHONIOENCODING':'utf-8','CUDA_VISIBLE_DEVICES':''},stdout=out,stderr=subprocess.STDOUT,check=True)
    receipt=RUN/'owned_graph.json'
    write(receipt,{n:sha(ROOT/n) for _,n in graph_rows() if n not in foreign and (ROOT/n).is_file()})
    subprocess.run([str(ROOT/'.venv/Scripts/python.exe'),'scripts/z13_build_vault_preserving_edits.py','--report',str(RUN/'vault_final.json'),'--refresh-owned',str(receipt)],cwd=ROOT,check=True,capture_output=True)
    foreign_unchanged()
    names=['scripts/z13_proposal_geometry_20261025.py','scripts/z13_proposal_geometry_recovery1_20261025.py',
           Path(__file__).relative_to(ROOT).as_posix(),'tracking/z13_proposal_geometry.py','tests/test_z13_proposal_geometry.py',
           'experiments/project_history.json','dissertation/ЕДИНЫЙ_ПЛАН.md','runs/z13_weekend_20261024/HANDOFF.md']
    names += [p.relative_to(ROOT).as_posix() for p in RUN.rglob('*.json') if p.name!='publication.json']
    names += [n for _,n in graph_rows() if n not in foreign and (ROOT/n).is_file()]
    names=sorted(set(names))
    assert not set(names)&foreign and '.gitignore' not in names
    staged=set(subprocess.check_output(['git','diff','--cached','--name-only','-z'],cwd=ROOT).decode().split('\0'))-{''}
    assert not staged and not (ROOT/'.git/index.lock').exists()
    assert subprocess.check_output(['git','remote','get-url','origin'],cwd=ROOT).decode().strip()=='https://gitlab.com/aziteam-group/drbdiss.git'
    for i in range(0,len(names),20):
        subprocess.run(['git','-c','core.autocrlf=input','add','-f','--',*names[i:i+20]],cwd=ROOT,check=True,capture_output=True)
    staged=set(subprocess.check_output(['git','diff','--cached','--name-only','-z'],cwd=ROOT).decode().split('\0'))-{''}
    assert staged==set(names)
    subprocess.run(['git','diff','--cached','--check'],cwd=ROOT,check=True,capture_output=True)
    subprocess.run(['git','commit','-m','Measure missing-proposal geometry and preserve bounded crop recovery','-m','Co-authored-by: Codex <noreply@openai.com>'],cwd=ROOT,check=True,capture_output=True)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip()
    subprocess.run(['git','push','origin','main'],cwd=ROOT,check=True)
    foreign_unchanged();write(RUN/'publication.json',dict(at=now().isoformat(),commit=commit,push_completed=True,paths=names))
    print(commit)


if __name__=='__main__':main()
