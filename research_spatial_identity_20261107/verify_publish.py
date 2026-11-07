"""Sequential record validation and copy-first graph update under memory guard."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys
import subprocess
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research_spatial_identity_20261107.run_common import RUN,read,sha,write,status,guard,verified_protocol


def main(obs):
    began=time.perf_counter();verified_protocol();results=[]
    def run(name,args):
        status('publication_checks',check=name,guard=obs.read())
        path=RUN/(name+'.log')
        with path.open('x',encoding='utf-8') as f:
            p=subprocess.run([sys.executable,*args],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
        results.append(dict(check=name,returncode=p.returncode,log=path.relative_to(ROOT).as_posix(),sha256=sha(path)))
        assert p.returncode==0,(name,p.returncode)
    run('catalog_validation',['scripts/validate_project_history.py'])
    run('catalog_tests',['-m','pytest','tests/test_project_history.py','-q'])
    protected=read(RUN/'graph_protected_before_recovery1.json')
    assert len(protected)==25
    run('graph_copy',['scripts/build_obsidian_vault.py','--vault',str(RUN/'vault_check')])
    for name,h in protected.items():assert sha(RUN/'vault_check'/name)==h,name
    run('graph_main',['scripts/build_obsidian_vault.py'])
    for name,h in protected.items():assert sha(ROOT/'ObsidianFlyVis'/name)==h,name
    before=read(RUN/'graph_all_files_before.json')
    assert all((ROOT/'ObsidianFlyVis'/p).exists() for p in before)
    boundary=read(RUN/'working_tree_boundary.json')
    assert sha(ROOT/'dissertation/План целей по Трекеру.md')==boundary['preexisting_tracker_plan_sha256']
    verified_protocol()
    write(RUN/'publication_checks.json',dict(passed=True,checks=results,protected_hypothesis_files=25,
        all_preexisting_vault_files_retained=len(before),preexisting_tracker_plan_preserved=True,
        seconds=time.perf_counter()-began),new=True)
    status('publication_checks_completed',passed=True)


if __name__=='__main__':main(guard())
