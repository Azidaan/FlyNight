import os,sys,subprocess
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_identity_assignment_run_20261108.common import *


def main(obs):
    checks=[]
    commands=[('unit_recovery1',['-m','pytest','tests/test_spatial_identity_assignment.py','-q']),
              ('catalog',['scripts/validate_project_history.py']),
              ('catalog_tests',['-m','pytest','tests/test_project_history.py','-q'])]
    for name,args in commands:
        status('technical_gate',check=name,guard=obs.read())
        log=RUN/(name+'.log')
        with log.open('x',encoding='utf-8') as f:r=subprocess.run([sys.executable,*args],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
        checks.append(dict(name=name,returncode=r.returncode,sha256=sha(log)))
        write(RUN/'technical_checks_recovery1.json',dict(passed=all(x['returncode']==0 for x in checks),checks=checks))
        if r.returncode:raise RuntimeError((name,r.returncode))
    from research_identity_assignment_run_20261108.zero_parity import main as zero
    import research_identity_assignment_run_20261108.zero_parity as module
    module.verify_phase=lambda name:verify_phase('zero_phase_recovery1.json')
    zero(obs)


if __name__=='__main__':main(guard())
