"""Run required checks with verified CPU visibility and explicit return codes."""
from pathlib import Path
import os,subprocess,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_candidate_reserve_20261021 import write,now
W=ROOT/'runs/z13_weekend_20261024'


def main():
    rows={}
    for name,args in [('catalog',['scripts/validate_project_history.py']),('tests',['-m','pytest','tests/test_project_history.py','tests/test_z13_competing_candidates.py','tests/test_z13_competing_execution.py','-q'])]:
        p=subprocess.run([str(ROOT/'.venv/Scripts/python.exe'),*args],cwd=ROOT,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONIOENCODING':'utf-8'},capture_output=True)
        (W/('heartbeat_'+name+'_0442.log')).write_bytes(p.stdout);(W/('heartbeat_'+name+'_0442.stderr.log')).write_bytes(p.stderr);rows[name]=p.returncode;print(name,p.returncode,flush=True)
        assert p.returncode==0,repr(p.stderr[-1500:])
    write(W/'heartbeat_checks_0442.json',dict(at=now().isoformat(),exit_codes=rows))


if __name__=='__main__':main()
