"""Sequential guarded catalog checks; no scientific computation."""
from common import *
start()
import subprocess
result={}
for name,args in [('catalog',['scripts/validate_project_history.py']),('catalog_tests',['-m','pytest','tests/test_project_history.py','-q'])]:
    with (OUT/(name+'.log')).open('wb') as log:
        p=subprocess.run([sys.executable,*args],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    result[name]=p.returncode
    assert p.returncode==0,name
write('record_checks.json',dict(passed=True,returncodes=result))
print('RECORD_CHECKS_PASS')
