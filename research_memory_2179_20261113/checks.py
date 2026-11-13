from common import *
import subprocess, time
os.environ['CUDA_VISIBLE_DEVICES']=''
results=[]
for title,args in [('catalog',[str(ROOT/'scripts/validate_project_history.py')]),('history_tests',['-m','pytest','tests/test_project_history.py','-q','--basetemp='+str(ROOT/'.cache/tmp/history2179_tests')])]:
    started=time.monotonic()
    with (OUT/(title+'.log')).open('xb') as log:
        r=subprocess.run([sys.executable,*args],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    results.append(dict(name=title,returncode=r.returncode,elapsed_seconds=time.monotonic()-started,log=title+'.log'))
    assert r.returncode==0,results[-1]
save('checks.json',dict(passed=True,results=results))
print('REQUIRED_CHECKS_PASSED',flush=True)
