from common import *
import subprocess,time
os.environ['CUDA_VISIBLE_DEVICES']=''
inputs={n:sha(ROOT/n) for n in ['experiments/project_history.json','experiments/system_problems.json','experiments/hypothesis_calendar.json','experiments/deferred_identity_hypotheses.json','scripts/obsidian_level_graph.py','scripts/obsidian_deferred_identity_hypotheses.py']}
results=[]
for name,args in [('catalog_v2',[str(ROOT/'scripts/validate_project_history.py')]),('history_tests_v2',['-m','pytest','tests/test_project_history.py','-q','--basetemp='+str(ROOT/'.cache/tmp/visible2179_final_tests')])]:
    start=time.monotonic()
    with (OUT/(name+'.log')).open('xb') as log:r=subprocess.run([sys.executable,*args],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    results.append(dict(name=name,returncode=r.returncode,elapsed_seconds=time.monotonic()-start,log=name+'.log'))
    assert r.returncode==0,results[-1]
assert all(sha(ROOT/n)==h for n,h in inputs.items())
save('checks_v2.json',dict(passed=True,inputs=inputs,results=results))
print('REQUIRED_CHECKS_V2_PASSED',flush=True)
