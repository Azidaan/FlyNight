from common import *
import subprocess,time
os.environ['CUDA_VISIBLE_DEVICES']=''
inputs={n:sha(ROOT/n) for n in ['experiments/project_history.json','experiments/system_problems.json','experiments/obsidian_experiment_topics.json','experiments/metrics_glossary.json','experiments/useful_literature.json']}
results=[]
for name,args in [('catalog',[str(ROOT/'scripts/validate_project_history.py')]),('history_tests',['-m','pytest','tests/test_project_history.py','-q','--basetemp='+str(ROOT/'.cache/tmp/visibility_cross_source_20261114_final_tests')])]:
 start=time.monotonic()
 with (OUT/(name+'.log')).open('xb') as log:r=subprocess.run([sys.executable,*args],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
 results.append(dict(name=name,returncode=r.returncode,elapsed_seconds=time.monotonic()-start,log=name+'.log'));assert r.returncode==0,results[-1]
assert all(sha(ROOT/n)==h for n,h in inputs.items())
save('checks.json',dict(passed=True,inputs=inputs,results=results));print('REQUIRED_CHECKS_PASSED',flush=True)
