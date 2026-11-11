from common import *
guard();verify()
checks={}
for name,args in [('catalog',['scripts/validate_project_history.py']),('tests',['-m','pytest','tests/test_project_history.py','-q'])]:
    with (OUT/(name+'.log')).open('w',encoding='utf8') as f:
        p=subprocess.run([sys.executable,*args],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
    checks[name]=dict(returncode=p.returncode,log=name+'.log',sha256=sha(OUT/(name+'.log')))
    assert p.returncode==0,checks[name]
verify();save('validation.json',dict(passed=True,checks=checks))
print('VALIDATED')
