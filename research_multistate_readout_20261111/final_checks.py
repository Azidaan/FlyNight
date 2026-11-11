"""Mandatory metadata checks, copy-first graph generation, protected-tree audit."""
from common import *
start()
import shutil
import subprocess
from scripts.obsidian_fundamental_overview import main
from scripts.audit_obsidian_levels import audit

assert read(OUT/'recording.json')['passed'] and read(OUT/'audit.json')['passed']
verify();result={}
for name,args in [('final_catalog',['scripts/validate_project_history.py']),('final_catalog_tests',['-m','pytest','tests/test_project_history.py','-q'])]:
    with (OUT/(name+'.log')).open('xb') as log:
        p=subprocess.run([sys.executable,*args],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    assert p.returncode==0,name
    result[name]=p.returncode
before=read(OUT/'before.json')
def protected(vault):
    for rel,digest in before['protected_hypothesis_graph'].items():
        assert sha(vault/Path(rel).relative_to('ObsidianFlyVis'))==digest,rel
preview=OUT/'preview';assert not preview.exists()
shutil.copytree(ROOT/'ObsidianFlyVis',preview,ignore=shutil.ignore_patterns('plugins','.trash','workspace.json','workspace-mobile.json'))
for name,vault in [('preview',preview),('live',ROOT/'ObsidianFlyVis')]:
    sys.argv=['build_obsidian_vault.py','--vault',str(vault)]
    assert main()==0
    result[name]=audit(vault);assert result[name]['passed']
    protected(vault)
    nodes=read(vault/'.graph-visible.json')['nodes'];node=nodes['exp:'+read(OUT/'recording.json')['card']]
    assert node['color']=='без-положительного-итога'
    result[name]['new_card_note']=node['path']
verify()
result.update(passed=True,protected_hypothesis_notes=len(before['protected_hypothesis_graph']),
    scientific_summary_sha256=sha(OUT/'summary.json'),scientific_sources_unchanged=True)
write('final_checks.json',result)
print('FINAL_CHECKS_PASS',result)
