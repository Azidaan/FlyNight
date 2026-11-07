from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.b2_tracker_recovery4_20261029 import code_for
result={}
for kind in ['prepare','gpu','audit','record']:
    code=code_for(kind)[0]
    compile(code,'<prepared-'+kind+'>','exec')
    result[kind]={'gpu_audit_dependency':'b2_training_gpu_predictor_audit' in code,
                  'cpu_loader':'load_cpu_branches' in code,
                  'new_jobs':[x.strip() for x in code.splitlines() if any(s in x for s in ["('146'","('149'","('152'"])]}
assert all(v['gpu_audit_dependency'] for v in result.values())
assert len(result['prepare']['new_jobs'])==3
assert result['audit']['cpu_loader']
path=ROOT/'runs/continuation_20261028/b2_tracker_preparation_recovery4/code_check.json'
path.write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result))
