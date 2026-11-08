"""Resume reporting after a heterogeneous older case lacked component.

The original recovery file and its failure log remain unchanged. The exact-frame
audit, extracted ready scores and completed catalog date correction are reused.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import contextlib,hashlib,json,runpy,shutil,sys,tempfile
from datetime import date,datetime,timedelta,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'runs/flyvis_repairs_20261106/unification_20261107'))
from system_memory_guard import guard_entrypoint
guard_entrypoint(timeout=900)
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
assert not (OUT/'recovery_checks.json').exists()
assert read(OUT/'direct_pair_scores.json')['passed']
card=next(c for c in read(ROOT/'experiments/project_history.json')['experiments'] if c['id']==OUT.name)
assert card['started']==card['done']=='2026-11-09'
assert sum(e['ref']==f'runs/{OUT.name}/direct_pair_scores.json' for e in card['evidence'])==1
r=read(OUT/'results.json');now=datetime.now(timezone(timedelta(hours=3)));actual='2026-11-09'
changes=[dict(field=v['field'],old=v['old'],new=actual) for v in card['revisions'] if v['field'] in ['started','done']]
failed_log=ROOT/'runs/flyvis_repairs_20261106/unification_20261107/uncapped_checks/90cd054c9ada417d942eeb9d0a55d45c/memory_guard.worker.log'
partial_failure=ROOT/'runs/flyvis_repairs_20261106/unification_20261107/uncapped_checks/b1c24b387c8746e1844b29a9ad889fc6/memory_guard.worker.log'
assert "KeyError: 'component'" in partial_failure.read_text(encoding='utf8')
source=(OUT/'recover_checks.py').read_text(encoding='utf8')
continuation=source[source.index("path=ROOT/'experiments/system_problems.json'"):]
assert continuation.count("x['component']")==1
continuation=continuation.replace("x['component']","x.get('component')")
continuation=continuation.replace("old62_tests_passed=True,reference_targets_compared=29","old62_tests_passed=True,recovery_v1_failure_sha256=sha(partial_failure),reference_targets_compared=29")
continuation=continuation.replace("['recover_checks.py','results.json'","['recover_checks.py','recover_checks_v2.py','results.json'")
exec(compile(continuation,str(OUT/'recover_checks.py')+' [resumed v2]','exec'),globals())
