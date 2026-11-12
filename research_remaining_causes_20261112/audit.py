"""Final independent provenance/ownership audit; no neural or tracker calls."""
from common import *
import psutil,shutil,subprocess
names=['geometry832/independent_audit.json','geometry832/coordinate_audit.json','geometry832/source_support_audit.json','geometry832/factorial_audit.json','overlap574/audit.json','overlap574/raw_partner_audit.json','overlap574/matched_query_audit.json','independent_review/reid/audit_v2.json','independent_review/reid/low_audit.json']
checks={n:read(OUT/n) for n in names}
assert all(v['passed'] for v in checks.values())
bound={};queues={};guards={};ended=[]
for n in ['batch_protocol.json']+[f'queue{i}.json' for i in range(1,7)]:
 p=read(OUT/n)
 for rel,h in p['inputs'].items():
  assert rel not in bound or bound[rel]==h,(n,rel);bound[rel]=h
for rel,h in bound.items():assert sha(ROOT/rel)==h,rel
for n in ['workers_batch.json']+[f'queue{i}_execution.json' for i in range(1,7)]:
 q=read(OUT/n);assert q['complete'];queues[n]=q
 if n not in ['workers_batch.json','queue1_execution.json','queue4_execution.json']:assert q['passed'],n
 jobs=list(q['jobs'].values()) if isinstance(q['jobs'],dict) else q['jobs']
 for job in jobs:
  result=job['result'];guards[job['token']]=dict(report=job['report'],result=result)
  for pid,birth in result['owned'].items():
   try:alive=psutil.Process(int(pid)).create_time()==birth
   except psutil.NoSuchProcess:alive=False
   assert not alive,(n,pid,birth)
   ended.append(dict(pid=int(pid),birth=birth,ended=True))
for name in ['project_history.json','system_problems.json','obsidian_experiment_topics.json','useful_literature.json']:
 assert (OUT/'before'/name).read_bytes().replace(b'\r\n',b'\n')==subprocess.check_output(['git','show','HEAD:experiments/'+name]),name
r=read(OUT/'independent_review/reid/low_result.json');assert r['max_previous_descriptor_error']==0 and r['max_previous_score_error']==0
assert sum(len(c['candidates']) for c in r['records'])==222
row=next(x for x in r['records'] if x['old_id']==574);ranked=sorted(row['candidates'],key=lambda x:x['rank_all_raw']);assert [x['raw_id'] for x in ranked[:3]]==[35280,35292,35274]
assert all(x['own'] and not x['native_first'] and not x['native_second'] for x in ranked[:2]);assert not ranked[2]['own']
d=OUT/'execution';d.mkdir(exist_ok=False)
for token,entry in guards.items():
 original=Path(entry['report']);assert read(original)==entry['result'];shutil.copyfile(original,d/(token+'.json'));log=original.with_name(original.stem+'.worker.log');shutil.copyfile(log,d/(token+'.log'))
science={}
for folder in ['geometry832','overlap574','independent_review/reid']:
 for f in (OUT/folder).glob('*'):
  if f.is_file() and f.suffix.lower() in ['.py','.json','.npz','.png','.svg','.md']:science[f.relative_to(ROOT).as_posix()]=sha(f)
save('audit.json',dict(passed=True,checked_at=now(),head=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),audits=checks,verified_inputs=bound,scientific_artifacts= science,all_owned_workers_ended=True,ended_processes=ended,guards=guards,failed_attempts_preserved=['workers_batch.json:overlap574','queue1_execution.json:original_ReID_audit','queue4_execution.json:low_v2_batch_parity'],limits='Independent arithmetic/provenance; cached neural maps and annotations shared, no independent neural forward. Exact visible-part correspondence and full tracker not evaluated.'))
print('FINAL_AUDIT_PASSED',len(bound),len(checks),len(guards))
