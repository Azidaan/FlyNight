from common import *
import shutil
# One sequentially checked batch for three related read-only audits; each retains its preregistered question and evidence.
members=['research_ownership_conflict_2179_20261114','research_memory_gap_transfer_20261114','research_gap_appearance_recovery_20261114','research_visible_support_recovery_20261114']
for name in members:
 d=ROOT/name;assert read(d/'audit.json')['passed'] and read(d/('queue_run_v2_execution.json' if (d/'queue_run_v2_execution.json').exists() else 'queue_run_execution.json'))['passed']
orig=(OUT/'publish.py').read_text(encoding='utf8')
start=orig.index("identity='ownership_conflict_2179_20261114'");end=orig.index("put('graph_science_color_verification.json'",start)
orig=orig[:start]+'''for identity in ['ownership_conflict_2179_20261114','memory_gap_transfer_20261114','gap_appearance_recovery_20261114','visible_support_recovery_20261114']:
    path=ROOT/('research_'+identity);expected='выполнено' if read(path/'result.json')['scientific_passed'] else 'без-положительного-итога';node=color_manifest['nodes']['exp:'+identity];assert node['color']==expected;color_checks[identity]=dict(path=node['path'],color=expected)
'''+orig[end:]
mark='# Archive every original selected byte before standard publication formatting.'
frag='''# Include explicit files for the other two audit cards, never archive a dataset or external cache.
for dirname in ['research_memory_gap_transfer_20261114','research_gap_appearance_recovery_20261114','research_visible_support_recovery_20261114']:
    based=ROOT/dirname
    names=['common.py','official.py','run_queue.py','register.py','run.py','assignment.py','evaluate.py','record.py','protocol.json','queue_run.json','queue_run_execution.json','audit.json','result.json','actions.json','vectors.npz','РЕЗУЛЬТАТ.md','journal_additions.txt','shared_checks.json']
    for filename in names:
        item=based/filename
        if item.is_file():selected[item.relative_to(ROOT).as_posix()]=str(item)
'''
orig=orig.replace('or D in source.parents:',"or D in source.parents or any((ROOT/k) in source.parents for k in ['research_memory_gap_transfer_20261114','research_gap_appearance_recovery_20261114','research_visible_support_recovery_20261114']):")
orig=orig.replace(mark,frag+'\n'+mark)
target=OUT/'publish_batch.py';target.write_text(orig,encoding='utf8')
for name in members:
 d=ROOT/name
 if d!=OUT:
  (d/'shared_checks.json').write_text(json.dumps(dict(validation=rel(OUT/'checks.json'),graph_verification=rel(OUT/'graph_science_color_verification.json'),scope='Single verification of the complete three-card batch after all records were written.'),indent=2)+'\n',encoding='utf8')
save('queue_batch_publication.json',dict(runner_sha256=sha(OUT/'run_queue.py'),deadline='2026-11-14T08:55:00+03:00',inputs={rel(target):sha(target)},jobs=[dict(entry=rel(target),args=['prepare'],timeout=650)]))
