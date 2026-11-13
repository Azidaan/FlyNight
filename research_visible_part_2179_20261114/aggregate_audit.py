from common import *
import subprocess
checks={}
for file in ['appearance_audit/pixel_audit.json','appearance_audit/level_audit.json','assignment_audit/independent_audit.json','mask_review/finalreview.json','summary.json']:
    value=read(OUT/file);assert value['passed'],file
    checks[file]={'passed':True,'sha256':sha(OUT/file)}
sources={}
for name in ['protocol.json','pixel_queue.json','level_queue.json']:
    for path,expected in read(OUT/name).get('inputs',{}).items():
        actual=sha(ROOT/path);assert actual==expected,(path,actual,expected)
        sources[path]=actual
for name in ['appearance_audit/pixel_audit.json','appearance_audit/level_audit.json']:
    for path,expected in read(OUT/name)['inputs'].items():
        assert sha(path)==expected,path
queues={}
for name in ['pixel_queue_execution.json','level_queue_execution.json','mask_review/mask_queue_execution.json']:
    value=read(OUT/name);assert value['passed'] and value['complete'],name
    queues[name]={'passed':True,'sha256':sha(OUT/name)}
summary=read(OUT/'summary.json');assert summary['production_changes']==summary['core_calls']==summary['tracker_calls']==0
save('audit.json',dict(passed=True,audited=now(),head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),independent_evaluations=checks,unchanged_registered_inputs=sources,finished_queues=queues,scope='Pixel intervention and conditional level selection on one known event. Diagnostic visible cores are manual; no automatic segmentation or identity lifecycle claim.'))
print('AGGREGATE_AUDIT_PASSED',flush=True)
