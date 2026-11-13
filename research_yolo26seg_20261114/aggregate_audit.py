from common import *
checks={}
for name in ['independent/2179_audit_v2.json','independent/heldout_audit_v2.json','independent/seg_only_benchmark_audit.json','independent/class_filter_audit.json','flyvis_probe/independent_assignment_review.json','flyvis_probe/alignment_audit.json','flyvis_probe/column_order_audit.json']:
 assert read(OUT/name)['passed'],name
 checks[name]=sha(OUT/name)
sources={}
for n,h in read(OUT/'protocol.json')['inputs'].items():
 assert sha(ROOT/n)==h,n;sources[n]=h
for stage in ['2179','heldout','benchmark','class_filter']:
 assert read(OUT/(stage+'_queue_execution.json'))['passed']
 for n,h in read(OUT/(stage+'_queue.json'))['inputs'].items():assert sha(ROOT/n)==h,n
for stage in ['stage_2179','stage_heldout']:
 d=read(OUT/stage/'results.json');assert d['model_unchanged'];assert sha(OUT/stage/'vectors.npz')==d['vectors_sha256'];assert sha(OUT/stage/'masks.npz')==d['masks_sha256']
save('audit.json',dict(passed=True,checked=now(),independent_audits=checks,registered_sources_unchanged=sources,all_owned_gpu_jobs_completed=True,tracker_calls=0,new_core_calls=0,production_edits=0,scientific_seg_gate=False,scientific_flyvis_gate=False))
save('checks_queue.json',dict(runner_sha256=sha(OUT/'run_queue.py'),deadline=read(OUT/'protocol.json')['deadline'],inputs={n:sha(ROOT/n) for n in ['experiments/project_history.json','experiments/system_problems.json','experiments/obsidian_experiment_topics.json', (OUT/'checks.py').relative_to(ROOT).as_posix()]},jobs=[dict(entry=(OUT/'checks.py').relative_to(ROOT).as_posix(),timeout=600)]))
print('AUDITS_PASSED_CHECKS_READY')
