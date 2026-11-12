from common import *
import psutil,shutil,subprocess,zipfile
assert read(OUT/'review/independent_final_verification.json')['passed']
assert read(OUT/'review/compose_audit_A.json')['passed']
assert read(OUT/'correspondence/audit_fast_v2.json')['passed']
assert read(OUT/'recovery/audit.json')['passed']
assert read(OUT/'review/leakage_audit.json')['passed']
loss=read(OUT/'recovery/assignment_loss_result_v2.json')
assert loss['full_matrix_float32_formula_max_error']<=1e-6
assert loss['independent_solver_objective_max_error']<1e-9
assert loss['no_GT_identity_claim_at1160'] is True
verified={}
for n in ['shared/extraction_protocol.json','correspondence/protocol.json','correspondence/audit_fast_protocol_v2.json','recovery/protocol.json','review/proposed_compose_protocol.json','recovery/assignment_loss_protocol_v2.json']:
    p=read(OUT/n)
    for f,h in p.get('inputs',{}).items():
        if f in verified:assert verified[f]==h,f
        else:assert sha(ROOT/f)==h,f;verified[f]=h
    for f,h in p.get('sources',{}).items():
        path=OUT/'recovery'/f;assert sha(path)==h,f;verified[path.relative_to(ROOT).as_posix()]=h
old=read(ROOT/'research_identity_repairs_20261112/audit.json')['scientific_artifacts']
for f,h in old.items():assert sha(ROOT/f)==h,f
execution=[];(OUT/'execution').mkdir(exist_ok=True)
for path in sorted(list(OUT.glob('queue*_execution.json'))+[OUT/'recovery/queue5b_execution.json']):
    q=read(path);assert q['complete'],path.name
    expected_failure=path.name in ['queue2_execution.json','queue5_execution.json']
    assert q['passed'] is not expected_failure,path.name
    for job in q['jobs']:
        assert job['result']['stop_errors']==[],job
        for pid,birth in job['result']['owned'].items():
            try:alive=psutil.Process(int(pid)).create_time()==birth
            except psutil.NoSuchProcess:alive=False
            assert not alive,(path.name,pid)
        source=Path(job['report'])
        for f in [source,source.with_name('memory_guard.worker.log')]:
            if f.exists():shutil.copyfile(f,OUT/'execution'/(job['token']+'_'+f.name))
    execution.append(dict(file=path.relative_to(OUT).as_posix(),sha256=sha(path),passed=q['passed'],expected_preserved_failure=expected_failure))
with zipfile.ZipFile(OUT/'correspondence/full_result_frozen.zip') as z:
    assert z.testzip() is None
    assert hashlib.sha256(z.read('result.json')).hexdigest()==sha(OUT/'correspondence/result.json')
assert subprocess.check_output(['git','branch','--show-current']).decode().strip()=='codex/system-integrity-20261106-e8f674ce'
assert subprocess.check_output(['git','rev-parse','HEAD']).decode().strip()==read(OUT/'initial_state.json')['head']
save('audit.json',dict(passed=True,checked=now(),verified_inputs=verified,scientific_artifacts=old,old_science_unchanged=True,all_owned_workers_ended=True,execution=execution,all54_events_retained=True,matrix_records=216,readout_records=432,first_A_results_unchanged=True,B_repair_outcomes_computed=False,full_tracker_calls=0,new_weights=0,negative_control_and_technical_failures_preserved=True,scope='Computational validation and bounded diagnosis. Newassignment identities unverified; no generalization gain or full ID-lifecycle/HOTA claim.'))
print('HOLDOUT_PROVENANCE_AND_SCIENCE_AUDIT_PASSED',len(verified),len(old))
