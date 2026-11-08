"""Prepare an explicit owned-file review; do not mutate Git, foreign files or data."""
import sys,re,subprocess,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_identity_assignment_run_20261108.common import *
RUN=ROOT/'runs/identity_evidence_20261108/publication'
SOURCE_PATTERNS=['research_identity_assignment*_20261108','research_singleton_*20261108','research_native_confirm_*20261108','research_bbox_*20261108','research_identity_evidence_20261108']
TRACKING=['spatial_identity_assignment','free_singleton_assignment','own_template_verification','native_confirm_identity','bbox_singleton_gate','box_normalized_correspondence']
TESTS=['spatial_identity_assignment','free_singleton_assignment','own_template_verification','native_confirm_identity','bbox_singleton_gate','native_confirm_small_cost','box_normalized_correspondence']
PRIMARY={'protocol.json','execution_protocol.json','run_protocol.json','zero_parity.json','restored_zero_parity.json','input_complete.json','features_complete.json','comparison_report.json','independent_audit.json','selection.json','scores_complete.json','queue_result.json','full_skipped.json','inventory.json','inventory_recovery1.json','mechanism_report.json','quality_protocol.json','quality_report.json','report.json','diagnosis.json','readiness.json','label_readiness.json','label_readiness_protocol.json','precision_probe.json','completed_previous_input_audit.json','reconstructed_binary_check.json','views_complete.json','technical_gate.json','first_new_feature_check.json','additional_report.json','metrics_complete.json','rollouts_complete.json','native_trace_complete.json','all_files_before.json','preview_audit.json','result.json','system_problems_before.json','system_problems_before_normalized.json','system_problem_update_receipt.json','normalized_problem_update_receipt.json','normalized_full_record_receipt.json','alignment_diagnostic.json','alignment_diagnostic_protocol.json','return_count_diagnostic.json','return_count_diagnostic_protocol.json','image_geometry_addendum.json','image_geometry_comparison.json','author_window_20261108.json','focus_order_20261108.json','current_input_dependency.json','dependency_sources_verified.json','age_units_clarification.json','parent_full_result_review.json','comparison_preflight_correction.json','verification.json','catalog_check.json','data_complete.json','data_stage_complete.json'}


def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT)


def main():
    assert not git('diff','--cached','--name-only'),'Preserve an already staged index; do not stage over it.'
    assert sha(ROOT/'dissertation/План целей по Трекеру.md')=='09063ca06264ad0cfeab777af6ae827053f1c2b8abd047d7e0dc45c4ede02577'
    full=ROOT/'runs/bbox_normalized_signals_20261108/full';assert read(full/'queue_result.json')['passed'];assert read(ROOT/'runs/identity_evidence_20261108/graph_update/recovery1/queue_result.json')['passed'];assert read(ROOT/'runs/identity_evidence_20261108/graph_update/recovery1/result.json')['passed']
    sources=set();roots=[]
    for pattern in SOURCE_PATTERNS:
        for folder in sorted(ROOT.glob(pattern)):
            if not folder.is_dir():continue
            roots.append(folder.relative_to(ROOT).as_posix())
            for path in folder.glob('*'):
                if path.is_file() and path.suffix in ['.py','.md','.json','.sha256','.txt']:sources.add(path.relative_to(ROOT).as_posix())
    sources.add('research_bbox_fit_readout_readiness_20261108.py')
    for n in TRACKING:sources.add('tracking/'+n+'.py')
    for n in TESTS:sources.add('tests/test_'+n+'.py')
    sources.add('scripts/obsidian_fundamental_overview.py')
    files=set(sources);files.update(['experiments/project_history.json','experiments/metrics_glossary.json','experiments/system_problems.json'])
    changed=git('-c','core.quotePath=false','diff','--name-only','-z').decode('utf-8').split('\0')
    for n in changed:
        if n.startswith('ObsidianFlyVis/'):files.add(n)
    run_names=['identity_assignment_20261108','singleton_free_20261108','singleton_verify_20261108','singleton_age_20261108','singleton_error_audit_20261108','singleton_admission_20261108','singleton_native_confirm_20261108','native_confirm_policy_20261108','native_confirm_risk_20261108','bbox_confirmation_inventory_20261108','bbox_singleton_pilot_20261108','bbox_singleton_audit_recovery1_20261108','bbox_cost_pilot_20261108','bbox_spatial_readiness_20261108','bbox_spatial_signals_20261108','bbox_normalized_signals_20261108','bbox_fit_readout_readiness_20261108','bbox_fit_maps_20261108','identity_evidence_20261108']
    run_names += ['bbox_readout_calibration_20261108']
    PRIMARY.update(['execution_inputs.json','fit_diagnosis_record_receipt.json','schema_passed.json','window_close.json','recovery_protocol.json','publication_receipt.json'])
    skipped=[]
    for name in run_names:
        folder=ROOT/'runs'/name
        for path in folder.rglob('*'):
            if not path.is_file():continue
            relative=path.relative_to(folder);parts=relative.parts
            if any('temp' in p or 'preview' in p or 'owned_backup' in p or 'publication' in p or p in ['reuse_pool','retinal','processed','features','native','native_trace','outcomes','rollouts','outputs'] for p in parts[:-1]):continue
            selected=path.name in PRIMARY or path.name.endswith('_phase.json') or path.name.endswith('.ended.json') or path.name.startswith('system_problems_before') and path.suffix=='.json'
            if name in ['bbox_normalized_signals_20261108','native_confirm_risk_20261108','singleton_verify_20261108','singleton_age_20261108'] and 'metrics' in parts:selected=path.suffix=='.json' or path.name.endswith('.details.json.gz')
            if selected:
                if path.stat().st_size<=5*1024**2:files.add(path.relative_to(ROOT).as_posix())
                else:skipped.append(dict(path=path.relative_to(ROOT).as_posix(),reason='Large primary metadata retained locally',bytes=path.stat().st_size))
    # Preserve actual source bytes separately from Git line-ending normalization.
    RUN.mkdir(parents=True,exist_ok=True);archive=RUN/'author_sources.zip'
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for n in sorted(sources):z.writestr(n,(ROOT/n).read_bytes())
    files.add(archive.relative_to(ROOT).as_posix())
    plan='dissertation/ЕДИНЫЙ_ПЛАН.md';diff=git('diff','--unified=0','--',plan);chunks=re.split(rb'(?=^@@ )',diff,flags=re.M);header=chunks[0];own=[];foreign=[]
    for chunk in chunks[1:]:
        if b'+**07.11.2026, 20:43 MSK**' in chunk:foreign.append(chunk);continue
        assert b'+**08.11.2026,' in chunk and not any(line.startswith(b'-') for line in chunk.splitlines()[1:]),'Unexpected journal edits require review.';own.append(chunk)
    assert own and len(foreign)==1,'Do not absorb an unrecognized foreign journal change.'
    patch=RUN/'owned_plan.patch';patch.write_bytes(header+b''.join(own));original=git('show','HEAD:'+plan).splitlines(keepends=True);expected=list(original)
    for chunk in reversed(own):
        m=re.match(rb'@@ -(\d+),0 \+\d+,\d+ @@',chunk);assert m;lines=chunk.splitlines(keepends=True)[1:];assert all(x.startswith(b'+') for x in lines);expected[int(m[1]):int(m[1])]=[x[1:] for x in lines]
    (RUN/'expected_index_plan.md').write_bytes(b''.join(expected));write(RUN/'source_manifest.json',dict(created=datetime.now(MSK).isoformat(),sources={n:sha(ROOT/n) for n in sorted(sources)},archive_sha256=sha(archive),scope='Author code only; no environment, weights, movies, activation caches or foreign goal-plan edits. Run sourceHEADremains7f702f5withdirtysourceSHA; a later archival commit is not the original run commit.'),new=True)
    files.add((RUN/'source_manifest.json').relative_to(ROOT).as_posix());files.add(patch.relative_to(ROOT).as_posix())
    files=sorted(files);assert all((ROOT/n).is_file() for n in files);assert all(not n.endswith(('.pt','.npy','.npz','.mov','.mp4')) for n in files);assert sum((ROOT/n).stat().st_size for n in files)<60*1024**2
    (RUN/'pathspec.nul').write_bytes(b''.join(n.encode('utf-8')+b'\0' for n in files));review=dict(prepared=datetime.now(MSK).isoformat(),base_head=git('rev-parse','HEAD').decode().strip(),source_roots=roots,files={n:dict(sha256=sha(ROOT/n),bytes=(ROOT/n).stat().st_size) for n in files},bytes=sum((ROOT/n).stat().st_size for n in files),plan_patch_sha256=sha(patch),expected_index_plan_sha256=sha(RUN/'expected_index_plan.md'),working_plan_sha256=sha(ROOT/plan),foreign_hunks_preserved=1,excluded_foreign_goal_plan_sha256=sha(ROOT/'dissertation/План целей по Трекеру.md'),skipped=skipped)
    write(RUN/'review.json',review,new=True);print(dict(files=len(files),bytes=review['bytes'],source_roots=roots,skipped=skipped))


if __name__=='__main__':main()
