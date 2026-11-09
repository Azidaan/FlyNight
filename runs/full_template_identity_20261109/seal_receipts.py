"""Seal every ended stage, including three metadata failures and recoveries."""
from pathlib import Path
import hashlib,json,shutil
from datetime import datetime,timezone,timedelta
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent;BASE=ROOT/'runs/flyvis_repairs_20261106/unification_20261107/uncapped_checks'
stages=[('register','2171483b1bd64089b954d5225940e8c2',0),('math_geometry','cb597691c5bb4e2cb2fa63d7b7a43b0e',0),
    ('comparison_register_journal_failed','e8c8ce7f8912483db9298abd2c5cf6cd',1),('journal_recovery','51e1642d13754ea087b1f2b979a1c522',0),
    ('full_features','09ee37805f0d48f0a5ad0f7b3908c4cd',0),('full_augmented_G_fit','88bec8bfaf7b4fbc94703fe38ca32ec9',0),('full_audit','b0931559c0594cba814acdd53262eeca',0),
    ('record_cache_assumption_failed','c4514243362f4113a7f73e8d457bffc9',1),('records_recovered','42e669e4060342fabcf26d3e58039298',0),('phase2_checks','ac1701d4f2d441ae80d241cb57877e45',0),
    ('original_geometry_register','4fa7859305f44b2fb7801091be0351d9',0),('original_geometry_fit','7200fe982f014dce8adb9c02b26f1d8a',0),('original_geometry_audit','cda221a1d40c42b3a828ddc1f3a6a0da',0),('original_geometry_record','c6979440473c420aaf0c551adccde0c3',0),
    ('polish_journal_import_failed','4d305f35de194afea57679fb1f3239eb',1),('polish_journal_finished','8e830e87aebd4df0b7e837ad887ea37a',0),('final_checks','8bfbeb0d2cba4aa89839ee12cbb6cda0',0)]
files=[]
for label,token,expected in stages:
    source=BASE/token;r=json.loads((source/'memory_guard.json').read_text(encoding='utf8'));assert r['returncode']==expected and not r['stop_errors'],label
    dest=OUT/'closed_guards'/label;dest.mkdir(parents=True,exist_ok=True)
    for name in ['memory_guard.json','memory_guard.worker.log','supervisor.json']:
        path=source/name
        if path.exists():
            target=dest/name;shutil.copyfile(path,target);files.append(dict(stage=label,returncode=expected,source=path.relative_to(ROOT).as_posix(),copy=target.relative_to(ROOT).as_posix(),sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
old=OUT/'status.json';archive=OUT/'status_phase2.json';assert not archive.exists();shutil.copyfile(old,archive)
old.write_bytes(json.dumps(dict(stage='signal_diagnostics_complete',diagnostics=3,admitted='C34_full7_channels',actual_ID_pilot_not_run=True,proposal='research_full_template_native_pilot_20261109/proposal_v2.json',updated=datetime.now(timezone(timedelta(hours=3))).isoformat())).encode('utf8'))
report=dict(passed=True,stages=len(stages),files=files,all_scientific_stages_zero=True,retained_metadata_failures=3,actual_ID_pilot_unrun=True,
    recovery_notes=['Saved comparison protocol remained valid; only failed journal write was recovered.','Large processed frames absent despite markers; proposal corrected to reuse cached maps and sparse checkpoint/video recovery. No science repeated.','Missing journal helper import after metadata save; completed only missing journal append.'],
    failed_record_source_archive=dict(path='runs/full_template_identity_20261109/record_before_missing_cache_check.py',sha256=hashlib.sha256((OUT/'record_before_missing_cache_check.py').read_bytes()).hexdigest()),
    failed_polish_source_archive=dict(path='runs/full_template_identity_20261109/polish_current_before_import_fix.py',sha256=hashlib.sha256((OUT/'polish_current_before_import_fix.py').read_bytes()).hexdigest()))
(OUT/'closed_receipts.json').write_bytes((json.dumps(report,ensure_ascii=False,indent=2)+'\n').encode('utf8'))
print('Sealed',len(stages),'ended stages; final candidateC34, ID pilot unrun')
