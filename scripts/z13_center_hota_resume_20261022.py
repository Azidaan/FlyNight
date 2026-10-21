"""Resume unchanged scientific jobs in a separately sealed author window."""
from datetime import datetime
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import z13_center_hota_completion_20261021 as previous
science=previous.science;RUN=science.RUN
AUTH=RUN/'resume1_authorization.json';PREFLIGHT=RUN/'resume1_preflight.json'
SAVE='2026-10-22T07:35:00+03:00';HARD='2026-10-22T07:39:30+03:00'


def require_unchanged_markers(jobs,markers,identity):
    by={j['id']:j for j in jobs};seen=set()
    for item in markers:
        job=item['job'];assert job['id'] not in seen,'Duplicate completion marker'
        assert item['exit_code']==0 and item['identity']==identity and by[job['id']]==job,'Invalid completed job'
        seen.add(job['id'])
    return [j for j in jobs if j['id'] not in seen]


def verify_stitched_frames(frames,matches):
    assert isinstance(frames,list) and len(frames)==len(matches)
    assert all(f['name']==m['reference_name'] and f['frameIndex']==m['reference_index'] for f,m in zip(frames,matches))


def prepare():
    import numpy as np
    from scripts.z13_hota_deadline_guard import WindowsProcesses
    if AUTH.exists() or PREFLIGHT.exists():raise RuntimeError('Existing resume contract; do not overwrite')
    assert science.now()<datetime.fromisoformat(SAVE)
    old=science.read(RUN/'full_controller_result.json');assert old['all_own_workers_ended'] and not old['remaining_owned_processes']
    processes=WindowsProcesses()
    assert all(processes.creation(int(pid))!=born for pid,born in old['known_owned_processes'].items())
    former=science.read(RUN/'morning_save_launch.json')['controller']
    assert processes.creation(former['pid'])!=former['birth']
    identity=science.read(RUN/'execution_identity.json');jobs=science.read(RUN/'jobs.json');markers=[science.read(p) for p in (RUN/'completed_jobs').glob('*.json')]
    pending=require_unchanged_markers(jobs,markers,identity);assert pending==old['remaining'] and len(markers)==old['completed_jobs']==69
    p=science.read(RUN/'protocol.json');previous.guard(p)
    files=set()
    for item in markers:
        j=item['job'];v=j['video'];stage=j['stage']
        if stage=='signatures':
            directory=RUN/j['arm']/'signatures'/v;meta=science.read(directory/'signatures.json')
            with np.load(directory/'signatures.npz') as actual,np.load(ROOT/'runs/fly_signatures'/v/'signatures.npz') as baseline:
                assert int(actual['frame_count'])==meta['frames']==int(baseline['frame_count']) and np.isfinite(actual['outputs']).all()
            files.update((directory/'signatures.npz',directory/'signatures.json'))
        elif stage=='masa':
            for arm in p['arms']:
                with np.load(RUN/arm/'signatures'/v/'signatures.npz') as sig:n=len(sig['track_id'])
                path=RUN/arm/'masa'/(v+'.npz')
                with np.load(path) as z:assert z['first'].shape==z['last'].shape==(n,256) and np.isfinite(z['first']).all() and np.isfinite(z['last']).all()
                files.add(path)
        elif stage=='stitch':
            for arm in p['arms']:
                path=RUN/'eval'/(arm+'_masa')/(v+'.json');frames=science.read(path)
                verify_stitched_frames(frames,science.read(ROOT/'runs/train_cache'/v/'alignment.json')['matches'])
                files.add(path)
        else:raise RuntimeError('Unexpected completed evaluation')
    local=['scripts/z13_center_hota_resume_20261022.py','scripts/z13_record_center_hota_resume_20261022.py','scripts/z13_center_hota_progress_20261022.ps1','tests/test_z13_center_hota_resume.py']
    frozen={x.relative_to(ROOT).as_posix():science.sha(x) for x in sorted(files)}
    science.write(AUTH,dict(created=science.now().isoformat(),authorization='Author22.10: Это потом, продолжи работу на самой диссертацией. Continue existing permitted experiment; no new scientific intervention.',save_at=SAVE,hard_stop=HARD,protocol_sha256=science.sha(RUN/'protocol.json'),execution_identity=identity,previous_result_sha256=science.sha(RUN/'full_controller_result.json'),completed_at_resume=len(markers),remaining_at_resume=len(pending),first_remaining=pending[0],change='Only execution date/window and separate runtime filenames. Original science/settings/models/data/criteria and69completed jobs unchanged. No retraining or repeated completed jobs.',sha256={n:science.sha(ROOT/n) for n in local},saved_sha256=frozen,saved_stats={n:previous.service.fingerprint(ROOT/n) for n in frozen}))
    science.write(PREFLIGHT,dict(at=science.now().isoformat(),authorization_sha256=science.sha(AUTH),full_frozen_scientific_bytes_verified=True,completed_artifacts_verified=len(files),completed_markers=len(markers),pending=len(pending),own_previous_tree_ended=True))
    science.journal('подготовка возобновления полного сравнения22октября','*Постановка.* Продолжить прежнюю z13_center_hota_v1 с69 готовых заданий; остаток107 и первый stitch_019c4fa9-d4219327 проверены по completed_jobs и прежнему результату. Новый служебный срок — сохранение22.10 07:35, конец своего дерева07:39:30 МСК.\n\n*Проверено.* resume1_preflight.json: все байты замороженных научных входов повторно сверены, готовые сигнатуры проверены по полноте кадров/конечности, MASA по размерам/конечности, готовая сшивка прочитана; хеши сохранённых артефактов в resume1_authorization.json. Старые ошибки и итог21октября сохраняются. Меняется только исполнение в новом окне; источники/пороги/научные критерии не редактируются, готовое не повторяется. .gitignore отложен по сообщению автора и в эту публикацию не входит.')
    print('Verified69 completed jobs;107 pending; separate window sealed')


def guard(p):
    auth=science.read(AUTH);assert science.read(PREFLIGHT)['authorization_sha256']==science.sha(AUTH)
    assert science.sha(RUN/'protocol.json')==auth['protocol_sha256']
    assert science.read(RUN/'execution_identity.json')==auth['execution_identity']
    assert science.sha(RUN/'full_controller_result.json')==auth['previous_result_sha256']
    for rel,digest in auth['sha256'].items():assert science.sha(ROOT/rel)==digest,rel
    stage=sys.argv[sys.argv.index('--stage')+1] if '--stage' in sys.argv else None
    previous.service.verify_files(ROOT,auth['saved_sha256'],auth['saved_stats'],full='--full-control' in sys.argv or stage=='eval')
    return previous.guard(p)


def stage_check():
    deadline=datetime.fromisoformat(science.read(AUTH)['hard_stop'])
    if time.monotonic()-science.T0>science.LIMIT or (RUN/'resume1.stop').exists() or science.now()>=deadline:raise TimeoutError('Owned resumed stage deadline')


def controller_source(source):
    replacements={"str(ROOT/'scripts/z13_center_hota_20261021.py')":"str(ROOT/'scripts/z13_center_hota_resume_20261022.py')", "RUN/'probe_controller_result.json'":"RUN/'probe_completion_controller_result.json'", "p['save_at']":"read(RUN/'resume1_authorization.json')['save_at']", "p['hard_stop']":"read(RUN/'resume1_authorization.json')['hard_stop']", "'--attempt','full'":"'--attempt','resume1'", "write(RUN/'jobs.json',jobs)":"assert read(RUN/'jobs.json')==jobs"}
    for old,new in replacements.items():assert old in source,old;source=source.replace(old,new)
    for old,new in [('full_controller_result.json','resume1_controller_result.json'),('full_controller.json','resume1_controller.json'),('active_job.json','resume1_active_job.json'),('queue_status.json','resume1_queue_status.json'),('save_checkpoint.json','resume1_save_checkpoint.json'),('job_logs','resume1_job_logs')]:
        assert old in source,old;source=source.replace(old,new)
    return source


def main():
    if '--prepare-resume' in sys.argv:prepare();return
    import tracking.z13_ridge_localizer as original
    from tracking.z13_ridge_localizer_v2 import RecoveredCenterRidgeLocalizer
    original.CenterRidgeLocalizer=RecoveredCenterRidgeLocalizer;science.verify=guard;science.check=stage_check
    if '--full-control' in sys.argv:
        source=controller_source((ROOT/'scripts/z13_center_hota_controller_20261021.py').read_text(encoding='utf-8'))
        scope=dict(__name__='resumed_center_hota_controller',__file__=str(ROOT/'scripts/z13_center_hota_controller_20261021.py'));exec(compile(source,'resumed_center_hota_controller','exec'),scope);scope['main']();return
    if '--stage' not in sys.argv:raise RuntimeError('Resume only explicit pending scientific stage; no repeated probe')
    science.main()


if __name__=='__main__':main()
