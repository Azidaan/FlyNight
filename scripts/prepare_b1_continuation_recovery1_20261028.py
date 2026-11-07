"""Отдельное техническое восстановление зависшего вывода, прежние сроки."""
from pathlib import Path
from datetime import datetime
import sys, time, copy, shutil, hashlib
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.recovery_research_service_20261027 import RUN, read, write, sha, matching, now


def main():
    old = RUN / 'b1_tracker/continuation1'; folder = RUN / 'b1_tracker/continuation_recovery1'
    assert not folder.exists()
    old_audit = RUN / 'b1_tracker_audit/continuation1'
    for source in [old, old_audit]:
        c = read(source / 'controller_result.json')
        if not c['all_owned_ended']:
            late = read(source / 'late_owned_end_verified.json')
            assert late['passed'] and late['actual_all_owned_ended'] and late['original_controller_sha256'] == sha(source / 'controller_result.json')
        assert not any(matching(k, v) for k, v in c['owned'].items())
    assert read(old / 'hang_evidence.json')['last_status_age_seconds'] > 1200
    old_deadline = read(old / 'execution_deadline.json')['deadline']
    remaining = int(datetime.fromisoformat(old_deadline).timestamp() - time.time() - 15)
    assert remaining > 600
    audit_time = (datetime.fromisoformat(read(old_audit / 'controller_result.json')['updated']) - datetime.fromisoformat(read(old_audit / 'execution_deadline.json')['started'])).total_seconds()
    audit_budget = int(4200 - audit_time - 20); assert audit_budget > 300
    p = read(old / 'protocol.json')
    seed = read(old / 'resume_seed.json'); completed = list(seed['completed_videos'])
    assert len(list((old / 'completed').glob('*.json'))) == len(completed) == 1
    new_hashes = {}
    for path, digest in seed['output_sha256'].items():
        source = ROOT / path; assert sha(source) == digest
        relative = source.relative_to(old); target = folder / relative
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source, target)
        assert sha(target) == digest
        new_hashes[target.relative_to(ROOT).as_posix()] = digest
    for v in completed:
        write(folder / 'completed' / (v + '.json'), dict(read(old / 'completed' / (v + '.json')),
            reused_without_inference=True, source=(old / 'completed' / (v + '.json')).relative_to(ROOT).as_posix()))
    seed['output_sha256'] = new_hashes
    write(folder / 'resume_seed.json', seed); write(folder / 'per_video.json', seed['per_video'])
    shutil.copy2(old / 'author_override.json', folder / 'author_override.json')
    write(folder / 'price.json', read(old / 'price.json'))
    from scripts.run_b1_tracker_continuation1_20261028 import continuation_source
    code = continuation_source().replace('if frame % 25 == 0:', 'if frame % 5 == 0:')
    audit_code = (ROOT / 'scripts/audit_b1_tracker_20261028.py').read_text(encoding='utf-8')
    token = "source = RUN / 'b1_tracker'; folder = RUN / 'b1_tracker_audit'"; assert audit_code.count(token) == 1
    audit_code = audit_code.replace(token, "source = RUN / 'b1_tracker/continuation_recovery1'; folder = RUN / 'b1_tracker_audit/continuation_recovery1'")
    extra = ['tracking/b1_cached_eye.py', 'tests/test_b1_cached_eye.py',
        'scripts/run_b1_tracker_continuation_recovery1_20261028.py',
        'scripts/audit_b1_tracker_continuation_recovery1_20261028.py',
        'scripts/prepare_b1_continuation_recovery1_20261028.py',
        (old / 'hang_evidence.json').relative_to(ROOT).as_posix(),
        (old / 'late_owned_end_verified.json').relative_to(ROOT).as_posix(),
        'scripts/recovery_queue_late_exit_20261028.py', 'tests/test_b1_late_owned_end.py',
        (old / 'controller_result.json').relative_to(ROOT).as_posix(),
        (old_audit / 'controller_result.json').relative_to(ROOT).as_posix(),
        (folder / 'resume_seed.json').relative_to(ROOT).as_posix()]
    p.update(created=now(), budget_seconds=remaining, audit_budget_seconds=audit_budget,
        original_continuation_gpu_deadline=old_deadline, executed_worker_sha256=hashlib.sha256(code.encode()).hexdigest(),
        executed_audit_sha256=hashlib.sha256(audit_code.encode()).hexdigest(),
        technical_recovery='Observed >27min without frames; repeated py-spy at BoxEye.hex_render eye.py166 host-to-CUDA torch.tensor shape constructor, CPU busy/GPU almost idle. Only tiny constructor/index geometry is cached; same convolution, same integer indices, same gather/reshape. Exact official CPU sampler parity passed and startup GPU parity required. No scientific change or weakened check.',
        watchdog='faulthandler dump/exit600s without real frame/stage status; updates every5frames. Cannot silently retain dead process as progress.',
        recovery_budget=dict(gpu_absolute_deadline_unchanged=old_deadline, gpu_remaining_seconds=remaining,
            old_audit_spent_seconds=audit_time, audit_remaining_seconds=audit_budget,
            author_save_at='2026-10-28T07:30:00+03:00', author_hard_stop='2026-10-28T07:40:00+03:00'))
    p['sha256'].update({path: sha(ROOT / path) for path in extra})
    write(folder / 'protocol.json', p)
    target_audit = RUN / 'b1_tracker_audit/continuation_recovery1'
    write(target_audit / 'protocol.json', dict(budget_seconds=audit_budget, parent_experiment='b1_full_tracker_pilot_v1',
        source='b1_tracker/continuation_recovery1', original_gpu_deadline=old_deadline,
        same_global_deadline=True, previous_failed_dependency_audit_spent_seconds=audit_time))
    for name, f, script in [('350_b1_cached_eye_recovery1', folder, 'scripts/run_b1_tracker_continuation_recovery1_20261028.py'),
                            ('360_b1_cached_eye_audit', target_audit, 'scripts/audit_b1_tracker_continuation_recovery1_20261028.py')]:
        target = RUN / 'ready_jobs' / (name + '.json'); assert not target.exists()
        write(target, dict(name=name, folder=f.relative_to(ROOT).as_posix(), script=script,
            script_sha256=sha(ROOT / script), protocol_sha256=sha(f / 'protocol.json')))
    write(RUN / 'b1_active_stage.json', dict(source='b1_tracker/continuation_recovery1', audit='b1_tracker_audit/continuation_recovery1', updated=now()))
    from scripts.b123_safe_journal_20261027 import journal
    journal('b1_full_tracker_pilot_v1', 'Б1: обнаружено зависание вывода, отдельное исполнение с кэшем индексов',
        '*Наблюдение.* continuation1/hang_evidence.json: на втором видео прогресс остался на50кадре более27минут; actualchild24060сPID+birthработалоднимCPUядром, повторныеpy-spyпоказалиBoxEye.hex_render:166 torch.tensor. GPUпочтипростаивал. Это не результаткачества. Старое дерево остановлено черезегоstop.request,старые330/340отказысохранены.\n\n*Исправление.* Новый CachedHexSampler хранит те же целочисленные координаты рецепторов и убирает повторный tiny host→CUDAконструктор из каждого глаза. ОфициальныйCPUhex_render сравнен побитно, все6тестовpassed; до кадров обязателенGPUgather parity. Добавлен600с watchdogсдампомстека/реальнымотказом, прогресскаждые5кадров. Настройки/веса/контроли неизменны.\n\n*Срок.* b1_tracker/continuation_recovery1/protocol.json: толькоостатокстарогопродолженногоGPUокна,тотже06:14:50МСК;CPUаудитиспользует4200сминусужепотраченное,общий07:40непродлён. ПервоеГОТОВОЕвидеопереиспользуетсябезинференса;неполноевтороеначинаетсязаново. Еслизависаниеповторится—разобратьновыйдамп,небесконечнорестартить. Авторскоепоручениепродолжитьвопрекиценесохраняется.')
    print(p['recovery_budget'])


if __name__ == '__main__': main()
