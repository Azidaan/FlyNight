"""Исправить только единичную ось живого глаза в остатке исходного270окна."""
from pathlib import Path
from datetime import datetime
import sys, time, hashlib
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.recovery_research_service_20261027 import RUN, read, write, sha, matching, now


def main():
    original = RUN / 'b1_tracker'
    old_end = datetime.fromisoformat(read(original / 'execution_deadline.json')['deadline']).timestamp()
    for name in ['b1_tracker/recovery1', 'b1_tracker_audit/recovery1', 'b1_live_shape_diagnostic']:
        c = read(RUN / name / 'controller_result.json')
        assert c['all_owned_ended'] and not any(matching(k, v) for k, v in c['owned'].items())
    assert read(original / 'recovery1/incremental_regression.json')['passed']
    folder = original / 'recovery2'; assert not folder.exists()
    remaining = int(old_end - time.time()); audit_budget = 1800; budget = remaining - audit_budget - 60
    assert budget > 600
    p = read(original / 'recovery1/protocol.json')
    old = "source = RUN / 'b1_tracker'; folder = RUN / 'b1_tracker_audit'"
    code = (ROOT / 'scripts/audit_b1_tracker_20261028.py').read_text(encoding='utf-8')
    assert code.count(old) == 1
    code = code.replace(old, "source = RUN / 'b1_tracker/recovery2'; folder = RUN / 'b1_tracker_audit/recovery2'")
    files = ['tracking/b1_eye_layout.py', 'tests/test_b1_eye_layout.py',
        'scripts/run_b1_tracker_recovery2_20261028.py', 'scripts/audit_b1_tracker_recovery2_20261028.py',
        'scripts/prepare_b1_tracker_recovery2_20261028.py',
        'runs/recovery_research_20261027/b1_tracker/recovery1/incremental_regression.json',
        'runs/recovery_research_20261027/b1_tracker/recovery1/controller_result.json',
        'runs/recovery_research_20261027/b1_tracker_audit/recovery1/controller_result.json',
        'runs/recovery_research_20261027/b1_live_shape_diagnostic/controller_result.json']
    p.update(created=now(), budget_seconds=budget, executed_audit_sha256=hashlib.sha256(code.encode()).hexdigest(),
        layout_root_cause='BoxEye возвращает B,frames,1,N; старый EyeBank._render[:,0] оставляет B,1,N. Live after сохранял1,N вместоN. ДляR1 current[:,None] создавал лишнююось, reference становилась4D. В файлах fr_hex ужеT,N. Это исправляется только reshape B,1,1,N→B,1,N передсуществующим[:,0], значениянеизменны; многоканальныйвводотвергается.',
        layout_evidence='tools/flyvis-runtime/flyvis/datasets/rendering/eye.py __call__ возвращает reshape(samples,frames,1,-1); tests/test_b1_eye_layout.py воспроизводит ошибочную R1форму и правильную1,34,721 наCPU.7tests passed. GPUрегрессия C/D/R1/B вrecovery1прошлаТОЧНО и переиспользуется, не повторяется.',
        recovery_budget=dict(original270_seconds=10800, remaining_at_preparation=remaining, gpu_seconds=budget,
            new_version_audit_seconds=audit_budget, reserve_seconds=60,
            note='GPUи аудит новойверсии внутри остатка исходного270абсолютного окна; старые270/275/285/290отказы/сроки сохранены.'),
        original_exact_assertions_unchanged=True)
    p['sha256'].update({f: sha(ROOT / f) for f in files})
    write(folder / 'protocol.json', p)
    audit = RUN / 'b1_tracker_audit/recovery2'
    write(audit / 'protocol.json', dict(parent_experiment='b1_full_tracker_pilot_v1', budget_seconds=audit_budget,
        original_absolute_deadline_unix=old_end, source='b1_tracker/recovery2'))
    write(original / 'recovery2_allocation.json', dict(**p['recovery_budget'], created=now(), original_absolute_deadline_unix=old_end))
    for name, f, script in [('295_b1_tracker_layout_recovery2', folder, 'scripts/run_b1_tracker_recovery2_20261028.py'),
                            ('300_b1_tracker_layout_audit', audit, 'scripts/audit_b1_tracker_recovery2_20261028.py')]:
        target = RUN / 'ready_jobs' / (name + '.json'); assert not target.exists()
        write(target, dict(name=name, folder=f.relative_to(ROOT).as_posix(), script=script,
            script_sha256=sha(ROOT / script), protocol_sha256=sha(f / 'protocol.json')))
    write(RUN / 'b1_active_stage.json', dict(source='b1_tracker/recovery2', audit='b1_tracker_audit/recovery2', updated=now()))
    from scripts.b123_safe_journal_20261027 import journal
    journal('b1_full_tracker_pilot_v1', 'Б1: точная GPU-регрессия пройдена, исправлена форма живого глаза',
        '*Получено.* b1_tracker/recovery1/incremental_regression.json: C/D/R1/B ref/current/identity совпали побитно при детерминированной сумме; проверка не ослаблялась. Первый живой кадр остановился до полного видео из-за формыR1. Дополнительная краткая диагностика сохранила исходный отказ и свой отказ записи метаданных; не научный результат.\n\n*Причина.* BoxEye возвращает B,frames,1,N; прежнийживой код сохраняллишнююединичнуюось, вR1 получался4D broadcast. Сохранённыеfr_hexимеютT,N. CPU-тест воспроизвёл ошибочную форму и точное исправление reshapeбез изменениязначений, не ослабление масок/проверок. Все7тестовpassed.\n\n*Продолжение.* Отдельная recovery2, протокол и allocationвb1_tracker/recovery2. Старыеисточники/отказы285/290/292сохранены, до сихпорполныхвидео0. УспешныйGPU replayне повторяется. НовыеGPU+аудиттолькоизостаткаисходного270окна, безновых10800с. ВидимыйPowerShell по прямойпросьбеавтора тольконаблюдатель; закрытиеего не останавливаетрасчёт.')
    print(p['recovery_budget'])


if __name__ == '__main__': main()
