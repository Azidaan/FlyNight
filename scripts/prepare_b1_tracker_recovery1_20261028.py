"""Новая версия суммы, все проверки прежние, GPU+аудит внутри старого270окна."""
from pathlib import Path
from datetime import datetime
import sys, time, hashlib
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.recovery_research_service_20261027 import RUN, read, write, sha, matching, now


def main():
    original = RUN / 'b1_tracker'
    old_end = datetime.fromisoformat(read(original / 'execution_deadline.json')['deadline']).timestamp()
    for folder in ['b1_tracker', 'b1_tracker_audit', 'b1_incremental_diagnostic']:
        c = read(RUN / folder / 'controller_result.json')
        assert c['all_owned_ended'] and not any(matching(k, v) for k, v in c['owned'].items())
    diagnostic = read(RUN / 'b1_incremental_diagnostic/report.json')
    assert diagnostic['complete']
    assert all(not a['weights_changed'] for a in diagnostic['results'].values())
    assert all(z['exact'] for a in ['R1', 'B'] for pair in diagnostic['results'][a]['pairs'].values() for z in pair)
    assert any(not z['exact'] for z in diagnostic['results']['C']['pairs']['direct1-direct2'])
    folder = original / 'recovery1'; assert not folder.exists()
    p = read(original / 'protocol.json')
    audit_budget = 1800
    remaining = int(old_end - time.time())
    gpu_budget = remaining - audit_budget - 60
    assert gpu_budget > 600
    audit_source = (ROOT / 'scripts/audit_b1_tracker_20261028.py').read_text(encoding='utf-8')
    old = "source = RUN / 'b1_tracker'; folder = RUN / 'b1_tracker_audit'"
    assert audit_source.count(old) == 1
    executed = audit_source.replace(old, "source = RUN / 'b1_tracker/recovery1'; folder = RUN / 'b1_tracker_audit/recovery1'")
    extra = ['tracking/b1_deterministic_runtime.py', 'scripts/run_b1_tracker_recovery1_20261028.py',
        'scripts/audit_b1_tracker_recovery1_20261028.py', 'scripts/prepare_b1_tracker_recovery1_20261028.py',
        'tests/test_b1_deterministic_runtime.py', 'runs/recovery_research_20261027/b1_incremental_diagnostic/report.json',
        'runs/recovery_research_20261027/b1_tracker/controller_result.json',
        'runs/recovery_research_20261027/b1_tracker_audit/controller_result.json']
    p.update(created=now(), budget_seconds=gpu_budget, original_absolute_deadline=datetime.fromtimestamp(old_end).astimezone().isoformat(),
        original_absolute_deadline_unix=old_end, executed_audit_sha256=hashlib.sha256(executed.encode()).hexdigest(),
        recovery_reason='Даже два одинаковых старых GPU-прохода C/D различались до1.2e-6; R1/Bточны, веса неизменны. network.target_sum использует scatter_add_. Нельзя требовать недостижимую побитную повторяемость старой атомарной редукции. Вместо ослабления assertions включается torch deterministic algorithms ТОЛЬКО внутри FlyVis init/step. Численный порядок суммы меняется явно; MASA/головы/веса/пороги/действия прежние.',
        recovery_budget=dict(original270_seconds=10800, remaining_at_preparation=remaining, gpu_seconds=gpu_budget,
            new_version_audit_seconds=audit_budget, reserve_seconds=60,
            note='Все восстановление GPU+новый аудит внутри ОСТАТКА старого270абсолютного окна. Старый отказ275/его окно не продлеваются; это отдельная проверка новой версии, время взято из270, не добавлено к сроку.'),
        old_failures_preserved=True, original_exact_assertions_unchanged=True)
    p['sha256'].update({n: sha(ROOT / n) for n in extra})
    write(folder / 'protocol.json', p)
    audit = RUN / 'b1_tracker_audit/recovery1'
    write(audit / 'protocol.json', dict(parent_experiment='b1_full_tracker_pilot_v1', budget_seconds=audit_budget,
        original_absolute_deadline_unix=old_end, source='b1_tracker/recovery1', maximum_start_before_unix=old_end-audit_budget,
        allocation='Reserved inside old270 remaining budget, not extension of failed275'))
    write(original / 'recovery1_allocation.json', dict(created=now(), **p['recovery_budget'], original_absolute_deadline_unix=old_end))
    for name, f, script in [('285_b1_tracker_deterministic_recovery1', folder, 'scripts/run_b1_tracker_recovery1_20261028.py'),
                            ('290_b1_tracker_deterministic_audit', audit, 'scripts/audit_b1_tracker_recovery1_20261028.py')]:
        target = RUN / 'ready_jobs' / (name + '.json'); assert not target.exists()
        write(target, dict(name=name, folder=f.relative_to(ROOT).as_posix(), script=script,
            script_sha256=sha(ROOT / script), protocol_sha256=sha(f / 'protocol.json')))
    from scripts.b123_safe_journal_20261027 import journal
    journal('b1_full_tracker_pilot_v1', 'Б1: причина раннего отказа и отдельное детерминированное исполнение',
        '*Получено.* 270остановилась до первого полного видео: строгая проверка истории не совпала. 275сохранила отказ зависимости; оба дерева закончены. В b1_incremental_diagnostic/report.json повторения того же старого вычисления C/D тоже не побитны (до1,2e-6), R1/Bточны, веса неизменны. Это не отрицательная HOTA.\n\n*Исправление.* Отдельные b1_deterministic_runtime.py/run_b1_tracker_recovery1_20261028.py включают детерминированное суммирование FlyVis; точные assertions не ослабляются. Пять CPU-тестов прошли. Новые проверки GPU+CPU выделены ВНУТРИ остатка прежнего270окна, источник b1_tracker/recovery1_allocation.json;10800сне возобновляются заново, старый275не продлевается. Ни одного полного видео ещё нет, повтор готового отсутствует.\n\n*Осталось.* Подтвердить точное совпадение инкрементального/прямого прохода в этом режиме, первый полный7-ветвевой видеопроход и прежний гейт цены, полная серия только при допуске. КачествоБ1не установлено.')
    print(p['recovery_budget'])


if __name__ == '__main__': main()
