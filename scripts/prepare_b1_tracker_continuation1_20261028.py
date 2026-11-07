"""Зафиксировать прямое разрешение автора продолжить отрицательный ценовой гейт."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import sys, shutil, hashlib, copy
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.recovery_research_service_20261027 import RUN, read, write, sha, matching, now


def prepare():
    from scripts.run_b1_tracker_continuation1_20261028 import continuation_source
    from scripts.audit_b1_tracker_continuation1_20261028 import audit_source
    folder = RUN / 'b1_tracker/continuation1'; assert not folder.exists()
    old = RUN / 'b1_tracker/recovery2'; r = read(old / 'partial_result.json')
    a = read(old / 'independent_audit.json'); assert a['passed'] and not a['full_series']
    for sub in ['b1_tracker/recovery2', 'b1_tracker_audit/recovery2', 'b1_result_checks']:
        c = read(RUN / sub / 'controller_result.json')
        assert c['exit_code'] == 0 and c['all_owned_ended'] and not any(matching(k, v) for k, v in c['owned'].items())
    authorization = dict(created=now(), user_request='Запускай прогон дальше, сейчас то хоть результат может быть, чего останавливаться то?',
        interpretation='Прямое поручение продолжить оставшиеся19видеоБ1 после явно сообщённого автору отрицательного гейтастоимости. Первое полное видео сохраняется, не повторяется. Это разрешение отменяет остановку этой серии по прежней цене, но не общий срок28.10 07:40МСК и не научные критерии.',
        cost_stop_override_authorized=True, original_gate_proceed=False,
        original_price_sha256=sha(old / 'price.json'), inference_repeated=False,
        scientific_settings_changed=False, author_save_at='2026-10-28T07:30:00+03:00',
        author_hard_stop='2026-10-28T07:40:00+03:00', gpu_stop_target='2026-10-28T06:15:00+03:00',
        audit_reserved_seconds=4200, limitation='Консервативнаяоценкане помещаетсявновоеокно;естьрискнеполнойсерииксроку. Разрешениеавторане превращаетстарыйгейтвположительный. Независимыйконтроллеростанетсяобязательным.')
    write(folder / 'author_override.json', authorization)
    p = read(old / 'protocol.json')
    p.update(created=now(), budget_seconds=1, author_continuation=True,
        gpu_stop_target=authorization['gpu_stop_target'], audit_budget_seconds=4200,
        author_override=(folder / 'author_override.json').relative_to(ROOT).as_posix(),
        original_cost_gate_preserved=True, cost_gate_overridden_by_direct_author_request=True,
        cost_gate='Первое видео уже полностью измерено и аудировано. Отрицательный исход прежнего гейта сохранён. По прямому поручению автора оставшиеся19продолжаются без повторного ценового отказа и без выбора по HOTA. Сохраняются фиксированные общий срок и независимая остановка; незавершённая серия не получает научного вывода.',
        continuation_scope='Same20/same7arms/sameweights/sameprecision/samethreshold. Reuse first full video by verified copies; infer only remaining19. No training or parameter selection.',
        executed_worker_sha256=hashlib.sha256(continuation_source().encode()).hexdigest(),
        executed_audit_sha256=hashlib.sha256(audit_source().encode()).hexdigest())
    # Старые сроки/отказы остаются в старом протоколе, новый действует по прямому поручению.
    for key in ['original_absolute_deadline', 'original_absolute_deadline_unix', 'recovery_budget']:
        p.pop(key, None)
    copied = {}; first = r['completed_videos'][0]
    for arm in p['variants']:
        for rel in [Path(arm) / (first + '.json'), Path('events') / arm / (first + '.json'), Path('raw') / arm / (first + '.npz')]:
            src = old / rel; target = folder / rel; target.parent.mkdir(parents=True, exist_ok=True)
            expected = sha(src); shutil.copy2(src, target); assert sha(target) == expected
            copied[target.relative_to(ROOT).as_posix()] = expected
    first_marker = read(old / 'completed' / (first + '.json'))
    write(folder / 'completed' / (first + '.json'), dict(first_marker, reused_without_inference=True, source=(old / 'completed' / (first + '.json')).relative_to(ROOT).as_posix()))
    seed = dict(completed_videos=[first], per_video=read(old / 'per_video.json'), proof={first: first_marker['proof']},
        timing={first: first_marker['seconds']}, output_sha256=copied,
        old_audit_sha256=sha(old / 'independent_audit.json'), old_source=old.relative_to(ROOT).as_posix())
    write(folder / 'resume_seed.json', seed)
    write(folder / 'per_video.json', seed['per_video'])
    write(folder / 'price.json', dict(read(old / 'price.json'), author_override=True, original_gate_proceed=False,
        original_source=(old / 'price.json').relative_to(ROOT).as_posix(), resumed_after_direct_author_request=True))
    extra = ['scripts/run_b1_tracker_continuation1_20261028.py', 'scripts/audit_b1_tracker_continuation1_20261028.py',
        'scripts/prepare_b1_tracker_continuation1_20261028.py', 'tests/test_b1_tracker_continuation.py',
        (folder / 'author_override.json').relative_to(ROOT).as_posix(), (folder / 'resume_seed.json').relative_to(ROOT).as_posix(),
        (old / 'independent_audit.json').relative_to(ROOT).as_posix(), (old / 'partial_result.json').relative_to(ROOT).as_posix()]
    p['sha256'].update({f: sha(ROOT / f) for f in extra})
    write(folder / 'protocol.json', p)
    write(RUN / 'b1_tracker_audit/continuation1/protocol.json', dict(budget_seconds=4200, parent_experiment='b1_full_tracker_pilot_v1',
        source='b1_tracker/continuation1', author_save_at=authorization['author_save_at']))
    path = ROOT / 'experiments/project_history.json'; before = path.read_bytes(); cat = read(path)
    card = next(c for c in cat['experiments'] if c['id'] == 'b1_full_tracker_pilot_v1')
    for key in ['obtained', 'interpretation', 'decision', 'cannot_claim', 'done', 'acceptance']:
        if card.get(key):
            card.setdefault('revisions', []).append(dict(date='2026-10-28', commit='', field=key, old=copy.deepcopy(card[key]),
                reason='Прямое поручение автора продолжить после отрицательной оценки стоимости; предыдущий ценовой стоп и его научная неопределённость сохранены.'))
    card.update(state='running', done='', obtained=card['obtained']+'\n28.10авторпрямопоручилпродолжитьоставшиеся19; первое видео переиспользуется по хешам, источник '+(folder/'author_override.json').relative_to(ROOT).as_posix(),
        interpretation='На первом видео технический путь и независимый аудит прошли; по качеству научного вывода пока нет. Прежний отрицательный ценовой гейт не изменён, но его остановка отменена прямым поручением автора. Продолжается та же серия20со всеми7вариантами.',
        decision='Продолжить19безповторапервого и без изменения весов/порогов/входов. GPUограничен06:15МСКсрезервомдляполногоCPUаудита, общийсрок07:40МСКнеизменен. Приёмка только после20и независимогоаудита; до завершения Б1непереходитькБ2/Б3.',
        cannot_claim='Прежняяконсервативнаяоценкаостаётсяотрицательной;новоеразрешениенеобещаетзавершениексрокуилиприростHOTA. Single seedпилотнеподтверждаетвсюБ1. Не менятьнастройкипоувиденномувидео. Историческийvalненовыйнезависимыйтест. Старыеокна/отказы/первичныефайлынепереписываются.')
    for criterion in card['acceptance']:
        criterion.update(met=None, evidence='Продолжение разрешено автором; полный критерий ещё не оценён. Прежний исход сохранён в revisions.')
    card['evidence'].append(dict(kind='file', ref=(folder / 'author_override.json').relative_to(ROOT).as_posix(), claim='Прямое разрешение продолжить вопреки цене, без смены научных условий'))
    assert path.read_bytes() == before; write(path, cat)
    from scripts.b123_safe_journal_20261027 import journal
    journal(card['id'], 'автор поручил продолжить Б1 после ценового стопа',
        '*Прямое поручение.* «'+authorization['user_request']+'». Автору перед этим сообщены отрицательная цена и один полный результат.\n\n*Изменение исполнения.* Продолжаются оставшиеся19, первое видео и все21файлвыходов скопированы с проверкой SHA, не пересчитываются. Веса/пороги/7контролей/критерии не меняются. Отрицательный ценовой гейт сохранён, не назван пройденным; его остановка отменена автором. Новый протокол b1_tracker/continuation1/protocol.json.\n\n*Сроки.* GPUдо06:15МСК,4200срезервCPUаудита,сохранение07:30/общаяостановка07:40неизменны. Есть риск неполнойсерии по консервативнойцене. Новоеокноявноразрешено,старые270/recovery2непродлеваютсязаднимчислом. Прежние итоговые поля и критерии карточки сохранены вrevisions. ДваCPUтестапродолженияпрошли;передпубликациейобязательныекаталог/pytest.')


def publish():
    folder = RUN / 'b1_tracker/continuation1'; p = read(folder / 'protocol.json')
    target = datetime.fromisoformat(p['gpu_stop_target']).timestamp()
    import time
    p['budget_seconds'] = int(target - time.time() - 10)
    assert p['budget_seconds'] > 600
    write(folder / 'protocol.json', p)
    for name, f, script in [('330_b1_author_continuation', folder, 'scripts/run_b1_tracker_continuation1_20261028.py'),
                            ('340_b1_author_continuation_audit', RUN / 'b1_tracker_audit/continuation1', 'scripts/audit_b1_tracker_continuation1_20261028.py')]:
        dest = RUN / 'ready_jobs' / (name + '.json'); assert not dest.exists()
        write(dest, dict(name=name, folder=f.relative_to(ROOT).as_posix(), script=script,
            script_sha256=sha(ROOT / script), protocol_sha256=sha(f / 'protocol.json')))
    write(RUN / 'b1_active_stage.json', dict(source='b1_tracker/continuation1', audit='b1_tracker_audit/continuation1', updated=now()))


if __name__ == '__main__':
    publish() if '--publish' in sys.argv else prepare()
