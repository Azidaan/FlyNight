"""Служебная запись финального расписания, без повторения измерений."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
from b123_safe_journal_20261027 import journal

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/b123_20261027'
DEST = RUN / 'final_stop'
source = Path(r'C:\Users\Егор\.codex\automations\claude-flyvis\automation.toml')
fields = {}
for line in source.read_text(encoding='utf-8').splitlines():
    if line.startswith(('id = ', 'name = ', 'status = ', 'rrule = ', 'target_thread_id = ', 'updated_at = ')):
        key, value = line.split(' = ', 1)
        fields[key] = json.loads(value)
assert fields['status'] == 'ACTIVE'
assert fields['rrule'] == 'RRULE:FREQ=DAILY;BYHOUR=4;BYMINUTE=29;BYSECOND=0;UNTIL=20261027T043000Z'
proof = dict(checked=datetime.now(timezone.utc).isoformat(), saved_automation=fields,
             reason='Финальный вызов перед сроком вместо получасового ожидания; срок расчётов не изменён.',
             intended_final_wake_utc='2026-10-27T04:29:00+00:00',
             intended_final_wake_msk='2026-10-27T07:29:00+03:00',
             local_hour_form_rejected_by_tool=True, utc_hour_form_accepted_by_tool=True,
             automation_duplicate_created=False,
             prior_automation_preserved='automation_before_final_wake.toml',
             no_new_calculations=True)
proof['sources_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (
    ROOT/'scripts/b123_pre_stop_check_20261027.py',
    ROOT/'scripts/b123_pre_stop_check_recovery1_20261027.py',
    DEST/'pre_stop_check.json', DEST/'pre_stop_check_recovery1.json')}
(DEST/'automation_schedule_verification.json').write_text(json.dumps(proof, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
text = ('Научные расчёты не запускались. final_stop/pre_stop_check_recovery1.json проверил PID и время создания: '
        'осталось только дерево существующей ожидающей очереди (её Python и conhost), готовых заданий нет; '
        'независимый guard recovery1 жив, статус свежий, источник и NIGHT_RESULTS.md не изменены. '
        'Первая служебная проверка ошибочно назвала conhost очереди посторонним расчётом; '
        'старые исходник и passed=false сохранены, отдельная версия проверяет принадлежность всему дереву очереди и passed=true. '
        'Существующая автоматизация claude-flyvis через automation_update переназначена на финальный вызов 07:29 МСК '
        '(04:29 UTC), чтобы следующий получасовой интервал не перескочил срок. '
        'Первая форма с локальным часом была отклонена инструментом как не имеющая будущего запуска; '
        'форма UTC принята, сохранённые поля проверены. Дубликатов нет. '
        'Источник final_stop/automation_schedule_verification.json, прежний TOML в final_stop/automation_before_final_wake.toml. '
        'Очередь должна завершиться 07:20, guard — закончить зарегистрированные деревья 07:29:30. '
        'В финальном вызове дождаться 07:30 ожиданиями не более60 секунд, проверить deadline_result.json/all_owned_ended '
        'и фактические PID+birth, затем приостановить эту автоматизацию через automation_update. '
        'При задержке вызова после срока сразу провести остановку/проверку и PAUSED; срок не продлевается. '
        'Никаких новых измерений, обучения, повторных тестов или пересборки графа ради ожидания. '
        'Каталог и граф в этом служебном шаге не менялись; .git/index.lock не трогался.')
journal('b123_final_stop_schedule_20261027', 'Финальная остановка ночной работы подготовлена', text)
with (RUN/'HANDOFF.md').open('a', encoding='utf-8') as stream:
    stream.write('\n\n## '+datetime.now().strftime('%H:%M')+' МСК — финальный вызов переназначен на07:29\n\n'+text+'\n')
print(json.dumps(dict(saved=True, proof='final_stop/automation_schedule_verification.json', no_new_calculations=True)))
