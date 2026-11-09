"""Improve readability of current conclusions; preserve previous versions."""
import json
from pathlib import Path
from datetime import datetime, timezone, timedelta
import hashlib

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
RUN = ROOT / 'runs/identity_channel_training_readiness_20261109'
now = datetime.now(timezone(timedelta(hours=3))).isoformat()

def read(path):
    return json.loads(path.read_text(encoding='utf8'))

def save(path, obj):
    temp = path.with_suffix(path.suffix + '.readability.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    temp.replace(path)

def revise(obj, field, new):
    old = obj[field]
    if old != new:
        obj.setdefault('revisions', []).append(dict(date=now, field=field, old=old,
            reason='Исправлены пробелы и объяснение терминов; числа, критерии и статус не изменены.'))
        obj[field] = new

path = ROOT / 'experiments/project_history.json'
data = read(path)
card = next(c for c in data['experiments'] if c['id'] == RUN.name)
revise(card, 'obtained', 'В 69 исторических видео, исключая 15 уже просмотренных, найдено 103 однозначно размеченные связи с последним наблюдением. Для 91 связи хватает ячеек: 15 своих и 76 чужих, 32 видео. Во всех 91 случаях опора совпала с последним фактическим наблюдением исходного трека. При исключении любого из 32 видео в оставшейся обучающей группе сохраняются оба класса. Ещё 233 связи используют более раннюю размеченную опору; поддержаны 204 (28 своих и 176 чужих), 45 видео. Они учтены отдельно. Сохранены 50 неоднозначных текущих меток. Выполнено 579 проверок исходной разметки; признаки, обучение и трекер не запускались. Источники: runs/identity_channel_training_readiness_20261109/results.json, causal_reference_audit.json, audit.json.')
revise(card, 'interpretation', 'Прежние 45 условных пар не исчерпывают архив. Получены отдельные прямо размеченные примеры, в которых прошлый шаблон доступен к моменту решения. Но 15 своих пар из 10 видео всё ещё создают риск переобучения. Наличие обоих классов в обучении не доказывает надёжность модели или сохранение ID.')
revise(card, 'decision', 'Допустить один заранее зафиксированный ограниченный прототип обучаемой оценки пространственных совпадений каналов. Проверять с исключением целых видео и одинаковыми контролями на обычном изображении и геометрии. По 15 просмотренным видео порог не выбирать. Ранние размеченные опоры в основную группу не добавлять.')
revise(card, 'cannot_claim', '91 связь не означает 91 независимый сюжет или 91 сохранённый ID. Запросы взяты до общего назначения: часть рамок уже имеет другого владельца в исходном выводе. Это не исключительно свободный остаток после назначения. Публичный владелец не доказывает этап назначения. Историческая группа fit84 уже использовалась прежними обучаемыми методами; новой официальной тестовой выборки здесь нет. Новые признаки, обучаемая оценка, качество классификации и HOTA на этом этапе не получены.')
save(path, data)
body = '# ' + card['title'] + '\n'
for title, field in [('1. Получено', 'obtained'), ('2. Значение', 'interpretation'), ('3. Решение', 'decision'), ('4. Ограничения', 'cannot_claim')]:
    body += '\n' + title + '\n\n' + card[field] + '\n'
(RUN / 'ИТОГ.md').write_text(body, encoding='utf8')

v4 = (HERE / 'СВОДКА_v3.md').read_text(encoding='utf8').split('\n7. Проверена доступность', 1)[0]
v4 += '\n7. Проверена доступность обучающих данных\n\n' + card['obtained'] + '\n\n' + card['interpretation'] + '\n\n' + card['decision'] + '\n\n' + card['cannot_claim'] + '\n'
(HERE / 'СВОДКА_v4.md').write_text(v4, encoding='utf8')
(HERE / 'АКТУАЛЬНОЕ.md').write_text('# Актуальная накопленная постановка\n\n[Текущая сводка — версия 4](СВОДКА_v4.md).\n\n[Критическая проверка Gemini](ПРОВЕРКА_GEMINI.md).\n\n[Проверка доступности размеченных пар](../runs/identity_channel_training_readiness_20261109/ИТОГ.md). Предыдущие версии сохранены.\n', encoding='utf8')

path = ROOT / 'experiments/system_problems.json'
data = read(path)
p = next(p for p in data['problems'] if p['id'] == 'С2')
revise(p, 'fact', p['fact'].replace('СВОДКА_v3.md', 'СВОДКА_v4.md'))
revise(p, 'next_check', 'Накопленная постановка: research_identity_synthesis_20261109/СВОДКА_v4.md. Для отдельной проверки доступны 91 прямо размеченная пара с последним фактическим наблюдением: 15 своих и 76 чужих, 32 видео вне 15 уже просмотренных. Следующий шаг — один заранее зафиксированный прототип обучаемой оценки пространственных совпадений отдельных каналов, с ограничением сложности модели, исключением целых видео и одинаковыми контролями на обычном изображении и геометрии. Это рабочая гипотеза, а не подтверждённое решение. Ранние размеченные опоры в основную группу не включать. До обучения нужен собственный протокол.')
p['evidence'].append(dict(source='research_identity_synthesis_20261109/СВОДКА_v4.md', sha256=hashlib.sha256((HERE / 'СВОДКА_v4.md').read_bytes()).hexdigest(), claim='Читаемая актуальная постановка; прежние версии и результаты сохранены.'))
p['updated'] = now
save(path, data)

# Journal is append-only, including the correction of the earlier unreadable wording.
plan = ROOT / 'dissertation/ЕДИНЫЙ_ПЛАН.md'
text = plan.read_text(encoding='utf8')
stamp = datetime.now(timezone(timedelta(hours=3))).strftime('%d.%m.%Y, %H:%M MSK')
entry = '**' + stamp + '** — Накопленная постановка и проверка данных: читаемые текущие выводы\n\nИсправлены слитные слова в текущей карточке `identity_channel_training_readiness_20261109` и её отчёте. Прежние поля сохранены в revisions; прежние сводки v1–v3 сохранены. Актуальная версия — `research_identity_synthesis_20261109/СВОДКА_v4.md`. Числа, критерии, ограничения и статус не изменены. Правило накопления проблем записано в `experiments/PROBLEM_SYNTHESIS_RULE.md`. Нового научного расчёта здесь нет.\n\n'
plan.write_text(text.replace('## Журнал изменений\n', '## Журнал изменений\n\n' + entry, 1), encoding='utf8')
print('Readable current records saved; previous fields retained.')
