"""Append only the completed first engineering batch and its evidence."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import hashlib
import json
import re

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/flyvis_repairs_20261106'


def main():
    suite='runs/flyvis_repairs_20261106/integrated_1793990677683421900'
    assert json.loads((ROOT/suite/'result.json').read_text())['passed']
    assert '222 passed' in (ROOT/suite/'tests.log').read_text()
    sources=[
        ('runs/flyvis_system_fixes_20261106/loading_native_report.json','Строгая загрузка и явное устройство'),
        ('runs/flyvis_system_fixes_20261106/connectome_transfer_native_report.json','Перенос свободных параметров по именам групп'),
        ('runs/flyvis_system_fixes_20261106/motion/manifest.json','Геометрия и зависимые точки входа'),
        ('runs/flyvis_system_fixes_20261106/motion/probe_k1_report.json','Дифференцируемое преобразование К1'),
        ('runs/flyvis_timestamp_repair_20261106/real_D_v2/report.json','Новый временной путь Б1'),
        ('runs/flyvis_timestamp_repair_20261106/independent_validation.json','Независимый пересчёт временной проверки'),
        ('runs/flyvis_repairs_20261106/cache_native_v2/report.json','Совпадение детекций настоящего YOLO'),
        (suite+'/result.json','Совместная проверка исправлений')]
    for ref,_ in sources:
        assert (ROOT/ref).is_file()
    card=dict(id='flyvis_system_repairs_batch1_20261106',title='Авторская система — загрузка, геометрия, время и достоверность кэшей',
        area='infra',state='infrastructure',started='2026-11-06',done='2026-11-06',
        question='Какие воспроизводимые технические дефекты могут подменять веса, геометрию, время или происхождение сигналов?',
        hypothesis='Технический аудит существующего исполнения; научные гипотезы о пользе компонентов не проверяются.',
        expected='Устранение подтверждённых дефектов с проверкой зависимых потребителей, неизменности корректных расчётов и отказом при несовместимых входах.',
        acceptance=[dict(criterion='Совместно пройдены проверки загрузки, устройств, геометрии, временного пути Б1, сброса и кэшей. Критерий внесён после технической проверки, предварительной научной регистрацией не является.',met=True,evidence=suite+'/tests.log'),
                    dict(criterion='Штатные веса/активности при строгой загрузке и детекции корректного кэша совпадают с исходным исполнением; новое время проверено независимо по сохранённым массивам.',met=True,evidence='; '.join(ref for ref,_ in sources if 'native' in ref or 'independent' in ref))],
        obtained='Совместно пройдены 222 проверки ('+suite+'/tests.log). Строгая загрузка дала нулевое расхождение штатных выходов (runs/flyvis_system_fixes_20261106/loading_native_report.json). Для трёх сохранённых изображений и 55 детекций YOLO все прежние массивы совпали (runs/flyvis_repairs_20261106/cache_native_v2/report.json). Геометрия и зависимые потребители проверены в runs/flyvis_system_fixes_20261106/motion/manifest.json. Новый путь Б1 учитывает интервалы исходных кадров; последовательное исполнение совпало с обучающим проходом, контрольная точка технической пробы явно запрещена для обычного исполнения (runs/flyvis_timestamp_repair_20261106/real_D_v2/report.json).',
        interpretation='Подтверждены технические ошибки допуска неполных весов, выбора устройства, относительного масштаба осей и точек, короткого разгона, режима К1 только для оценки, времени Б1, изоляции Б2 и происхождения кэшей. Привязка параметров коннектома по именам групп является защитой от перестановки; в проверенных сохранённых вариантах неправильного порядка не было.',
        decision='Применить исправленные общие загрузчики, геометрию и построители кэшей. Старые кэши читать явно как сохранённые результаты; неизвестное происхождение не дописывать задним числом. Новый временной путь Б1 использовать с отдельно обученной совместимой моделью; исторический Localizer З13 не мигрирован.',
        cannot_claim='Нет оценки HOTA, полного переобучения Б1, новой калибровки общего усиления декодера или доказательства отсутствия всех возможных ошибок. Техническое обучение в две итерации не является научным результатом. Исправление геометрии подтверждает относительный масштаб осей, а не точность абсолютной скорости.',
        metrics=[],evidence=[dict(kind='json',ref=ref,claim=claim) for ref,claim in sources])
    cat=ROOT/'experiments/project_history.json'
    before=cat.read_bytes(); text=before.decode('utf-8'); data=json.loads(text)
    assert all(c['id']!=card['id'] for c in data['experiments'])
    start=re.search(r'"experiments"\s*:\s*\[',text).end()-1
    _,end=json.JSONDecoder().raw_decode(text,start)
    offset=len(text[:end-1].rstrip())
    insertion=',\n'+'\n'.join('    '+line for line in json.dumps(card,ensure_ascii=False,indent=2).splitlines())+'\n  '
    updated=text[:offset]+insertion+text[offset:]
    expected={**data,'experiments':[*data['experiments'],card]}
    assert json.loads(updated)==expected
    now=datetime.now(timezone(timedelta(hours=3)))
    entry=f'''\n\n**{now:%d.%m.%Y, %H:%M} MSK** — Первая проверенная группа исправлений авторской системы

*Постановка.* Продолжен разрешённый аудит с главным критерием «не создавать новых ошибок». Проверки этой записи технические; их критерии фиксируются после воспроизведения и не выдаются за предварительную регистрацию научного опыта.

*Получено (`{card['id']}`).* {card['obtained']}

*Вывод.* {card['interpretation']} {card['cannot_claim']}

*Решение.* {card['decision']} Проверка атомарного восстановления З13 и порядка отсева пар в самостоятельном трекере продолжается; в результаты этой записи они не включены.
'''.encode('utf-8')
    plan=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md'; old=plan.read_bytes(); marker='## Журнал изменений'.encode()
    assert old.count(marker)==1
    new=old.replace(marker,marker+entry,1)
    assert new.replace(marker+entry,marker,1)==old
    (OUT/'batch1_card.json').write_text(json.dumps(card,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'batch1_journal.txt').write_bytes(entry)
    cat.write_bytes(updated.encode('utf-8')); plan.write_bytes(new)
    sha=lambda b:hashlib.sha256(b).hexdigest()
    receipt=dict(at=now.isoformat(),card=card['id'],previous_cards_unchanged=True,previous_plan_bytes_preserved=True,
        before=dict(catalogue=sha(before),plan=sha(old)),after=dict(catalogue=sha(cat.read_bytes()),plan=sha(new)))
    (OUT/'batch1_record_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(receipt,ensure_ascii=False))


if __name__=='__main__':
    main()
