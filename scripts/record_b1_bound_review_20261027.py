"""Записать завершённую математическую сверку; отдельный граф не менять."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import json

ROOT = Path(__file__).resolve().parents[1]


def main():
    catalog = ROOT/'experiments/project_history.json'
    original = catalog.read_bytes()
    data = json.loads(original.decode('utf-8-sig'))
    identifier = 'b1_residual_bound_audit_v1'
    assert not any(c['id'] == identifier for c in data['experiments'])
    source = 'runs/b123_20261027/b1_residual_bound/'
    assert json.loads((ROOT/source/'independent_audit.json').read_text(encoding='utf-8'))['passed']
    data['experiments'].append(dict(id=identifier,
        title='Б1 — граница исправления ошибок ограниченной временной поправкой',
        area='flyvis', state='infrastructure', started='2026-10-27',
        question='Существуют ли ошибки ранжирования личности, которые текущая поправка tanh математически не может исправить?',
        hypothesis='', expected='Граница устанавливается для фиксированных оценок и кандидатов; условия подсчёта сохранены до чтения результатов.',
        acceptance=[dict(criterion='Для полностью размеченных реальных кандидатов проверить строгую непересекаемость достижимых оценок и независимо сверить через разрыв логарифмов шансов.',
                         met=True, evidence=source+'independent_audit.json')],
        done='Выполнен расчёт по сохранённым файлам без Torch, обучения и изменения работающего Б1. Другая формула независимо воспроизвела счётчики.',
        obtained='В '+source+'report.json, splits.fit84: 1039 подходящих запросов, 72 ошибки ранжирования, 46 неустранимы ограниченной поправкой. В splits.fit20: 117 запросов, 5 ошибок, 1 неустранима. Контрпример uid4074 и разрыв2,8661533344333745 — '+source+'independent_audit.json, fit20_counterexamples. HOTA не измерялась.',
        interpretation='Подтверждена граница выразимости нынешнего исправления оценок, а не бесполезность временной информации или всей Б1. Набор действий и последующий выбор не сводятся к этому ранжированию.',
        decision='Сохранить ограничение для разбора итогов. Идущий протокол не менять. Отдельный граф «Гипотезы» не обновлять без команды автора.',
        cannot_claim='Это не доля всех ошибок трекера, не потолок HOTA, не доказательство провала Б1–Б3 и не основание менять критерии после измерения.',
        files=['scripts/b1_residual_bound_audit_20261027.py','scripts/b1_residual_bound_verify_20261027.py',
               'dissertation/Литературная проверка Б1–Б3 — ограничения и восстановление пропусков.md'],
        evidence=[dict(kind='file',ref=source+n,claim=claim) for n,claim in
            [('protocol.json','Условия до подсчёта'),('report.json','Первичные счётчики и границы'),
             ('independent_audit.json','Независимая формула и контрпример fit20')]]))
    maincard = next(c for c in data['experiments'] if c['id']=='flyvis_temporal_residual_association_v1')
    maincard['evidence'].append(dict(kind='file',ref=source+'independent_audit.json',
                                  claim='Узкое ограничение текущей амплитуды поправки; не результат научной приёмки Б1'))
    assert catalog.read_bytes() == original, 'Concurrent catalog change; preserve it'
    newline = '\r\n' if b'\r\n' in original else '\n'
    catalog.write_bytes((json.dumps(data, ensure_ascii=False, indent=2)+'\n').replace('\n',newline).encode('utf-8'))
    plan = ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md'
    raw = plan.read_bytes(); text = raw.decode('utf-8')
    stamp = datetime.now(timezone(timedelta(hours=3))).strftime('%d.%m.%Y, %H:%M MSK')
    entry = f'''**{stamp}** — граница нынешней Б1, адресная литературная проверка и восстановление исполнения

*Получено (`{identifier}`).* Текущая tanh-поправка ограничивает сдвиг логарифма шансов каждого кандидата величиной от −1 до +1. Разрыв более 2 между ошибочной и любой правильной областью неустраним. По полностью известным меткам реальных областей с наблюдаемой историей: fit84 — 1039 запросов, 72 ошибки исходного ранжирования, 46 неустранимы; fit20 — 117 запросов, 5 ошибок, 1 неустранима. Источники: `{source}report.json`, `splits`; независимая сверка `{source}independent_audit.json`. Условия сохранены до подсчёта. Это ограничение ранжирования, не отрицательная приёмка Б1 и не измерение HOTA; выбор действия может использовать другие признаки и отказ.

*Литература.* В `dissertation/Литературная проверка Б1–Б3 — ограничения и восстановление пропусков.md` сохранены первичные источники и границы переноса: отрицательное добавление геометрии/движения в QDTrack; BUSCA как кандидат восстановления пропусков, включая опыт BDD100K с дообучением на SHIFT; условная гарантия SPIBB. Обещания 100-процентной прибавки нет. BUSCA не запускалась, веса и данные не скачивались; Б1–Б3 не заменены новым планом.

*Восстановление исполнения.* Исходный рабочий процесс Б1 остановлен после повторных наблюдений задержки в добавлении стимула; checkpoint C сохранён. `runs/b123_20261027/b1_training/service_interruption.json` и `service_recovery1.json` фиксируют сбой и отдельную служебную версию: присваивание в нулевой буфер вместо сложения, проверенная эквивалентность на CPU, сохранение оценки по хешу весов. Причина исходной задержки окончательно не доказана. Обучение C не повторялось. C завершил оценку fit20 (`C/report.json`), теперь учится D. Поле seconds отчёта C относится к восстановленному запуску и не является полным временем первоначального обучения.

*Переходы и сохранность.* Очередь `scripts/b123_ready_queue_recovery1_20261027.py` ожидает именно `b1_training/recovery1_controller.json` и подтверждённого конца дерева; прежняя версия сохранена. Источник и зависимость — `runs/b123_20261027/queue_recovery1_manifest.json`; процесс зарегистрирован у независимого контроллера общего срока. Бюджеты и научные условия не продлены. Каталог обновлён, обязательные проверки выполняются после записи. Отдельный граф «Гипотезы» не пересобирался по новому правилу автора. Коммит блокирует существующий `.git/index.lock`; он не удалён.

'''
    marker='## Журнал изменений\n'
    assert marker in text
    text=text.replace(marker,marker+'\n'+entry,1)
    assert plan.read_bytes()==raw, 'Concurrent journal change; preserve it'
    plan.write_bytes(text.encode('utf-8'))
    handoff=ROOT/'runs/b123_20261027/HANDOFF.md'
    with handoff.open('a',encoding='utf-8') as f:
        f.write('\n## '+stamp+' — готовая очередь и ограничения Б1\n\n')
        f.write('Активна queue_recovery1, зарегистрированный корень PID2668. Она ждёт recovery1_controller.json, а не старый training_controller.json. C/report.json готов, D учится. Обучение C не повторено, срок ступени02:28:33 сохранён. Каталог после математической сверки проверяется. Группа «Гипотезы» не обновляется без команды автора.\n\n')
        f.write('b1_residual_bound: независимая сверка прошла, это граница ранжирования, не HOTA и не провал всейБ1. Литература сохранена в dissertation/Литературная проверка Б1–Б3 — ограничения и восстановление пропусков.md; BUSCA не скачивалась/не запускалась, нового поручения на граф нет.\n')


if __name__=='__main__':main()
