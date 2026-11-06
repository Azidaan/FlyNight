"""Preserve the first audit record and append its evidence-based correction."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/flyvis_repairs_20261106'


def main():
    evidence='runs/flyvis_repairs_20261106/geometry_calibration_counterevidence.json'
    report=json.loads((ROOT/evidence).read_text(encoding='utf-8'))
    first=report['results'][0]
    assert first['resampling']=='hold' and first['aggregation']=='mean'
    assert first['resized_axes_old_gain_rmse']>first['old_rmse']
    cat=ROOT/'experiments/project_history.json'
    before=cat.read_bytes(); text=before.decode('utf-8'); data=json.loads(text)
    target=next(c for c in data['experiments'] if c['id']=='flyvis_system_repairs_batch1_20261106')
    assert not target.get('geometry_correction_20261106'), 'Already recorded'
    corrected=dict(target)
    now=datetime.now(timezone(timedelta(hours=3)))
    changes={
        'interpretation':'Подтверждены дефекты допуска неполных весов, выбора устройства, координат центров пикселей, короткого разгона, режима К1 только для оценки, времени Б1, изоляции Б2 и происхождения кэшей. Прежнее утверждение этой карточки об ошибке относительного масштаба векторов снято: масштаб обученного декодера задавался эмпирической калибровкой, и одной геометрии входа недостаточно для его изменения. Привязка параметров коннектома по именам групп является защитой от перестановки; в проверенных сохранённых вариантах неправильного порядка не было.',
        'decision':'Исправления загрузки и достоверности кэшей приняты по их собственным проверкам. Публикация геометрического пакета приостановлена после контрпримера к изменению масштаба векторов. Готовится ограниченная поправка координат с точным сохранением прежних векторов; её собственные проверки и риски изменения выборки внутри рамок должны быть рассмотрены отдельно. Изменение масштаба осей допускается только как явно выбранная диагностическая гипотеза без автоматического переноса старого коэффициента. Новый временной путь Б1 требует совместимых обученных весов; исторический Localizer З13 не мигрирован.',
        'cannot_claim':'Нет оценки HOTA, полного переобучения Б1 или доказательства отсутствия всех возможных ошибок. Техническое обучение в две итерации не является научным результатом. Прохождение тестов геометрических формул не доказывает правильность изменения масштаба обученного декодера. Сохранённая калибровка дала RMSE 2,0892894536234135 до изменения и 6,994735029744112 после переноса прежнего коэффициента в режиме hold/mean; даже повторная подгонка одного коэффициента дала 3,419143876619913 ('+evidence+', results[0]). В пяти из шести сохранённых сочетаний перенос прежнего коэффициента ухудшает RMSE; это не утверждение о всех возможных настройках.'}
    revisions=list(target.get('revisions',[]))
    for field,value in changes.items():
        revisions.append(dict(date=now.date().isoformat(),commit='',field=field,old=target[field],
                              reason='Независимый пересчёт сохранённой калибровки опроверг прежний вывод о масштабе векторов; '+evidence))
        corrected[field]=value
    corrected['revisions']=revisions
    corrected['geometry_correction_20261106']=evidence
    corrected['evidence']=[*target.get('evidence',[]),dict(kind='json',ref=evidence,claim='Контрпример к автоматической поправке масштаба векторов')]
    decoder=json.JSONDecoder(); start=text.index('"experiments"')
    # Locate this complete object without reformatting unrelated user content.
    pos=text.index('{',text.index('[',start))
    while True:
        obj,end=decoder.raw_decode(text,pos)
        if obj.get('id')==target['id']:
            break
        pos=text.index('{',end)
    replacement=json.dumps(corrected,ensure_ascii=False,indent=2)
    replacement='\n'.join(('    ' if i else '')+line for i,line in enumerate(replacement.splitlines()))
    updated=text[:pos]+replacement+text[end:]
    expected=dict(data); expected['experiments']=[corrected if c['id']==target['id'] else c for c in data['experiments']]
    assert json.loads(updated)==expected
    entry=f'''\n\n**{now:%d.%m.%Y, %H:%M} MSK** — Поправка собственного вывода о геометрии FlyVis

*Постановка.* Приоритет автора — не создавать новых ошибок. Сохранение прежних режимов и отдельная версия не заменяют проверку последствий. Перед публикацией геометрии независимо пересчитана сохранённая калибровка; исходные измерения не менялись.

*Получено (`{target['id']}`).* В сочетании hold/mean прежний RMSE равен 2,0892894536234135; масштабирование горизонтальной компоненты при прежнем коэффициенте даёт 6,994735029744112; повторная подгонка одного коэффициента даёт 3,419143876619913. Источник: `{evidence}`, `results[0]`. Ухудшение при переносе прежнего коэффициента обнаружено в пяти из шести сохранённых сочетаний; одно сочетание улучшилось.

*Вывод.* Прежняя запись об исправлении относительного масштаба осей была чрезмерной. Геометрия подготовки входа не доказывает эквивариантность обученного декодера. Речь о модуле FlyVis целиком; неверных сохранённых весов или связей ядра этим пересчётом не установлено. Старые формулировки сохранены в `revisions[]`; прежняя запись журнала остаётся без правок.

*Решение.* Публикация пакета геометрии остановлена до отдельной проверки. Рассматривается поправка только координат центров с побитовым сохранением прежних векторов и коэффициента; её влияние на выборку отсчётов внутри рамки проверяется отдельно. Новая калибровка не принята, исходные веса, кэши и результаты не перезаписаны. Наличие номера версии не является критерием безопасности.
'''.encode('utf-8')
    plan=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md'; old=plan.read_bytes(); marker='## Журнал изменений'.encode()
    assert old.count(marker)==1
    new=old.replace(marker,marker+entry,1)
    assert new.replace(marker+entry,marker,1)==old
    (OUT/'geometry_correction_card.json').write_text(json.dumps(corrected,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'geometry_correction_journal.txt').write_bytes(entry)
    cat.write_bytes(updated.encode('utf-8')); plan.write_bytes(new)
    sha=lambda raw:hashlib.sha256(raw).hexdigest()
    receipt=dict(at=now.isoformat(),old_final_fields_preserved_in_revisions=True,
        other_cards_unchanged=True,previous_plan_bytes_preserved=True,
        before=dict(catalogue=sha(before),plan=sha(old)),after=dict(catalogue=sha(cat.read_bytes()),plan=sha(new)))
    (OUT/'geometry_correction_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(json.dumps(receipt))


if __name__=='__main__':
    main()
