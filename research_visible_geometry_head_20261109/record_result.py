"""Record the mask's failed preregistered transfer gate without erasing its CV gain."""
import json, hashlib, time
from pathlib import Path
from datetime import datetime, timezone, timedelta
ROOT=Path(__file__).resolve().parents[1]; OUT=Path(__file__).resolve().parent
def read(p): return json.loads(p.read_text(encoding='utf8'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def atomic(path,data,before):
    temporary=path.with_suffix(path.suffix+'.visible_result.pending')
    assert not temporary.exists();temporary.write_bytes(data)
    for _ in range(30):
        assert path.read_bytes()==before
        try: temporary.replace(path);return
        except PermissionError: time.sleep(.1)
    raise PermissionError(str(path))
now=datetime.now(timezone(timedelta(hours=3)))
s=read(OUT/'summary.json');a=read(OUT/'audit.json');screen=read(OUT/'deployment_screen.json');final=read(OUT/'final_head_audit.json')
assert s['gates']['C34__mask_only'] and not s['gates']['C34__aligned'] and a['passed'] and final['passed']
assert screen['passed'] and not screen['gate'] and screen['old_errors']==screen['new_errors']==3
assert len(final['cv_decision_changes'])==1 and final['cv_decision_changes'][0]['label']=='foreign'
report='''# Видимая часть шаблона: частный выигрыш не перенёсся на рабочие ошибки

1. Итог

На 26 обучающих связях выбор видимой части улучшил локальный допуск FlyVis. Дополнительное преобразование координат ухудшило его. Проверка выбранной маски на пяти заранее перечисленных рабочих случаях оставила те же три ошибки. По записанному до этой проверки критерию полный трекер данной версии не запускается. Проблема сохранения личности остаётся открытой.

2. Сравнение на одних и тех же связях

Использованы 20 своих и 6 чужих связей из 20 обучающих видео, с исключением целого видео при обучении. Порог 0, штраф 1, прежние 19 сдвигов и семь геометрических признаков сохранены. Сеть и карты не пересчитывались.

FlyVis: прежний полный шаблон принял 17 своих и 2 чужие; только видимая опора — 17 и 1; та же опора с согласованным масштабом и смещением — 17 и 4. Оба контроля масштаба признаков дали 17 и 2. Источник: summary.json → metrics, ready_baselines, gates.

Обычный пространственный вход: видимая опора — 16 своих и 3 чужие, преобразование координат — 15 и 4; прежний вариант — 17 и 3. Обычный временной вход: 13 и 5, 16 и 5, прежний — 16 и 4. Все их допуски отрицательны. Источник: summary.json. Это одинаковые правила чтения для разных сигналов, а не вывод о пользе ядра для всего трекера.

Единственное изменение решения FlyVis относительно прежнего: чужая связь 193 → raw4824 на кадре 72 видео 02521568-a335c1a0 перестала проходить. Все решения на своих сохранились. Источник: final_head_audit.json → cv_decision_changes. Частный результат на шести чужих не является независимым доказательством.

3. Проверка рабочих ошибок без нового полного запуска

До четырёх окончательных обучений зафиксированы пять текущих запросов прежнего полного пилота, четыре своих и один чужой. Они не обучали модель и не определяли маску или порог. Нужно сохранить два прежних верных решения и уменьшить три ошибки. Этот отдельный критерий зарегистрирован после сравнения обучающих связей, но до получения окончательных весов и пяти новых оценок: deployment_screen_protocol.json.

Своя 232 → raw54943, кадр 544: балл −0,426740 → −0,478298; отказ остался. Своя 832 → raw31468, кадр 540: −0,398460 → −0,077641; отказ остался. Чужая 574 → raw35274, кадр 937: +0,546382 → +0,481625; разрешение осталось. Свои 70 и 1102 сохранили разрешение. Источник: deployment_screen.json → records. Баллы — выход классификатора, не вероятность и не факт передачи ID.

Все пять геометрических преобразований переиспользованы после точного совпадения состояний и входов. Новых поисков геометрии, прогонов ядра, обработки видео и полных траекторий не было. Источник: deployment_screen.json → ready_geometry_reused=5, missing_geometry_restored=0.

4. Проверка реализации и граница вывода

Независимо проверены 240 новых моделей с исключением видео; максимальная невязка градиента 6,04·10⁻¹⁰. Четыре окончательные модели проверены отдельным расчётом функции потерь, градиента и эквивалентности развёрнутых коэффициентов; готовые обучения не повторялись. Источники: audit.json и final_head_audit.json.

Выбор опоры помог одной обучающей чужой связи, но не исправил известные рабочие ошибки. Это не доказывает бесполезность любых видимых шаблонов. Оценка масштаба Калманом также не объявляется истинным размером объекта. Нельзя утверждать, что улучшены реальные ID, HOTA или ядро FlyVis.

5. Следствие для общего заключения

Не увеличивать стоимость вмешательства и не подбирать порог под эти примеры. Сопоставить ограничения самого представления и его обучения; следующий запрос Gemini должен искать одну различающую проверку, учитывая отрицательные результаты контраста пика, координат и маски. Общая конкуренция треков остаётся отдельным механизмом.
'''
with (OUT/'РЕЗУЛЬТАТ.md').open('x',encoding='utf8') as f:f.write(report)
identity='visible_geometry_head_20261109';path=ROOT/'experiments/project_history.json';before=path.read_bytes();cat=json.loads(before);card=next(c for c in cat['experiments'] if c['id']==identity)
assert card['state']=='planned'
updates=dict(obtained='На 26 связях (20 своих, 6 чужих) FlyVis: полный 17/2, видимая опора 17/1, согласованные координаты 17/4; масштабные контроли 17/2. Источник: research_visible_geometry_head_20261109/summary.json → metrics, ready_baselines. Единственное изменение — отказ чужой 193→raw4824, свои не менялись: final_head_audit.json → cv_decision_changes. На пяти заранее выбранных рабочих запросах ошибки остались 3→3: deployment_screen.json. Независимо проверены 240 промежуточных и 4 окончательные модели: audit.json; final_head_audit.json.',
    interpretation='Сама видимая опора имеет частный выигрыш на обучающем сравнении; геометрическое преобразование его не поддержало. Выбранная маска не прошла отдельный зарегистрированный критерий переноса на рабочие ошибки. Дефект ядра не установлен.',
    decision='Не запускать полный трекер этой версии. Сохранить частный выигрыш и отрицательный перенос раздельно. Следующий механизм выбирать из накопленной сводки и первичных источников, без подбора порога или повторения этих расчётов.',
    cannot_claim='26 связей и пять ранее просмотренных рабочих запросов — разработческие данные, не независимый итоговый тест. Балл не означает выдачу ID. Нет результата HOTA новой маски, подтверждённой пользы ядра или доказанной истинности масштаба Калмана.')
for field,value in updates.items():
    if card.get(field):card.setdefault('revisions',[]).append(dict(date=now.isoformat(),field=field,old=card[field],reason='Завершены компонентное сравнение и отдельный зарегистрированный рабочий экран.'))
    card[field]=value
card.update(state='completed_negative',done=now.strftime('%Y-%m-%d'))
for item,met,ref in zip(card['acceptance'],[True,True,True],['features.json; readiness.json','summary.json → gates','audit.json; final_head_audit.json']):item.update(met=met,evidence='research_visible_geometry_head_20261109/'+ref)
card['acceptance'].append(dict(criterion='Отдельный критерий до окончательных весов, deployment_screen_protocol.json: на пяти рабочих запросах сохранить все прежние верные решения и строго уменьшить число ошибок с трёх. При провале полный пилот не запускать.',met=False,evidence='research_visible_geometry_head_20261109/deployment_screen.json → old_errors=3, new_errors=3, gate=false'))
card['evidence'] += [dict(kind='file',ref='research_visible_geometry_head_20261109/'+n,claim=claim) for n,claim in [('summary.json','Частный выигрыш маски и отрицательное преобразование координат'),('deployment_screen_protocol.json','Отдельный критерий до окончательных весов и рабочих баллов'),('deployment_screen.json','Те же три рабочие ошибки'),('final_head_audit.json','Независимая арифметика и единственное изменение решений'),('РЕЗУЛЬТАТ.md','Отрицательный перенос при сохранении частного выигрыша')]]
atomic(path,(json.dumps(cat,ensure_ascii=False,indent=2)+'\n').encode(),before)
path=ROOT/'experiments/system_problems.json';before=path.read_bytes();problems=json.loads(before);problem=next(x for x in problems['problems'] if x['id']=='С2')
problem.setdefault('revisions',[]).append(dict(date=now.isoformat(),field='next_check',old=problem['next_check'],reason='Согласованное обучение видимого шаблона завершено; рабочий экран отрицательный.'))
problem['next_check']='Не запускать отрицательные версии контраста, координат и видимой маски. По накопленной сводке отделить потерю информации в поканальной корреляции от недостаточного обучения принадлежности. Перед следующей заменой получить одну различающую проверку с первичным источником и теми же обычными входами; не возвращаться к подбору порога, готовой ранней памяти или усилению поправки стоимости.'
problem['updated']=now.isoformat()
problem['cases'].append(dict(date='2026-11-09',component='Видимая опора и согласованное обучение считывателя',fact='FlyVis на обучающих связях 17 своих/1 чужая вместо 17/2; единственное изменение — отказ чужой. Согласованные координаты 17/4. На пяти рабочих запросах маска сохранила все три ошибки. Источники: summary.json, deployment_screen.json, final_head_audit.json.',hypothesis='Улучшения опоры недостаточно для переносимого признака принадлежности. Потеря информации при корреляции и недостаточное обучение остаются разными версиями.',discriminating_check='Выбрать одну проверку между ограничением представления и обучения по накопленному заключению; обычный вход контролирует специфичность сигнала.',limitation='Нет новых полных траекторий или независимого теста. Ошибка ядра не установлена.',evidence=['research_visible_geometry_head_20261109/'+n for n in ['summary.json','deployment_screen_protocol.json','deployment_screen.json','final_head_audit.json']],sha256={n:sha(OUT/n) for n in ['summary.json','deployment_screen.json','final_head_audit.json']}))
atomic(path,(json.dumps(problems,ensure_ascii=False,indent=2)+'\n').encode(),before)
path=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';before=path.read_bytes();nl=b'\r\n' if b'\r\n' in before else b'\n';marker='## Журнал изменений'.encode()+nl
entry='**'+now.strftime('%d.%m.%Y, %H:%M MSK')+'** — Видимая опора не исправила рабочие ошибки\n\n*Получено (`'+identity+'`).* На тех же 26 связях FlyVis: полный шаблон 17 своих/2 чужие, видимая опора 17/1, согласованные координаты 17/4; масштабные контроли 17/2. Единственное изменение маски — отказ чужой связи 193→raw4824; свои не изменились. Источники: research_visible_geometry_head_20261109/summary.json; final_head_audit.json. Обычные пространственный и временной входы не прошли допуск. 240 новых моделей проверены независимо, audit.json.\n\n*Рабочая проверка.* После компонентного допуска, но до окончательных весов зарегистрирован отдельный критерий на пяти известных текущих запросах: сохранить два верных решения и уменьшить три ошибки. Четыре окончательные модели обучены только на 26 обучающих связях; функция потерь, градиент и развёрнутые коэффициенты проверены независимо. Ошибки остались 3→3: свои 232 и 832 по-прежнему отвергнуты, чужая 574 разрешена. Пять преобразований геометрии точно переиспользованы; сеть, видео и полные траектории не повторялись. Источники: deployment_screen_protocol.json; deployment_screen.json; final_head_audit.json.\n\n*Вывод.* Частный выигрыш опоры сохранён, но перенос на конкретные рабочие отказы не подтверждён. Согласованные координаты ухудшили компонентное сравнение. Это не ошибка ядра и не доказательство бесполезности всех локальных шаблонов.\n\n*Решение.* Полный трекер этой версии не запускать. Обновить накопленное заключение и отделить ограничения представления от обучения до следующей замены. Пороги и силу вмешательства по этим примерам не менять; проблема С2 остаётся открытой.\n\n'
assert before.count(marker)==1
atomic(path,before.replace(marker,marker+nl+entry.replace('\n',nl.decode()).encode(),1),before)
with (OUT/'recording.json').open('x',encoding='utf8') as f:json.dump(dict(passed=True,state='completed_negative',component_gain_preserved=True,working_screen_failed=True,no_native_pilot=True,goal_achieved=False,sources={n:sha(OUT/n) for n in ['summary.json','audit.json','deployment_screen_protocol.json','deployment_screen.json','final_head_audit.json','РЕЗУЛЬТАТ.md']}),f,ensure_ascii=False,indent=2)
print('VISIBLE_MASK_RESULT_RECORDED_NO_NATIVE_PILOT',flush=True)
