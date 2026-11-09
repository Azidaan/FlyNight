"""Record the coordinate result and the newly measured training-domain gap."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import json,hashlib,subprocess
ROOT=Path(__file__).resolve().parents[1];OUT=Path(__file__).resolve().parent
MSK=timezone(timedelta(hours=3));now=datetime.now(MSK);head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip()
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def replace(path,data,before):
    temp=path.with_suffix(path.suffix+'.gate.pending');assert not temp.exists();temp.write_bytes(data);assert path.read_bytes()==before;temp.replace(path)
def update(path,obj,before):replace(path,(json.dumps(obj,ensure_ascii=False,indent=2)+'\n').encode(),before)
def revise(item,field,new,reason):
    old=item.get(field)
    if old==new:return
    if old:item.setdefault('revisions',[]).append(dict(date='2026-11-09',commit=head,field=field,old=old,reason=reason))
    item[field]=new
assert read(OUT/'coordinate_report.json')['passed'] and read(OUT/'implementation_check.json')['passed']
r=read(OUT/'training_states_complete.json');assert r['geometry_valid']==6 and r['geometry_valid_foreign']==0
assert (ROOT/'research_working_domain_identity_20261109/protocol.json').exists()
facts='На 213 готовых запросах C34 преобразования различаются в 207 случаях; медиана типичного расхождения 2,202579 шага сетки. Геометрически выровненная видимая опора от трёх ячеек есть во всех 213; в двух ошибочных своих отказах — 94 и 66. Источник: research_geometry_consistency_20261109/coordinate_report.json → summary, known. Обратимость проверена до реальных координат, coordinate_math.json; экспериментальный переход прошёл 7 аналитических тестов, implementation_check.json.\n\nВосстановлены недостающие состояния 91 обучающей пары на 32 видео с точным совпадением 20 488 кадров и 182 487 публичных строк. Необходимый геометрический допуск проходит только 6 своих пар и 0 чужих; прежняя выборка содержит 15 своих и 76 чужих. Источник: training_states_complete.json → geometry_valid_own, geometry_valid_foreign, exact_frames, exact_rows. Это только необходимая геометрическая часть полного рабочего допуска: все остальные условия могут дополнительно сократить шесть своих.'
decision='Не обучать и не развёртывать новую координатную голову на шести своих без чужих примеров рабочего допуска. Кандидат координатного чтения математически реализован, но не принят. Сначала восстановить из существующих 84 обучающих видео примеры точно в рабочей области нативного модуля; обычный вход и FlyVis должны получить один набор и одни условия. Условия до нового отбора: research_working_domain_identity_20261109/protocol.json. Пороги и ядро не менять.'
path=ROOT/'experiments/project_history.json';before=path.read_bytes();catalog=json.loads(before)
card=next(c for c in catalog['experiments'] if c['id']=='geometry_coordinate_gate_20261109')
card.update(state='inconclusive',done='2026-11-09',obtained=facts,
    interpretation='Различие координат измерено и может быть существенным, но физическая правильность выбранного масштаба и его роль в отказах не доказаны. Обнаружен более ранний разрыв: чужие обучающие связи не покрывают необходимую геометрическую область применения головы. Этот факт не доказывает, что правильный отбор сам исправит ID.',decision=decision)
for a in card['acceptance']:a.update(met=True,evidence='research_geometry_consistency_20261109/coordinate_math.json; coordinate_report.json; implementation_check.json')
card['evidence'] += [dict(kind='file',ref='research_geometry_consistency_20261109/'+f,claim=claim) for f,claim in [
    ('coordinate_report.json','Все213координатныхзапросов и явно известные личности'),('training_states_complete.json','Точные публичные свидетели восстановленных состояний и ноль чужих внутри необходимого допуска'),('training_state_protocol_v3.json','Условия восстановления, исходники и техническое исправление')]]
identity='working_domain_identity_inventory_20261109';assert all(c['id']!=identity for c in catalog['experiments'])
catalog['experiments'].append(dict(id=identity,title='Обучающие свои и чужие — точный рабочий допуск трекера',area='flyvis',state='planned',started='2026-11-09',
    question='Есть ли в существующих обучающих видео подтверждённые чужие и свои продолжения, действительно предъявляемые текущему считывателю?',
    hypothesis='Прежние чужие примеры расположены за пределами рабочего геометрического допуска. Нативный отбор до меток может дать более релевантные примеры. Альтернатива — прямой разметки внутри допуска недостаточно; тогда бинарная голова на этих данных не готова.',
    expected='Только восстановление недостающего контекста на прежних 84 fit84 видео, точная сверка готовых нативных выходов и прямые метки. Нет видео, сетчатки, ядра, обучения, сходства и метрик.',
    acceptance=[dict(criterion='Все сохранённые нативные публичные массивы воспроизводятся точно; источники неизменны; GT не выбирает рамку, геометрию или ячейки; неизвестные сохранены.',met=None),dict(criterion='Среди прямо размеченных рабочих связей оба класса сохраняются в обучении после исключения каждого полного видео. Это только обучаемость, не достаточность выборки.',met=None)],
    obtained='',interpretation='',decision='',cannot_claim='Нет нового независимого теста или решения ID; фит84 ранее использовался другими считывателями. В этот новый рабочий пул входят и ранее просмотренные обучающие видео: весь пул является разработческим, а20valотдельны.',
    evidence=[dict(kind='file',ref='research_working_domain_identity_20261109/protocol.json',claim='Отбор и критерии до новых данных'),dict(kind='file',ref='research_geometry_consistency_20261109/training_states_complete.json',claim='Причина необходимости нового контекста')],tags=['fly-problem-P1']))
update(path,catalog,before)
path=ROOT/'experiments/obsidian_experiment_topics.json';before=path.read_bytes();topics=json.loads(before)
topics['topics'][identity]=['Сохранение личности','Выбор продолжения среди рамок','Рабочая область обучения']
update(path,topics,before)
path=ROOT/'experiments/system_problems.json';before=path.read_bytes();problems=json.loads(before);s=next(x for x in problems['problems'] if x['id']=='С2')
revise(s,'fact',s['fact']+'\n\n'+facts+' Актуальное заключение: research_identity_synthesis_20261109/СВОДКА_v11.md.','Новая координатная проверка и точное восстановление рабочих состояний обучения.')
revise(s,'hypotheses',[
    'Подтверждённое ограничение данных: ни одна из76чужих обучающих связей не проходит необходимый геометрический допуск. Рабочая причина: голова переносится с чужих вне допуска на чужие внутри него. Причинное объяснение провала ещё не установлено.',
    'Подтверждено координатами: в207из213нативныхзапросов геометрия допуска отличается от чтения карт. Ошибка соответствия участков остаётся рабочей версией; сам прогноз полного размера также может быть неверен.',
    'Сильнейшая альтернатива: признаки или обученные веса не различают экземпляры даже при согласованной геометрии и рабочем обучающем пуле. Ядро не признано дефектным.',
    'Конкуренция и удаление дубликатов — отдельные измеренные потери номера. Численная граница0,8 и общий сырой уровень каналов не объясняют оба отказа.'],'Уточнение ранга причин по новым данным; прежние версии сохранены.')
revise(s,'next_check',decision,'Координаты восстановлены; бинарная модель не готова без рабочего чужого класса.')
s['cards'] += [x for x in ['geometry_coordinate_gate_20261109',identity] if x not in s['cards']]
s['updated']=now.isoformat()
s['cases'].append(dict(date='2026-11-09',component='Согласованность чтения и рабочей области обучения',fact=facts,
    hypothesis='Геометрия чтения и распределение обучающих связей расходятся с рабочим допуском. Это два проверяемых механизма, не единая доказанная причина.',
    discriminating_check=decision,limitation='207из213нативныхличностейнеизвестны; исходные скрытые состояния обучения не сохранены для прямого сравнения, но публичный выход/области/референсы/стоимости проверены точно.',
    evidence=[dict(path='research_geometry_consistency_20261109/coordinate_report.json',sha256=sha(OUT/'coordinate_report.json')),dict(path='research_geometry_consistency_20261109/training_states_complete.json',sha256=sha(OUT/'training_states_complete.json'))]))
update(path,problems,before)
text='''# Сохранение личности: координаты и рабочая область обучения

1. Цель и последний полный результат

Нужно вернуть прежний ID своей машине и отвергнуть чужую. Полный нативный пилот20прежнихvalвидео не подтвердил прибавку34каналовFlyVis к геометрии или двум обычным входным контролям. Версия не принята; ядро не обучалось и его ошибка не установлена. Первичный итог: runs/full_template_native_20261109/РЕЗУЛЬТАТ.md. Прежние версии1–10сохранены.

2. Что выяснено о координатах

'''+facts+'''

3. Что это меняет в объяснении

Нельзя считать91историческуюпару проверкой различения чужих в том же геометрическом допуске, в котором голова работала онлайн: ни одна чужая не проходит даже необходимую геометрическую часть. Готовые числа и межвидеовая проверка остаются правильными для своего прежнего пула; их нельзя переносить на другой условный пул. Это ограничение прежнего вывода, не отмена расчётов и не доказанная причина провала.

Если оставить лишь шесть своих, свободный член регуляризованной логистической модели уходит к +∞ при нулевых весах: для всех y=1 производная по нему отрицательна при любом конечном значении. Поэтому прежняя бинарная процедура с двумя классами на этом условном наборе не определена как конечная обученная голова. Наличие76чужих вне допуска не создаёт наблюдений чужих внутри допуска. Это конечная выборка; нулевая частота не означает нулевую вероятность чужих в природе.

4. Следующее действие и условие остановки

'''+decision+'''

Новый контекст нужен для полного причинного рабочего отбора, а не для повторения готовой метрики. Существующие84обучающихвидео и детекции остаются прежними; весь этот пул разработческий. Готовые базовые выходы сверяются точно, новые выходы для оценки и метрики не создаются. Оба класса должны оставаться в обучении после исключения каждого полного видео. Если это условие не выполнено, не запускать новую бинарную голову и не ослаблять пороги по увиденному результату.

5. Что не следует смешивать

Согласованность координат не доказывает правильный физический масштаб. Более релевантное обучение не доказывает достаточность признака. Правильное разрешение связи не доказывает победу в общем назначении или публичное сохранение номера. Численная граница и общий уровень каналов ранее проверены и не являются готовым исправлением. Следующая реализация требует одинаковых примеров, геометрии и условий для FlyVis и обычного входа, затем фактической судьбы ID.
'''
dest=ROOT/'research_identity_synthesis_20261109/СВОДКА_v11.md';assert not dest.exists();dest.write_text(text,encoding='utf8')
path=ROOT/'research_identity_synthesis_20261109/АКТУАЛЬНОЕ.md';before=path.read_bytes();replace(path,('# Актуальное накопленное заключение\n\n[Версия11 — координаты и рабочая область обучения](СВОДКА_v11.md).\n\nПолный пилот отрицательный. Координатный кандидат проверен математически, но не развёрнут: в91обучающейпаре нет чужих, проходящих необходимый геометрический допуск. Следующий зарегистрированный шаг — контекст точного рабочего отбора на прежних84обучающихвидео. Прежние версии1–10сохранены.\n').encode(),before)
path=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';before=path.read_bytes();nl=b'\r\n' if b'\r\n' in before else b'\n';marker='## Журнал изменений'.encode()+nl
entry='**'+now.strftime('%d.%m.%Y, %H:%M MSK')+'** — Координатная проверка выявила разрыв рабочего обучения\n\n'
entry+='*Получено (`geometry_coordinate_gate_20261109`).* '+facts+'\n\n*Вывод.* Согласованная геометрия математически реализована, но её пользу личности не проверяли: рабочий чужой класс в91прежнейпаре отсутствует. Корневая причина провала не доказана.\n\n*Решение.* '+decision+'\n\n*Постановка (`'+identity+'`).* Только недостающий контекст нативного отбора на84прежнихfit84видео; все готовые публичные выходы сверяются точно, метрики и изображения не повторяются. GTпосле отбора, неизвестные сохраняются. Прежний32видеоконтекст охватывал только91старыйзапрос и не содержит нового рабочего пула. Техническая ошибка APIпервого экспорта сохранена в протоколахv1–v3; успешное восстановление отдельно связано с исходниками. Раздел7плана не меняется.\n\n'
replace(path,before.replace(marker,marker+nl+entry.replace('\n',nl.decode()).encode(),1),before)
with (OUT/'recording.json').open('x',encoding='utf8') as f:json.dump(dict(recorded=now.isoformat(),head=head,version=11,next_card=identity,goal_achieved=False),f,indent=2)
print('RECORDED_COORDINATE_GATE_AND_NEXT_DOMAIN_ACTION',flush=True)
