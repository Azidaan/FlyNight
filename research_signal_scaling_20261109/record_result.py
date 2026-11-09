"""Record the development gain and failed operational screen without hiding either."""
import json,hashlib,time
from pathlib import Path
from datetime import datetime,timezone,timedelta
ROOT=Path(__file__).resolve().parents[1];OUT=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def atomic(path,data,before):
    temp=path.with_suffix(path.suffix+'.scale_result.pending');assert not temp.exists();temp.write_bytes(data)
    for _ in range(30):
        assert path.read_bytes()==before
        try:temp.replace(path);return
        except PermissionError:time.sleep(.1)
    raise PermissionError(str(path))
s=read(OUT/'summary.json');a=read(OUT/'audit.json');screen=read(OUT/'deployment_screen.json');final=read(OUT/'final_head_audit.json');u=read(OUT/'uncertainty.json')
assert s['passed'] and a['passed'] and final['passed'] and not screen['any_gate']
for name in ['protocol.json','deployment_screen_protocol.json']:
    for n,h in read(OUT/name)['sources'].items():assert sha(ROOT/n)==h,n
old=read(ROOT/'research_domain_matched_reader_20261109/predictions.json')['records'];new=read(OUT/'predictions.json')['records'];changes={}
for arm in ['C34','R134_spatial','R134_temporal']:
    changes[arm]=[]
    for x,y in zip(old,new):
        assert all(x[k]==y[k] for k in ['video','frame','old_id','raw_id','label'])
        if (x['scores'][arm]>0)!=(y['scores'][arm+'__common']>0):changes[arm].append(dict(**{k:x[k] for k in ['video','frame','old_id','raw_id','label']},old=x['scores'][arm],new=y['scores'][arm+'__common']))
with (OUT/'decision_changes.json').open('x',encoding='utf8') as f:json.dump(dict(passed=True,common_scale=changes),f,ensure_ascii=False,indent=2)
report='''# Общий масштаб помог обучающей проверке, но не исправил рабочие ошибки

1. Итог

Конкретная замена шкалы не проходит к новому полному трекеру. На тех же пяти рабочих запросах ни одно решение не изменилось. У FlyVis остаются три ошибки: отказ двум своим машинам и разрешение чужой. Оба обычных входа сохраняют по две ошибки. Источник: deployment_screen.json → records, counts, gates.

2. Что всё же измерено

На исключении обучающего видео FlyVis сохранил 17 принятых своих из 20 и уменьшил число принятых чужих с 2 до 1 из 6. Изменился только отказ чужой связи 490 на кадре844 видео02a70c1f-6f18e57f. Обычный пространственный вход: 17/3 → 17/2; временной при общем масштабе: 16/4 → 16/2. Источники: summary.json; decision_changes.json. У пространственного и временного входов одинаковые итоговые числа своих скрывают взаимные исправления и новые отказы; все изменения сохранены поштучно.

Поканальный вариант не превзошёл общий множитель по заранее записанному правилу. Поэтому выбран общий масштаб для всех трёх сигналов. Прибавка сбалансированной точности FlyVis равна 0,08333, или 10,99% относительно прежних 0,75833; описательный95%интервал [0;0,25] включает ноль. Из4000перевыборок4не содержали оба класса. Это не подтверждённая прибавка и не независимая проверка. Источники: summary.json; uncertainty.json.

3. Рабочие случаи

Своя232: балл FlyVis −0,426740 → −1,346407; своя832: −0,398460 → −1,421700; чужая574: +0,546382 → +1,499321. Порог0 неизменен. Знак всех трёх ошибок сохранился, величина усилилась. Правильные решения по своим70и1102 тоже сохранились. Положительный балл1102 не означает выданный ID: общая конкуренция остаётся отдельным этапом. Источник: deployment_screen.json; исходная судьба ID — runs/domain_matched_native_20261109/previous_cases_followup.json.

4. Проверка исполнения и границы

Проверены120моделей с исключением видео и3итоговые модели. Максимальная независимая невязка обучающих моделей1,049·10⁻⁹, итоговых6,883·10⁻¹¹; развёртывание шкал в исходные коэффициенты и15новых оценок проверены с ошибкой менее2·10⁻¹⁵. Источники: audit.json; final_head_audit.json. Новых карт, прогонов ядра и видео нет, старые сохранённые модели не переобучались.

Сбой на100итерациях сохранён. Как минимум24новыхнесохранённыхмодели пришлось восстановить; это не повтор старых готовых результатов. Добавлено поштучное сохранение. Единственная модель, потребовавшая возврата состояния на пределе, имела невязку1,049·10⁻⁹ и укладывалась в прежний допуск независимого аудита1e−7; функция обучения и научные параметры не менялись. Отдельный сбой кодировки очереди произошёл до её дочернего запуска. Источники: recovery_protocol.json; recovery_v2_protocol.json; precision_recovery_result.json.

Масштаб действительно меняет эффективный штраф, но его исправление в этой постановке не объяснило и не устранило три рабочие ошибки FlyVis. Из этого не следует отсутствие информации в картах или дефект ядра. Источники: math_checks.json; deployment_screen.json. Цель сохранения личности открыта. Доля достигнутого потолка неизвестна; новых ID/HOTA не измерялось.

5. Следующее различающее действие

Не продолжать подбор масштаба, порога или штрафа на этих случаях. Перед новыми картами проверить доступность и роль трёх ранних путей L1/L2/L3 относительно уже читаемых34типов, сопоставить прежние ранние считыватели и обычный вход. Рассматривать это как кандидата на замену представления, а не как найденную причину. Новый научный протокол возможен только после первоисточников и математической проверки контролей; эта запись не заявляет новый запуск.
'''
with (OUT/'РЕЗУЛЬТАТ.md').open('x',encoding='utf8') as f:f.write(report)
now=datetime.now(timezone(timedelta(hours=3)));identity='signal_scale_identity_20261109';path=ROOT/'experiments/project_history.json';before=path.read_bytes();cat=json.loads(before);card=next(c for c in cat['experiments'] if c['id']==identity);assert card['state']=='planned'
updates=dict(obtained='На26разработческихсвязях общий масштаб: FlyVis17своих/1чужая вместо17/2, пространственныйвход17/2вместо17/3, временной16/2вместо16/4. Источник research_signal_scaling_20261109/summary.json. На5рабочихзапросах ни одного изменённого решения: FlyVis3ошибки, обаобычныхвхода по2; deployment_screen.json. Аудиты120+3моделей пройдены: audit.json; final_head_audit.json. Интервал C34[0;0,25]включаетноль: uncertainty.json.',interpretation='Общее усиление сигнала меняет решения обучающей проверки, но не исправляет рабочие отказы. Поканальная шкала не превосходит общий масштаб по правилу. Два отрицательных собственных и положительный чужой баллы FlyVis усилились; масштаб не признан причиной этих ошибок. Подтверждённого выигрышаIDнет.',decision='Не запускать полный трекер этой версии и не подбирать масштабы/пороги/штраф. Следующий кандидат представления: проверить целевые ранниеL1/L2/L3, их доступность и отличие от прежних ранних считывателей; сначала первоисточники и математика согласованного контроля, без объявления дефекта ядра.',cannot_claim='Теже26разработческихсвязей и5ранее просмотренныхзапросов. Нет независимого теста, новыхID/HOTA, подтверждённой пользы FlyVis или поканальной нормировки. Сбой потерял минимум24новыхнесохранённыхмодели; они восстановлены с прежними научными условиями, старые готовые модели/карты не повторены. Нельзя заключать об отсутствии информации в картах или ошибке ядра.')
for field,value in updates.items():
    if card.get(field):card.setdefault('revisions',[]).append(dict(date=now.isoformat(),field=field,old=card[field],reason='Завершены сравнение шкал и отдельно зарегистрированный до новых весов рабочий контроль.'))
    card[field]=value
card.update(state='completed_negative',done=now.strftime('%Y-%m-%d'))
for item in card['acceptance']:item.update(met=True,evidence='research_signal_scaling_20261109/summary.json; audit.json; uncertainty.json')
card['acceptance'].append(dict(criterion='Дописан послеCVдо3итоговыхмоделей: на5заранее просмотренныхрабочихзапросах сохранить все прежние правильные решения и строго уменьшить ошибки хотябыодного сигнала. Причина — проверить исходный механизм до дорогого трекера; deployment_screen_protocol.json.',met=False,evidence='research_signal_scaling_20261109/deployment_screen.json → gates, counts'))
card['evidence'] += [dict(kind='file',ref='research_signal_scaling_20261109/'+n,claim=t) for n,t in [('summary.json','Разработческий результат всех шести вариантов'),('decision_changes.json','Исправления и новые отказы поштучно'),('uncertainty.json','Неподтверждённость разработческого эффекта'),('deployment_screen_protocol.json','Критерий до итоговых весов'),('deployment_screen.json','Ни одно рабочее решение не исправлено'),('final_head_audit.json','Независимая проверка итоговых весов и оценок'),('РЕЗУЛЬТАТ.md','Итог и ограничения')]]
atomic(path,(json.dumps(cat,ensure_ascii=False,indent=2)+'\n').encode(),before)
path=ROOT/'experiments/system_problems.json';before=path.read_bytes();problems=json.loads(before);problem=next(x for x in problems['problems'] if x['id']=='С2');problem.setdefault('revisions',[]).append(dict(date=now.isoformat(),field='next_check',old=problem['next_check'],reason='Оба масштаба проверены; рабочие ошибки не исправлены.'))
problem['next_check']='Не подбирать шкалу дальше. Проверить целевые ранниеL1/L2/L3карты как отдельную замену представления: отличие от34выходныхтипов и прежних ранних считывателей, первичная литература, общая причинная история и математика контроля по обычному входу до регистрации нового опыта. Возможная польза не доказана; дефект ядра не установлен.';problem['updated']=now.isoformat()
problem['cases'].append(dict(date='2026-11-09',component='Шкала корреляций перед регуляризованным обучением',fact='Общий масштаб далC3417своих/1чужая вместо17/2 в исключении видео, но на5рабочихзапросах ни одного изменения: обе свои232/832отвергаются, чужая574разрешается. Обычные входы также сохранили свои рабочие ошибки.',hypothesis='Шкала изменяет эффективный штраф, но её согласованная замена не устраняет рабочие ошибки; причина в представлении либо покрытии обучения остаётся рабочей версией.',discriminating_check=problem['next_check'],limitation='Нет новыхID/HOTA; интервал разработческой прибавки включаетноль. Не доказан дефект ядра или отсутствие информации в картах.',evidence=['research_signal_scaling_20261109/'+n for n in ['summary.json','audit.json','deployment_screen.json','final_head_audit.json']],sha256={n:sha(OUT/n) for n in ['summary.json','audit.json','deployment_screen.json','final_head_audit.json']}))
atomic(path,(json.dumps(problems,ensure_ascii=False,indent=2)+'\n').encode(),before)
path=ROOT/'experiments/useful_literature.json';before=path.read_bytes();lit=json.loads(before);assert not any(x['id']=='sklearn_feature_scale_regularization' for x in lit['sources'])
lit['sources'].append(dict(id='sklearn_feature_scale_regularization',title='Шкала признаков и штраф обучения — документация scikit-learn',problem_group='Сохранение личности',method_group='Обучение признака принадлежности',publication='The scikit-learn developers. Importance of Feature Scaling. Официальная документация; проверено09.11.2026.',doi='Не применяется: официальная документация программной библиотеки.',url='https://scikit-learn.org/stable/auto_examples/preprocessing/plot_scaling_importance.html',sections='Effect of rescaling on model performance: зависимость регуляризации от шкалы; последнее предупреждение об усилении шума. В примере присутствуетPCA, наша модель его не использует.',established='Пример демонстрирует изменение обученной модели и выбранной силы регуляризации при смене масштаба; непредсказательные малые признаки после усиления могут увеличить переобучение.',problem='С2 — неверный выбор своего и чужого продолжения.',application='Помог сформулировать проверку эффективного штрафа и контроль общего усиления вместо предположения, что всякая стандартизация полезна. Точная алгебра для нашей модели проверена отдельно в research_signal_scaling_20261109/math_checks.json. Общее усиление и поканальное деление сравнены при фиксированных условиях; рабочие ошибки не устранены, deployment_screen.json.',limitation='Это методическое основание, не исследование идентичности и не подтверждение пользыFlyVis. ЧислаPCAизпримера не переносятся. Эксперимент нашего считывателя отрицательный по рабочему критерию.',experiments=[identity],evidence=['research_signal_scaling_20261109/math_checks.json','research_signal_scaling_20261109/summary.json','research_signal_scaling_20261109/deployment_screen.json']))
atomic(path,(json.dumps(lit,ensure_ascii=False,indent=2)+'\n').encode(),before)
path=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';before=path.read_bytes();nl=b'\r\n' if b'\r\n' in before else b'\n';marker='## Журнал изменений'.encode()+nl
entry='**'+now.strftime('%d.%m.%Y, %H:%M MSK')+'** — Шкала признаков не исправила рабочие решения принадлежности\n\n*Получено (`'+identity+'`).* На26связях C3417своих/1чужая вместо17/2; общий множитель выбран заранее записанным правилом. На5рабочихзапросах ни одно решение ни одного из3сигналов не изменилось: C34триошибки, входыподве. Своя232−0,426740→−1,346407; своя832−0,398460→−1,421700; чужая574+0,546382→+1,499321. Источники research_signal_scaling_20261109/summary.json; deployment_screen.json. Независимый аудит3итоговыхмоделей и15новыхоценок прошёл, final_head_audit.json.\n\n*Вывод.* Общая шкала меняет некоторые разработческие решения, но не исправляет исходный рабочий механизм. Прибавка C34сбалансированнойточности0,08333,10,99%кпрежней, описательный95%[0;0,25]включаетноль. Нет новыхID/HOTA или независимого подтверждения. Поканальное деление не доказало отдельной пользы. Все исправления и новые отказы обычных входов сохранены поштучно, decision_changes.json.\n\n*Решение.* Карточкаcompleted_negativeпо отдельно зарегистрированному до итоговых весов рабочему критерию. Новый полный трекер не запускать; не подбирать шкалы/пороги. Для следующего кандидата сначала проверить целевые ранниеL1/L2/L3, прежние ранние считыватели, первоисточники и математику контроля обычного входа. Источник не считается причиной заранее. Проверенная документацияscikit-learn сохранена в useful_literature.json с точной ролью и ограничениями. Общая цель открыта.\n\n'
assert before.count(marker)==1;atomic(path,before.replace(marker,marker+nl+entry.replace('\n',nl.decode()).encode(),1),before)
with (OUT/'recording.json').open('x',encoding='utf8') as f:json.dump(dict(passed=True,state='completed_negative',no_native_pilot=True,goal_achieved=False,sources={n:sha(OUT/n) for n in ['summary.json','audit.json','decision_changes.json','deployment_screen.json','final_head_audit.json','РЕЗУЛЬТАТ.md']}),f,ensure_ascii=False,indent=2)
print('SCALING_RESULT_RECORDED_NEGATIVE_OPERATIONAL_SCREEN',flush=True)
