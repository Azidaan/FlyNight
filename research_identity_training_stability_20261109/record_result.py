"""Independent saved-weight arithmetic and an honest negative sensitivity result."""
import json,hashlib,math,time
from pathlib import Path
from datetime import datetime,timezone,timedelta
ROOT=Path(__file__).resolve().parents[1];OUT=Path(__file__).resolve().parent;TRAIN=ROOT/'research_domain_matched_reader_20261109'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def atomic(path,data,before):
    temp=path.with_suffix(path.suffix+'.stability_result.pending');assert not temp.exists();temp.write_bytes(data)
    for _ in range(30):
        assert path.read_bytes()==before
        try:temp.replace(path);return
        except PermissionError:time.sleep(.1)
    raise PermissionError(str(path))
now=datetime.now(timezone(timedelta(hours=3)));r=read(OUT/'comparison.json');p=read(OUT/'protocol.json');assert r['passed'] and r['new_fits']==0 and r['reused_models']==80
for name,digest in p['sources'].items():assert sha(ROOT/name)==digest,name
models=read(TRAIN/'predictions.json')['folds'];folds={(v['video'],v['model']):v for v in models};deployment=read(ROOT/'runs/domain_matched_native_20261109/deployment.json')
error=0.;count=0
def score(model,norm,c,arm):
    g=[(v-m)/s for v,m,s in zip(c['geometry'],norm['mean'],norm['scale'])]
    x=g if arm=='G7' else g+(c['ready_C34_channels'] if arm=='C34' else c['new_image_features'][arm]['channels'])
    t=model['theta'];return math.fsum(a*b for a,b in zip(t[:-1],x))+t[-1]
for c in r['cases']:
    for arm in ['G7','C34','R134_spatial','R134_temporal']:
        error=max(error,abs(score(deployment['models'][arm],deployment['geometry_normalization'],c,arm)-c['baseline'][arm]));count+=1
        for row in c['fold_predictions']:
            v=row['removed_video'];fold=folds[(v,arm)];value=score(fold['fit'],folds[(v,'G7')]['normalization'],c,arm)
            error=max(error,abs(value-row['scores'][arm]));count+=1
assert count==420 and error<1e-12,(count,error)
assert not any(x['sign_changes'] for arm in ['C34','R134_spatial','R134_temporal'] for x in r['summary'][arm])
with (OUT/'audit.json').open('x',encoding='utf8') as f:json.dump(dict(passed=True,independent_scalar_scores=count,max_error=error,saved_OOF_prediction_error=r['old_prediction_parity_max'],sources={n:sha(OUT/n) for n in ['protocol.json','inventory.json','comparison.json']},source_script_sha256=sha(Path(__file__)),no_fits=True,no_core_or_video=True,goal_achieved=False),f,ensure_ascii=False,indent=2)
report='''# Ошибки не объясняются одним исключаемым обучающим видео

1. Результат

Использованы 80 уже готовых моделей с исключением каждого из 20 видео: FlyVis, два обычных входа и геометрия. На тех же пяти рабочих запросах все 20 вариантов FlyVis сохранили прежние решения. Два своих отказа и разрешённая чужая связь остались. Новых обучений, прогонов ядра, видео или трекера не было. Источник: comparison.json → summary.C34.

Своя 232 на кадре 544: прежний балл −0,426740, диапазон после исключения одного видео [−0,493783; −0,291634]. Своя 832 на кадре 540: −0,398460, диапазон [−0,496099; −0,252794]. Чужая 574 на кадре 937: +0,546382, диапазон [+0,369678; +0,798532]. Все диапазоны остаются на прежней стороне нуля. Источник: comparison.json → summary.C34. Это баллы классификатора, не вероятности и не выданные номера.

2. Контроли и точность

Оба обычных входных варианта также сохранили все пять решений при каждом исключении видео. Их ошибки совпадают с FlyVis лишь частично: оба принимают свою 232, оба отвергают свою 832 и разрешают чужую 574. Геометрическая модель менее устойчива: некоторые свои решения меняются. Источник: comparison.json → summary. Поэтому устойчивость весов сама по себе не означает правильность принадлежности.

Два исключённых видео содержали только одну чужую связь каждое; остальные четыре видео с чужими содержали также свои. Эти эффекты не объединяются под названием «удаление одной чужой». Состав записан в inventory.json → foreign_video_groups. Исключение любого видео также не удаляет все возможные причины общей непредставительности обучения.

Все 104 прежние оценки на исключённых видео воспроизведены до 4,45·10⁻¹⁶; независимая скалярная проверка охватила 420 новых применений готовых весов. Источники: comparison.json → old_prediction_parity_max; audit.json. Точные карты обычного входа проверены по закрытым метаданным; новые карты не создавались.

3. Что это меняет

Не найдено конкретного одиночного обучающего видео, удаление которого исправляло бы три ошибки FlyVis. Поэтому адресное удаление обучающих связей или выбор удачной головы не обоснованы. Нельзя заключать, что шести чужих достаточно, что вся информация потеряна в корреляции или что ошибка находится в ядре. Один общий порог также не становится решением.

Следующая содержательная замена должна сохранять пространственное соответствие отдельных частей объекта, которое исчезает в одной корреляции по всему каналу. До расчёта сопоставить её с прежними четырьмя частями рамки и пространственным считывателем, проверить математику и предусмотреть равные обычные входные и размерностные контроли. Это кандидат, не объявленное исправление.
'''
with (OUT/'РЕЗУЛЬТАТ.md').open('x',encoding='utf8') as f:f.write(report)
identity='identity_training_stability_20261109';path=ROOT/'experiments/project_history.json';before=path.read_bytes();cat=json.loads(before);card=next(x for x in cat['experiments'] if x['id']==identity);assert card['state']=='planned'
updates=dict(obtained='80 готовых моделей применены к пяти прежним рабочим входам; новых обучений 0. Все 20 исключений видео сохранили все решения FlyVis и обоих обычных входных вариантов. Собственные отказы 232/832 и чужая 574 не исправлены. Источник: research_identity_training_stability_20261109/comparison.json → summary. 104 прежние оценки воспроизведены до 4,45·10⁻¹⁶; 420 новых баллов независимо проверены, audit.json.',interpretation='Не подтверждено объяснение этих трёх ошибок зависимостью от одного исключаемого видео. Это не доказывает достаточность всего обучающего пула или потерю информации в признаках. Геометрия меняет некоторые собственные решения, тогда как стабильные сигнальные решения могут оставаться неверными.',decision='Не удалять удобные обучающие связи и не выбирать голову по пяти рабочим запросам. Не запускать полный трекер по этой диагностике. Следующий отдельный кандидат — сравнение соответствующих частей шаблона; сначала алгебра, история прежних пространственных считывателей и равные контроли.',cannot_claim='Ранее просмотренные пять запросов; ни новых ID, ни независимого теста. Готовые исключения четырёх смешанных видео удаляют также свои связи. Устойчивость к одиночному исключению не доказывает общую достаточность обучения или ошибку ядра.')
for field,value in updates.items():
    if card.get(field):card.setdefault('revisions',[]).append(dict(date=now.isoformat(),field=field,old=card[field],reason='Готовые модели применены без переобучения; проверена точность и границы отрицательного вывода.'))
    card[field]=value
card.update(state='completed_negative',done=now.strftime('%Y-%m-%d'))
for item in card['acceptance']:item.update(met=True,evidence='research_identity_training_stability_20261109/comparison.json; audit.json; все предусмотренные ветви результата сообщены, исправлений при исключении нет')
card['evidence'] += [dict(kind='file',ref='research_identity_training_stability_20261109/'+n,claim=t) for n,t in [('comparison.json','Все неизменные и изменённые решения'),('audit.json','Независимая арифметика готовых весов'),('РЕЗУЛЬТАТ.md','Отрицательная проверка объяснения одним видео')]]
atomic(path,(json.dumps(cat,ensure_ascii=False,indent=2)+'\n').encode(),before)
path=ROOT/'experiments/system_problems.json';before=path.read_bytes();problems=json.loads(before);problem=next(x for x in problems['problems'] if x['id']=='С2');problem.setdefault('revisions',[]).append(dict(date=now.isoformat(),field='next_check',old=problem['next_check'],reason='После сводки и ответа Gemini проверено влияние исключения видео; три ошибки устойчивы.'))
problem['next_check']='Следующий кандидат: локальные соответствия частей прошлого и текущего шаблона при прежнем общем сдвиге, без изменения ядра. До расчёта проверить алгебру, сравнить с z13_spatial_bins_gate_v1 и z13_spatial_set_readout_v1, зафиксировать контроль обычного входа, числа признаков и масштаба. Исключение одного из 20 обучающих видео не исправило три известные ошибки; недостаточность всего обучения этим не исключена.';problem['updated']=now.isoformat()
problem['cases'].append(dict(date='2026-11-09',component='Чувствительность решения к составу обучения',fact='Все 20 готовых исключений видео сохранили решения FlyVis и обоих обычных входов на пяти рабочих запросах. Диапазоны двух неверно отвергнутых своих отрицательны, неверно разрешённой чужой — положителен.',hypothesis='Отдельное обучающее видео не объясняет эти решения. Недостаточность всего пула и потеря локальной структуры остаются разными рабочими версиями.',discriminating_check='Содержательная замена представления локальными соответствиями частей с равным обычным и размерностным контролем после проверки алгебры.',limitation='Это точная проверка только одиночных исключений, не доказательство достаточности выборки или причины в ядре. Новых ID нет.',evidence=['research_identity_training_stability_20261109/'+n for n in ['comparison.json','audit.json','РЕЗУЛЬТАТ.md']],sha256={n:sha(OUT/n) for n in ['comparison.json','audit.json']}))
atomic(path,(json.dumps(problems,ensure_ascii=False,indent=2)+'\n').encode(),before)
path=ROOT/'experiments/useful_literature.json';before=path.read_bytes();lit=json.loads(before);lid='koh2017_training_influence';assert all(x['id']!=lid for x in lit['sources'])
lit['sources'].append(dict(id=lid,title='От какого обучающего примера зависит решение — анализ влияния',problem_group='Сохранение личности',method_group='Обучение признака принадлежности',publication='Pang Wei Koh, Percy Liang. Understanding Black-box Predictions via Influence Functions. ICML, PMLR70:1885–1894,2017.',url='https://proceedings.mlr.press/v70/koh17a/koh17a.pdf',sections='Разделы2.1 и2.3.',established='Исследуется изменение предсказания при изменении обучающих данных; влияние отличается от простой близости признаков. Формула через обратную матрицу вторых производных является приближением.',problem='Нужно отличить ошибку считывания от чувствительности общей головы к отдельным обучающим видео.',application='Источник помог отвергнуть центроид как доказательство влияния и выбрать применение уже готовых точных моделей с исключением видео. На пяти рабочих запросах решения FlyVis устойчивы, новых обучений нет.',limitation='Взята постановка вопроса, а не приближённый алгоритм статьи. Групповое исключение видео отличается от одного примера; устойчивость не доказывает достаточность пула и не сохраняет ID.',evidence=['research_identity_synthesis_20261109/ПРОВЕРКА_GEMINI_v18.md','research_identity_training_stability_20261109/comparison.json','research_identity_training_stability_20261109/audit.json'],related=['sys:С2']))
atomic(path,(json.dumps(lit,ensure_ascii=False,indent=2)+'\n').encode(),before)
path=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';before=path.read_bytes();nl=b'\r\n' if b'\r\n' in before else b'\n';marker='## Журнал изменений'.encode()+nl
entry='**'+now.strftime('%d.%m.%Y, %H:%M MSK')+'** — Исключение обучающего видео не исправляет три ошибки принадлежности\n\n*Получено (`'+identity+'`).* Применены 80 готовых голов, новых обучений 0. На пяти прежних рабочих запросах все 20 исключений видео сохранили все решения FlyVis и обоих обычных входов. Своя 232: диапазон [−0,493783; −0,291634], своя 832: [−0,496099; −0,252794], чужая 574: [+0,369678; +0,798532]. Источник research_identity_training_stability_20261109/comparison.json. Геометрия меняет отдельные собственные решения. Два исключения содержат только одну чужую связь, четыре — также свои.\n\n*Проверки.* 104 готовые оценки воспроизведены до 4,45·10⁻¹⁶; 420 новых применений весов независимо проверены скалярной суммой, audit.json. Карты, ядро, видео, прежние обучения и полный трекер не повторялись. Полезный первоисточник Koh и Liang2017 сохранён с точной ролью в реестре литературы; указанное Gemini требование 30–50 связей не принимается как доказанное.\n\n*Вывод.* Нет основания удалять одно обучающее видео или выбирать удачную голову. Устойчивые неверные решения не доказывают достаточность всей выборки и не устанавливают ошибку ядра.\n\n*Решение.* Сохранить отрицательную проверку одиночного влияния. Следующий отдельный кандидат считывателя должен сравнивать соответствующие части шаблона: сначала проверить математику и отличие от прежних пространственных опытов, затем равные обычные и размерностные контроли. Порог и общий бонус по пяти случаям не менять.\n\n'
assert before.count(marker)==1;atomic(path,before.replace(marker,marker+nl+entry.replace('\n',nl.decode()).encode(),1),before)
with (OUT/'recording.json').open('x',encoding='utf8') as f:json.dump(dict(passed=True,state='completed_negative',goal_achieved=False,new_fits=0,no_native_pilot=True,sources={n:sha(OUT/n) for n in ['comparison.json','audit.json','РЕЗУЛЬТАТ.md']}),f,ensure_ascii=False,indent=2)
print('STABLE_WRONG_SIGNS_RECORDED',count,error,flush=True)
