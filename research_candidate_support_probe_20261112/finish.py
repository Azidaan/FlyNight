from common import *
import shutil,psutil
a=read(OUT/'audit.json');s=read(OUT/'scores.json');assert a['passed'] and not a['primary_any']
changes=[]
for r in s['records']:
    base=next(x for x in s['records'] if x['old_id']==r['old_id'] and x['representation']==r['representation'] and x['head']==r['head'] and x['mode']=='full')
    if r['accepted']!=base['accepted']:changes.append(r)
assert not changes
guardpath=Path(read(OUT/'execution_start.json')['supervisor']['supervisor_report']);guard=read(guardpath);assert guard['returncode']==0 and guard['reason'] is None and not guard['stop_errors'];owned=[]
for pid,birth in guard['owned'].items():
    try:same=psutil.Process(int(pid)).create_time()==birth
    except psutil.NoSuchProcess:same=False
    assert not same;owned.append(dict(pid=int(pid),create_time=birth,ended_or_PID_reused=True))
shutil.copy2(guardpath.with_name('memory_guard.worker.log'),OUT/'worker.log');save('execution_receipt.json',dict(passed=True,guard=guard,process_checks=owned,all_owned_ended=True));rel=OUT.relative_to(ROOT).as_posix()
get=lambda h,m:next(r['score'] for r in s['records'] if r['old_id']==574 and r['representation']=='C34' and r['head']==h and r['mode']==m)
effects={h:dict(full=get(h,'full'),detector=get(h,'detector'),detector_and_annotation=get(h,'detector_and_annotation'),first_delta=get(h,'detector')-get(h,'full'),second_delta=get(h,'detector_and_annotation')-get(h,'detector'),total_delta=get(h,'detector_and_annotation')-get(h,'full'),positive_score_reduction_percent=100*(get(h,'full')-get(h,'detector_and_annotation'))/get(h,'full')) for h in ['native','agreement']}
save('effects.json',dict(passed=True,effects574=effects,decision_changes=changes,source_scores_sha256=sha(OUT/'scores.json'),warning='Score reduction is a paired intervention effect with recentering/renormalization, not additive per-pixel attribution and not tracking-quality improvement.'))
obtained=('На пяти случаях воспроизведены все 30 исходных оценок трёх представлений и двух сохранённых решающих правил. Независимо сверены 120 основных оценок и 7284 пересчёта контролей одинаковой численности, всего 7404; это не 7404 независимых наблюдения. '
 'У чужого продолжения 574 исходная корреляционная голова: +0,546382 → +0,487280 после исключения 22 точек за рамкой детектора → +0,399734 после дополнительного исключения 28 точек вне разметки кандидата. '
 'Голова согласия: +1,843682 → +1,609828 → +1,261573. Ни одно решение во всех трёх представлениях и обеих головах не изменилось; основной критерий не пройден. '
 'Все 120 циклических контролей каждого из двух размеров (98 и70 ячеек) также оставили574 разрешённым обеими головами C34. Источники '+rel+'/baseline_gate.json, scores.json, audit.json, effects.json.')
interpret=('Точки вне рамки действительно увеличивают ложный балл574: их исключение при неизменных остальных условиях уменьшило обе оценки. Однако это не достаточная причина ошибочного разрешения: оно сохраняется и после удаления обеих групп. '
 'Поэтому прежнее описание расширенной выборки уточнено количественным вмешательством, а само расширение не объявляется корнем отказа. Остаточное сходство и решающее правило внутри оставшейся области требуют отдельного объяснения. '
 'Контроли одинаковой численности показывают устойчивость знака в проверенном семействе подмножеств, но не являются статистическим доказательством для всех возможных масок.')
decision=('Не внедрять обрезку как исправление узнавания и не подбирать порог после уменьшения балла. Сохранить измеренный вклад двух групп. Следующий различающий вопрос — какие пространственные компоненты оставшейся области дают сходство разных объектов и чем они отличаются от соответствий своих пар; проверять сохранённые признаки до нового обучения ядра.')
cannot=('Точная причина всех отказов не установлена. Разметка рамок не отделяет каждый пиксель объекта от перекрывающего его объекта или фона; проба с разметкой является диагностикой с известным ответом. '
 'Число ячеек в геометрических признаках оставлено исходным для изоляции визуального вклада; это не готовое внедрение маски в трекер. Нельзя объявлять уменьшение положительного балла улучшением качества. Нового обучения, нейронных карт, вызовов ядра, полного трекера и изменения порогов нет.')
lines=['# Вклад точек за границами кандидата','','1. Результат','',obtained,'',interpret,'',decision,'',cannot,'','2. Вмешательство и проверка','',
 'После воспроизведения всех исходных оценок из сравниваемых участков удалялись одинаковые строки: сначала точки за рамкой детектора, затем дополнительно точки вне аннотированной рамки кандидата. Координаты оставшихся точек, карты, веса, сдвиги и геометрические признаки не менялись. Средние и нормировки сходства пересчитывались на оставшихся точках, поэтому эффект нельзя трактовать как сумму независимых вкладов удалённых пикселей.','',
 'Для каждого числа оставшихся точек k проверены все N циклических подмножеств длины k в сохранённом порядке N исходных ячеек. Подмножества перекрываются; они контролируют численность, но не полностью повторяют форму маски. При k=N часть контролей тождественна. Наилучшая маска по результату не выбиралась.','',
 'Независимый оценщик восстановил выборку по явным неравенствам рамок, проверил карту другим интерполятором, корреляцию через ковариацию, согласие через средний квадрат различий и итоговые суммы. Максимальная ошибка оценки '+str(a['max_score_error'])+'. Исходные карты, коэффициенты и разметка общие.','',
 '3. Все основные оценки FlyVis','']
for r in s['records']:
    if r['representation']=='C34':lines += [f"Номер {r['old_id']}, {r['head']}, {r['mode']}, {r['count']} ячеек: {r['score']:.9f}; {'принят' if r['accepted'] else 'отвергнут'}.",'']
lines += ['4. Контроли случая 574','']
for c in a['control_summaries']:
    if c['old_id']==574 and c['representation']=='C34':lines += [f"{c['head']}, {c['count']} ячеек: диапазон контролей {c['control_min']:.9f} … {c['control_max']:.9f}, медиана {c['control_median']:.9f}; отвергнуто {c['control_reject_count']} из {c['defined_controls']}. Оценка фактической маски {c['actual_score']:.9f}.",'']
lines += ['5. Основание и область вывода','',
 'Galoogahi, Fagg, Lucey, 2017, Learning Background-Aware Correlation Filters for Visual Tracking: https://openaccess.thecvf.com/content_iccv_2017/html/Galoogahi_Learning_Background-Aware_Correlation_ICCV_2017_paper.html. Работа обосновывает необходимость различать цель и окружение при сопоставлении. Здесь алгоритм BACF не реализован; выполнялось диагностическое ограничение готовой выборки, а не обучение фильтра.','',
 'Число правильных решений FlyVis относительно исходного условия: изменение 0 %, исправлено 0 из 3 ошибок. Уменьшение положительного балла574 после обоих исключений: '+f"{effects['native']['positive_score_reduction_percent']:.3f} % для исходной головы и {effects['agreement']['positive_score_reduction_percent']:.3f} % для согласия; это изменение балла, а не качества трекинга. Доля потолка HOTA не измерена."]
with (OUT/'РЕЗУЛЬТАТ.md').open('x',encoding='utf8') as f:f.write('\n'.join(lines)+'\n')
path=ROOT/'experiments/project_history.json';raw=path.read_bytes();d=json.loads(raw);c=next(c for c in d['experiments'] if c['id']==ID);c.update(state='completed_negative',done='2026-11-12',obtained=obtained,interpretation=interpret,decision=decision,cannot_claim=cannot)
for item,value in zip(c['acceptance'],[True,True,False]):item.update(met=value,evidence=rel+'/audit.json')
c['evidence'] += [dict(kind='file',ref=rel+'/'+n) for n in ['scores.json','audit.json','effects.json','baseline_gate.json','execution_receipt.json','РЕЗУЛЬТАТ.md']]
assert path.read_bytes()==raw;path.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
path=ROOT/'experiments/system_problems.json';raw=path.read_bytes();d=json.loads(raw);problem=next(x for x in d['problems'] if x['id']=='С1');problem['cases'].append(dict(date='2026-11-12',component='Состав пространственной выборки внешнего считывателя FlyVis',fact=obtained,hypothesis='Выход выборки за границы усиливает ложный балл, но исключение обеих групп не исправляет574. Причина сохраняющегося сходства внутри оставшейся области ещё не установлена.',discriminating_check=decision,limitation=cannot,evidence=[rel+'/audit.json',rel+'/scores.json',rel+'/effects.json'],source_sha256=sha(OUT/'audit.json')));problem['cards'].append(ID);assert raw==path.read_bytes();path.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
path=ROOT/'experiments/useful_literature.json';raw=path.read_bytes();d=json.loads(raw);ref=next(x for x in d['sources'] if x['id']=='galoogahi2017_background_patches');field='application';ref.setdefault('revisions',[]).append(dict(date=now(),field=field,old=ref[field],reason='Добавлена ограниченная диагностика границ выборки; прежняя роль источника сохранена.'));ref[field]+='\n\n12.11.2026. Раздельно исключены точки вне рамки детектора и дополнительно вне разметки кандидата на сохранённых картах. Ложный балл574 снизился, решение не исправилось. Использована постановка различения цели и окружения; алгоритм BACF не воспроизводился.';ref['evidence'] += [rel+'/protocol.json',rel+'/audit.json'];ref['related'].append('exp:'+ID);assert raw==path.read_bytes();path.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
journal('Получено (`'+ID+'`): '+obtained+' Вывод. '+interpret+' Решение. '+decision+' Рабочие пороги сохранены, полный трекер не запускался.')
print('RECORDED_NEGATIVE WORKER_ENDED')
