from official import *
import shutil,statistics
p=read(OUT/'protocol.json');s=read(OUT/'result.json');det=read(OUT/'detection_result.json');db=read(OUT/'cases.json');clips={r['id']:r for r in db['cases']};pred=np.load(OUT/'recovered_predictions.npz');z=zipfile.ZipFile(ROOT/p['labelzip']);cache={};verified=0
for r in s['records']:
 c=clips[r['case']];v=c['video'];offset=r['offset'];idx=c['index']+offset
 if v not in cache:cache[v]=labels(z,v)
 frame=cache[v][idx];t,valid,other=masks(z,v,frame,c['ids'][r['slot']]);mask=pred[f"{r['case']}__{r['slot']}__{offset}"]
 tn,fp,fn,tp=np.bincount(t[valid].astype(int)*2+mask[valid],minlength=4);assert (int(tp),int(fp),int(fn))==(r['tp'],r['fp'],r['fn']);verified+=1
from collections import Counter
for q in det['records']:
 assert q['full'] is None or q['full']==q['plus_tiles']
 if q['new_from_tiles']:assert q['plus_tiles']['overlap']>=.5 and q['plus_tiles']['tile']>=0 and q['full'] is None
reasons={};examples={}
for split in ['familiar','fresh']:
 missing=[q for q in det['records'] if q['split']==split and q['full'] is None];d=Counter()
 for q in missing:
  tags=[]
  if q['best_full_car'] and q['best_full_car']['iou']>=.5:tags.append('competing_reference')
  elif q['best_full_any'] and q['best_full_any']['iou']>=.5:tags.append('sufficient_only_other_class')
  elif q['best_full_car'] is None:tags.append('no_full_car_box')
  else:tags.append('insufficient_overlap_full_car')
  tags.append('recovered_with_tiles' if q['new_from_tiles'] else 'still_missing');key=' / '.join(tags);d[key]+=1;examples.setdefault(key,[]).append([q['case'],q['slot']])
 reasons[split]=d
times={};total_verified=0
for split in ['familiar','fresh']:
 qs=[q for q in det['records'] if q['split']==split];rr=[q for q in det['frames'] if q['split']==split];v=sorted({q['video'] for q in qs});stat=s['stats'][split];assert sum(q['full'] is not None for q in qs)==stat['full'] and sum(q['new_from_tiles'] for q in qs)==stat['added'];total_verified+=len(qs)
 times[split]={k:float(statistics.median([x[k] for x in rr])) for k in ['full','extra_tiles','full_candidates','raw_tile_candidates','additional_not_selected_for_two_references','duplicates_of_full']}
save('audit.json',dict(passed=True,full_baseline_unchanged=total_verified,independent_future_confusions=verified,failure_breakdown=reasons,examples=examples,timings=times,result_sha256=sha(OUT/'result.json'),detection_sha256=sha(OUT/'detection_result.json')))
summary='Повторный обзор четырёх фиксированных частей при прежних настройках YOLO26s. '+ ' '.join(f"{lab}: исходная рамка у {q['full']} из {q['objects']} объектов, дополнительно найдено {q['added']}; средняя по видео прибавка {q['video_equal_delta']:+.3f} [{q['interval'][0]:+.3f}; {q['interval'][1]:+.3f}]. Из видимых будущих контуров возвращённых объектов {q['recovered_future_passing']}/{q['recovered_future_visible']} достигли перекрытия 0,5." for k,lab in [('familiar','Знакомые эпизоды'),('fresh','Новая серия')] for q in [s['stats'][k]])+' Источник: research_missing_candidate_repair_20261114/result.json, stats.'
q=times['fresh'];perf=f"На свежих кадрах медиана времени одного полного обзора {1000*q['full']:.1f} мс, дополнительно четыре окна {1000*q['extra_tiles']:.1f} мс. Медиана дополнительных автомобильных рамок до объединения {q['raw_tile_candidates']:.1f}; из них повторяют целый кадр при перекрытии не меньше 0,7: {q['duplicates_of_full']:.1f}. Неназначенные двум проверяемым машинам рамки не обязательно ложные. Источник: audit.json, timings."
interp=('Дополнительные окна дали новые автомобильные рамки с достаточным пересечением.' if any(x['added'] for x in s['stats'].values()) else 'Ни одна отсутствующая автомобильная рамка не восстановилась; проверять продолжение по новым рамкам было не на чем. Во всех шести пропусках даже лучшая автомобильная рамка имела перекрытие ниже 0,5. В одном новом эпизоде дополнительное окно покрыло объект рамкой другого класса.')+' Нехватка пересечения с проверочной рамкой объясняет отказ этого оценщика, но сама по себе не доказывает, что машина пропала из изображения или детектор не видел её. Проверка масок по всем исходным прямоугольникам нужна отдельно.'
dec='Сохранить дополнительные окна как исследовательский способ поиска при пропуске; до интеграции проверить лишние назначения и отдельно стоимость выборочного запуска.' if s['scientific_passed'] else 'Не включать повторный обзор в рабочий трекер: заранее заданный критерий на новой серии не выполнен.'
scope='Оценка доступности по известному положению цели в прошлом. Выбор правильного нового кандидата при реальной потере номера и метрики полного трекера не измерялись. Изменение разрешения детали и окружения при обрезке не разделено.'
(OUT/'РЕЗУЛЬТАТ.md').write_text('\n\n'.join([summary,perf,interp,dec,scope])+'\n',encoding='utf8');i='missing_candidate_repair_20261114';cat=read(ROOT/'experiments/project_history.json');card=next(q for q in cat['experiments'] if q['id']==i);card.update(state='completed_positive' if s['scientific_passed'] else 'completed_negative',done='2026-11-14',obtained=summary+' '+perf,interpretation=interp,decision=dec,cannot_claim=scope)
for x,met in zip(card['acceptance'],[True,s['scientific_passed'],True]):x.update(met=met,evidence=rel(OUT/'result.json')+'; '+rel(OUT/'audit.json'))
card['evidence'] += [dict(kind='file',ref=rel(OUT/n)) for n in ['РЕЗУЛЬТАТ.md','result.json','detection_result.json','audit.json']];(ROOT/'experiments/project_history.json').write_text(json.dumps(cat,ensure_ascii=False,indent=2)+'\n',encoding='utf8');topics=read(ROOT/'experiments/obsidian_experiment_topics.json');topics['topics'][i]=['Сохранение личности','Содержание области наблюдения','Поиск отсутствующей рамки'];(ROOT/'experiments/obsidian_experiment_topics.json').write_text(json.dumps(topics,ensure_ascii=False,indent=2)+'\n',encoding='utf8');journal(summary+' '+perf+' '+dec)
src=ROOT/'research_official_contours_20261114';shutil.copyfile(src/'verify_staging.py',OUT/'verify_staging.py');(OUT/'checks.py').write_text((src/'checks.py').read_text(encoding='utf8').replace('official_contours_final_tests','missing_detect_final_tests'),encoding='utf8');(OUT/'publish.py').write_text((src/'publish.py').read_text(encoding='utf8').replace("identity='official_contours_20261114'","identity='"+i+"'").replace("'initial_official.npz','predictions.npz','initial_teacher.npz'","'detection_result.json','recovered_predictions.npz','cases.json'").replace("'scores.npz','initial_official.npz'","'scores.npz','recovered_predictions.npz'"),encoding='utf8')
for name,entry,args in [('checks','checks.py',[]),('publication','publish.py',['prepare'])]:save('queue_'+name+'.json',dict(runner_sha256=sha(OUT/'run_queue.py'),deadline='2026-11-14T08:55:00+03:00',inputs={rel(OUT/n):sha(OUT/n) for n in [entry,'common.py']},jobs=[dict(entry=rel(OUT/entry),args=args,timeout=600)]))
