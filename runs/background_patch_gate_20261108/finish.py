"""Summarize the frozen training gate, preserving unknown cases."""
import hashlib,json
from pathlib import Path
from datetime import datetime,timedelta,timezone
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
p=read(OUT/'protocol.json');files=read(OUT/'results.json')['files']
rows=[]
for path,digest in files.items():assert sha(ROOT/path)==digest;rows.append(read(ROOT/path))
known=[r for r in rows if r['kind']!='unknown' and r['supported'] and all(v.get('valid',False) for v in r['variants'].values())]
pos=[r for r in known if r['kind']=='own'];neg=[r for r in known if r['kind']=='foreign']
videos=sorted({r['video'] for r in known});vi={v:i for i,v in enumerate(videos)}
weights=np.random.default_rng(42).multinomial(len(videos),np.full(len(videos),1/len(videos)),5000)
wp=weights[:,[vi[r['video']] for r in pos]];wn=weights[:,[vi[r['video']] for r in neg]]
den=wp.sum(1)*wn.sum(1);valid=den>0;boots={};scores={}
for name in p['representations']:
    scores[name]={}
    for field in ['correlation_max','filter_max']:
        a=np.array([r['variants'][name][field] for r in pos]);b=np.array([r['variants'][name][field] for r in neg])
        matrix=(a[:,None]>b).astype(float)+.5*(a[:,None]==b)
        auc=float(matrix.mean());scores[name][field]=auc
        boots[name,field]=np.einsum('bi,ij,bj->b',wp,matrix,wn)[valid]/den[valid]
def difference(a,b):
    values=boots[a]-boots[b]
    return dict(delta=scores[a[0]][a[1]]-scores[b[0]][b[1]],ci95=np.quantile(values,[.025,.975]).tolist())
gate=scores['FlyVis']['filter_max']>scores['FlyVis']['correlation_max'] and scores['FlyVis']['filter_max']>max(scores[n]['filter_max'] for n in p['representations'] if n!='FlyVis')
comparisons={'FlyVis_filter_minus_correlation':difference(('FlyVis','filter_max'),('FlyVis','correlation_max'))}
comparisons.update({f'FlyVis_filter_minus_{n}':difference(('FlyVis','filter_max'),(n,'filter_max')) for n in p['representations'] if n!='FlyVis'})
summary=dict(events=len(rows),supported=sum(r['supported'] for r in rows),known=len(known),positive=len(pos),foreign=len(neg),
    unknown=sum(r['kind']=='unknown' for r in rows),known_videos=len(videos),scores=scores,comparisons=comparisons,
    gate_pass=gate,bootstrap=dict(seed=42,resamples=5000,valid=int(valid.sum()),unit='video'),
    reused_forward_rows=sum(v.get('correlation_reused',False) for r in rows for v in r['variants'].values()),
    missing_background_known=sum(r['variants']['FlyVis']['negative_count']==0 for r in known))
save(OUT/'summary.json',summary)
obtained=f'Из 200 сохранённых fit20 запросов физическую поддержку имеют {summary["supported"]}; условно известны {len(known)} ({len(pos)} своих, {len(neg)} чужих) на {len(videos)} видео. AUROC FlyVis: корреляция {scores["FlyVis"]["correlation_max"]:.6f}, фильтр {scores["FlyVis"]["filter_max"]:.6f}. Лучший простой фильтр, средние с соседями: {scores["neighbor8"]["filter_max"]:.6f}. Фиксированный направленный гейт не пройден. Источник `runs/{OUT.name}/summary.json`.'
interpretation='Одна зафиксированная реализация фильтра с реальными отрицательными пространственными участками не улучшила различение своих и чужих относительно той же корреляции и уступила простым признакам. Причина ошибки полного трекера не доказана. Это не воспроизведение BACF и не опровержение фонового обучения вообще.'
limitation='Только условные известные пары fit20, не независимое сохранение ID. 154 неизвестных исхода оставлены неизвестными. Поддержка определяется геометрией; один известный свой запрос не поддержан. Три известных поддержанных случая не имеют допустимых фоновых участков и сводятся к прежней корреляции. Полный трекер не изменён и не оценивался. DOI-resolver и Crossref API не открылись; содержание статьи проверено по первичному arXiv тексту, официальная MATLAB-ссылка вернула502, CVF PDF403.'
decision='Этот фиксированный фильтр не продвигать к независимому прогону трекера и не менять lambda по просмотренным меткам. Следующий узел активной цели — представление: проверить сохранность и восстановимость остальных26 из34 штатных выходных каналов, которые прежний экспорт C отбросил. Сравнение должно включать равные34 простых временных канала. Это проверка потери представления, а не анализ всех типов клеток ради полноты.'
report='# Пространственный фильтр с отрицательными участками прошлого кадра\n\n'+obtained+'\n\n'+interpretation+'\n\n'
report+='## 1. Метод и отличие от прошлых попыток\n\nФильтр получает вектор всей пространственной раскладки цели, отдельный коэффициент на каждую пару позиция/канал. Положительный образец — прошлый шаблон. До64 отрицательных целых участков взяты из того же прошлого глаза, без пересечения с наблюдавшейся рамкой и без циклического замыкания границ. Нормировка по каналам прежняя; регуляризатор единичный, отрицательная ошибка усредняется, параметры не подбирались. После обучения отклик нормирован на отклик прошлого шаблона.\n\nВ старом z13_target_model_probe_v1 один вес канала применялся независимо ко всем ячейкам. Старые производные добавляли локальные наклоны. Новый фильтр сохраняет всю раскладку позиций. При отсутствии отрицательных образцов он точно даёт прежнюю корреляцию. Четыре математических теста и независимое решение Вудбери проверяют эту реализацию.\n\n'
report+='## 2. Условное ранжирование известных пар\n\nAUROC — доля сравнений «оценка своей выше чужой» с половиной веса для равенства. Это диагностика, не качество ID полного трекера.\n\n| Представление | Прежняя корреляция | Пространственный фильтр |\n| --- | --- | --- |\n'
for name in p['representations']:report+=f'| {name} | {scores[name]["correlation_max"]:.6f} | {scores[name]["filter_max"]:.6f} |\n'
report+='\nПарные интервалы получены пересэмплированием видео, 5000 повторов, seed42. Все значения в summary.json, comparisons.\n\n'+limitation+'\n\n'
report+='## 3. Исполнение и решение\n\nВсе200 запросов сохранены, фильтр не получал метки личности. Они присоединены после расчёта. Восемь каналов каждого варианта, одинаковые опоры и19сдвигов. Карты и сеть не пересчитывались. 12 прямых строк сравнений взяты готовыми. Все новые результаты сохранены по запросам с защитой от повторного расчёта; интерполяция и независимая формула фильтра совпали до10⁻⁹.\n\n'+decision+'\n\n'
report+='## 4. Литературное основание\n\nHamed Kiani Galoogahi, Ashton Fagg, Simon Lucey. Learning Background-Aware Correlation Filters for Visual Tracking. ICCV2017. Первичный текст https://arxiv.org/pdf/1703.04590, разделы3–4, уравнение3, рисунок2. Использован принцип реальных отрицательных участков; опубликованные FFT/ADMM, HOG, обновление и результаты авторов не воспроизводились.\n\nПроцент прироста качества трекера и доля потолка не определены: гейт не пройден и трекер не запускался.\n'
(OUT/'РЕЗУЛЬТАТ.md').write_text(report,encoding='utf8')
path=ROOT/'experiments/project_history.json';cat=read(path);card=next(c for c in cat['experiments'] if c['id']==OUT.name);assert card['state']=='running'
card.update(state='completed_negative',done='2026-11-08',obtained=obtained,interpretation=interpretation,cannot_claim=limitation,decision=decision)
card['acceptance'][0].update(met=True,evidence=f'runs/{OUT.name}/audit.json')
card['evidence'] += [dict(kind='file',ref=f'runs/{OUT.name}/{file}',claim='Измеренный результат и проверка') for file in ['summary.json','results.json','audit.json','mathematical_checks.json','РЕЗУЛЬТАТ.md']]
card['metrics']=[dict(key='candidate_identity_auroc',value=scores[n][field],baseline=scores[n]['correlation_max'] if field=='filter_max' else None,subset=f'45 fit20 known / {n} / {field}',source=f'runs/{OUT.name}/summary.json',locator=f'scores.{n}.{field}') for n in p['representations'] for field in ['correlation_max','filter_max']]
save(path,cat)
path=ROOT/'experiments/obsidian_experiment_topics.json';topics=read(path);topics['topics'][OUT.name]=topics['topics']['singleton_support_size_20261108'];save(path,topics)
path=ROOT/'experiments/system_problems.json';registry=read(path);problem=next(p for p in registry['problems'] if p['id']=='С2')
problem['cases'].append(dict(date='2026-11-08',component='Пространственный фильтр с целыми фоновыми отрицательными примерами',fact=obtained,hypothesis='Выбор восьми выходных каналов может ограничивать различение экземпляров.',discriminating_check='Сверить сохранённые34 выхода и26 отброшенных каналов; восстанавливать только отсутствующее с проверкой старых8 и равным R1 контролем.',limitation=limitation,evidence=[f'runs/{OUT.name}/summary.json']))
problem['evidence'].append(dict(source=f'runs/{OUT.name}/summary.json',sha256=sha(OUT/'summary.json'),claim='Фиксированный фоновой фильтр не прошёл условный гейт'))
save(path,registry)
path=ROOT/'experiments/useful_literature.json';lit=read(path)
lit['sources'].append(dict(id='galoogahi2017_background_patches',title='Обучение фильтра на цели и реальных фоновых участках — Galoogahi и соавторы',
    problem_group='Сохранение личности',method_group='Отделение цели от фона',publication='Hamed Kiani Galoogahi, Ashton Fagg, Simon Lucey. Learning Background-Aware Correlation Filters for Visual Tracking. ICCV,2017,1135–1143.',
    doi='10.1109/ICCV.2017.129',url='https://arxiv.org/pdf/1703.04590',sections='Разделы3–4, уравнение3, рисунок2. DOI-resolver и Crossref API недоступны в этой проверке.',
    established='Авторы обучают пространственный корреляционный фильтр на положительном участке цели и реальных отрицательных фоновых участках; опубликована отдельная эффективная частотная оптимизация.',
    problem='С2 — чужой объект может давать высокое пространственное сходство.',application='Принцип целых реальных отрицательных участков использован в прямом пространственном ridge-фильтре на готовых гексагональных картах. Математика проверена, один фиксированный гейт fit20 не пройден.',
    limitation='Это ограниченная адаптация без FFT/ADMM,HOG и обновления BACF. Официальный MATLAB ZIP502, CVF PDF403; первичный arXiv текст открыт. Результаты статьи не приписываются нашей модели и не закрывают С2.',
    evidence=[f'runs/{OUT.name}/protocol.json',f'runs/{OUT.name}/summary.json',f'runs/{OUT.name}/РЕЗУЛЬТАТ.md'],related=['exp:'+OUT.name,'sys:С2']))
save(path,lit)
path=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';plan=path.read_text(encoding='utf8');now=datetime.now(timezone(timedelta(hours=3)))
entry=f'**{now:%d.%m.%Y, %H:%M} MSK** — Фильтр на реальных фоновых участках не прошёл обучающий гейт.\n\n*Постановка.* Активная цель сохранения личности. Фиксированная прямая пространственная адаптация принципа BACF, одинаковая геометрия и простые контроли. Карточка `{OUT.name}`.\n\n*Получено (`{OUT.name}`).* {obtained}\n\n*Вывод.* {interpretation} {limitation}\n\n*Решение.* {decision}\n\n'
path.write_text(plan.replace('## Журнал изменений\n','## Журнал изменений\n\n'+entry,1),encoding='utf8')
save(OUT/'execution_receipt.json',dict(checksums={name:sha(OUT/name) for name in ['study.py','finish.py','protocol.json','mathematical_checks.json','results.json','summary.json','audit.json','РЕЗУЛЬТАТ.md']}))
print(obtained)
