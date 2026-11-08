import hashlib,json
from pathlib import Path
from datetime import datetime,timedelta,timezone
import numpy as np
from sklearn.metrics import roc_auc_score
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent;PARENT=ROOT/'runs/background_patch_gate_20261108';MAPS=ROOT/'runs/full_decoder_outputs_20261108'
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
p=read(OUT/'protocol.json');rows=[read(ROOT/path) for path in read(OUT/'results.json')['files']]
oldrows={(r['video'],r['frame'],r['old'],r['new']):r for r in [read(ROOT/path) for path in read(PARENT/'results.json')['files']]}
known=[r for r in rows if r['kind']!='unknown' and r['supported'] and all(v['valid'] for v in r['variants'].values())]
assert len(known)==45
pos=[r for r in known if r['kind']=='own'];neg=[r for r in known if r['kind']=='foreign'];videos=sorted({r['video'] for r in known});vid={v:i for i,v in enumerate(videos)}
weights=np.random.default_rng(42).multinomial(len(videos),np.full(len(videos),1/len(videos)),5000)
wp=weights[:,[vid[r['video']] for r in pos]];wn=weights[:,[vid[r['video']] for r in neg]];den=wp.sum(1)*wn.sum(1);valid=den>0
scores={};boots={};fields=['correlation_max','filter_max']
def value(r,name,field):
    if name=='C8':return oldrows[r['video'],r['frame'],r['old'],r['new']]['variants']['FlyVis'][field]
    return r['variants'][name][field]
for name in ['C34','R134_temporal','R134_spatial','C8']:
    scores[name]={}
    for field in fields:
        a=np.asarray([value(r,name,field) for r in pos]);b=np.asarray([value(r,name,field) for r in neg]);matrix=(a[:,None]>b).astype(float)+.5*(a[:,None]==b)
        auc=float(matrix.mean());assert abs(auc-roc_auc_score([r['kind']=='own' for r in known],[value(r,name,field) for r in known]))<1e-12
        scores[name][field]=auc;boots[name,field]=np.einsum('bi,ij,bj->b',wp,matrix,wn)[valid]/den[valid]
chosen=max(fields,key=lambda field:scores['C34'][field]);baseline=read(PARENT/'summary.json')['scores']
best_simple8=max(v[f] for k,v in baseline.items() if k!='FlyVis' for f in fields)
gate=scores['C34'][chosen]>max(scores['C8'][chosen],scores['R134_temporal'][chosen],scores['R134_spatial'][chosen],best_simple8)
comparisons={}
for other in ['C8','R134_temporal','R134_spatial']:
    delta=boots['C34',chosen]-boots[other,chosen]
    comparisons[other]=dict(delta=scores['C34'][chosen]-scores[other][chosen],ci95=np.quantile(delta,[.025,.975]).tolist())
summary=dict(known=len(known),positive=len(pos),foreign=len(neg),videos=len(videos),scores=scores,chosen_reader=chosen,
    best_simple8=best_simple8,gate_pass=gate,comparisons=comparisons,bootstrap=dict(seed=42,resamples=5000,valid=int(valid.sum()),unit='video'))
save(OUT/'summary.json',summary)
evidence=[]
for e in read(PARENT/'protocol.json')['events']:
    key=(e['video'],e['frame'],e['old_tid'],e['new_tid']);r=oldrows[key]
    if not any((k['video'],k['frame'],k['old'],k['new'])==key for k in known):continue
    labels_path=ROOT/f'runs/bbox_singleton_pilot_20261108/labels/{e["video"]}.json'
    label=read(labels_path)['rows'][f'{e["frame"]}|{e["old_tid"]}|{e["new_tid"]}'];rf=e['row_data']['history'][-1]['frame']
    evidence.append(dict(video=e['video'],frame=e['frame'],old=e['old_tid'],new=e['new_tid'],kind=r['kind'],reference_frame=rf,
        reference_has_exact_frame_anchor=any(a[0]==rf for a in label['target_anchors']),current_has_exact_frame_anchor=any(a[0]==e['frame'] for a in label['fragment_anchors']),
        target_anchors=label['target_anchors'],fragment_anchors=label['fragment_anchors'],source=labels_path.relative_to(ROOT).as_posix(),sha256=sha(labels_path)))
quality=dict(records=evidence,reference_exact=sum(r['reference_has_exact_frame_anchor'] for r in evidence),current_exact=sum(r['current_has_exact_frame_anchor'] for r in evidence),
    both_exact=sum(r['reference_has_exact_frame_anchor'] and r['current_has_exact_frame_anchor'] for r in evidence),
    limitation='Exact-frame anchors refer to native track-output annotation matches. Unique raw candidate-box annotation matches were not separately rechecked here. Missing direct anchors do not prove a wrong template; reference corruption remains a hypothesis.')
save(OUT/'reference_evidence.json',quality)
obtained='На тех же45условно известных fit20 парах AUROC C34=0,593750 для обоих считывателей; по заранее заданному правилу выбран более простой корреляционный. C8:0,574519/0,569712. R134временной:0,608173/0,612981; R134пространственный:0,588942/0,586538. Направленный гейт не пройден. Источник `runs/'+OUT.name+'/summary.json`. Из45пар точные по кадру якоря прошлой траектории есть у15, текущей у11, обоих концов у4; это не доказательство ошибочности остальных меток (`reference_evidence.json`).'
interpretation='Возврат26отброшенных карт немного повысил условное ранжирование относительно8T4/T5, но не дал преимущества над равным временным контролем входа. Недостаточность текущего способа считывания остаётся. Техническая ошибка входа или ядра этим опытом не установлена: старые8иR134совпали точно на399последовательностях.'
limitation='Условный обучающий срез,32свои/13чужих на17видео,154неизвестных сохранены. Большинство меток опирается на наблюдения траектории до и после сравниваемого кадра; чистота именно последнего шаблона для всех45не доказана. Это не независимый выигрыш сохраненияID и не полная проверка всех способов чтения34карт. Ядро не обучалось.'
decision='Не внедрять этот фиксированный34-канальный считыватель в трекер и не подбирать по этим меткам новый порог. Следующий узел — качество и обновление памяти цели: проверить, не подменяет ли последний шаблон исходный объект ещё до восстановления. Разделить прямое подтверждение обоих кадров и траекторные метки до следующего обучаемого считывателя. Цель остаётся активной; готовые26карт сохраняются для следующего метода.'
report='# Полные34 карты активности и равные контроли входа\n\n'+obtained+'\n\n'+interpretation+'\n\n| Представление | Корреляция | Фоновый фильтр |\n| --- | --- | --- |\n'
for name,v in scores.items():report+=f'| {name} | {v["correlation_max"]:.6f} | {v["filter_max"]:.6f} |\n'
report+='\nC34собран из прежних8T4/T5и26восстановленных нормированных карт, входящих в34входа штатного декодера. R134временной —17EMAи17отклонений; R134пространственный —17EMAи17средних по соседям. Истории, опоры,19сдвигов, нормировки и оба считывателя неизменны. Нулевая добавка26каналов математически не меняет8-канальную оценку.\n\n'+limitation+'\n\n'+decision+'\n\nИсточники точных по кадру якорей сохранены в reference_evidence.json; прямое отсутствие разметки не названо ошибкой трекера.\n\nПроцент прироста качества полного трекера не определён: гейт не пройден, замкнутый трекер не запускался.\n'
(OUT/'РЕЗУЛЬТАТ.md').write_text(report,encoding='utf8')
path=ROOT/'experiments/project_history.json';cat=read(path);card=next(c for c in cat['experiments'] if c['id']==OUT.name);assert card['state']=='running'
card.update(state='completed_negative',done='2026-11-08',obtained=obtained,interpretation=interpretation,cannot_claim=limitation,decision=decision)
card['acceptance'][0].update(met=True,evidence=f'runs/{OUT.name}/audit.json');card['acceptance'][1].update(met=bool(gate),evidence=f'runs/{OUT.name}/summary.json:gate_pass')
card['evidence'] += [dict(kind='file',ref=f'runs/{OUT.name}/{name}',claim='Первичный итог или проверка качества меток') for name in ['summary.json','results.json','audit.json','reference_evidence.json','РЕЗУЛЬТАТ.md']]
card['metrics']=[dict(key='candidate_identity_auroc',value=v[f],baseline=scores['C8'][f] if n=='C34' else None,subset=f'45 fit20 known / {n} / {f}',source=f'runs/{OUT.name}/summary.json',locator=f'scores.{n}.{f}') for n,v in scores.items() for f in fields]
save(path,cat)
path=ROOT/'experiments/obsidian_experiment_topics.json';topics=read(path);topics['topics'][OUT.name]=topics['topics']['singleton_support_size_20261108'];save(path,topics)
path=ROOT/'experiments/system_problems.json';registry=read(path);problem=next(x for x in registry['problems'] if x['id']=='С2')
problem['cases'].append(dict(date='2026-11-08',component='Полные входные карты штатного декодера и качество памяти цели',fact=obtained,
    hypothesis='Последний шаблон может быть загрязнён чужим объектом до пропуска; отсутствие прямого якоря само по себе этого не устанавливает.',
    discriminating_check='Проверить сырые рамки и точные по кадру якоря обоих концов, затем сравнить стабильную прошлую память с последним шаблоном при тех же сигналах и простых контролях.',limitation=limitation,evidence=[f'runs/{OUT.name}/summary.json',f'runs/{OUT.name}/reference_evidence.json']))
problem['evidence'].append(dict(source=f'runs/{OUT.name}/summary.json',sha256=sha(OUT/'summary.json'),claim='C34 не превзошёл равный временной контроль'))
save(path,registry)
path=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';text=path.read_text(encoding='utf8');now=datetime.now(timezone(timedelta(hours=3)))
entry=f'**{now:%d.%m.%Y, %H:%M} MSK** — Полные34карты не превзошли равный контроль; выделено качество якоря памяти.\n\n*Получено (`{OUT.name}`).* {obtained}\n\n*Вывод.* {interpretation} {limitation}\n\n*Решение.* {decision}\n\n'
path.write_text(text.replace('## Журнал изменений\n','## Журнал изменений\n\n'+entry,1),encoding='utf8')
save(OUT/'independent_verification.json',dict(passed=True,auroc_pair_order_matches_sklearn=True,known=45,source_sha256=sha(OUT/'summary.json')))
save(OUT/'execution_receipt.json',dict(checksums={name:sha(OUT/name) for name in ['study.py','finish.py','protocol.json','mathematical_checks.json','results.json','summary.json','audit.json','reference_evidence.json','РЕЗУЛЬТАТ.md']}))
print(obtained)
