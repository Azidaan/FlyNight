"""Register one response-contrast gate and matched scale/image controls."""
import os,sys,json,hashlib,gzip,subprocess,time
from pathlib import Path
from datetime import datetime,timezone,timedelta
ROOT=Path(__file__).resolve().parents[1];OUT=Path(__file__).resolve().parent;TRAIN=ROOT/'research_domain_matched_reader_20261109';NATIVE=ROOT/'runs/domain_matched_native_20261109'
now=datetime.now(timezone(timedelta(hours=3)))
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def atomic(path,data,before):
    temp=path.with_suffix(path.suffix+'.shift_contrast.pending');assert not temp.exists();temp.write_bytes(data)
    for _ in range(30):
        assert path.read_bytes()==before
        try:temp.replace(path);return
        except PermissionError:time.sleep(.1)
    raise PermissionError(str(path))
assert read(OUT/'math_checks.json')['passed'] and read(NATIVE/'final_checks.json')['passed'] and not (OUT/'protocol.json').exists()
train=read(TRAIN/'execution_protocol.json');files=[Path(__file__),OUT/'math_check.py',OUT/'math_protocol.json',OUT/'math_checks.json',OUT/'REVIEW.md',OUT/'study.py',OUT/'audit.py',OUT/'shift_queue.py',ROOT/'tracking/shift_profile_contrast.py']
files += [TRAIN/n for n in ['compute.py','protocol_v2.json','execution_protocol.json','maps_complete.json','features.json','predictions.json','summary.json','audit.json']]
for value in train['inventory_requests'].values():files += [ROOT/item['path'] for item in value.get('ready_files',[])]
files += [p for p in (TRAIN/'maps').glob('*') if p.is_file()]
files += [ROOT/name for name in ['tracking/channel_identity_head.py','tracking/channel_balanced_correspondence.py','tracking/observed_spatial_reader.py','tracking/full_template_correspondence.py','tracking/hex_linear_interpolation.py','tracking/z13_readout.py','scripts/z13_target_model.py','runs/z13_local_peak_20261021/geometry.json','tracking/z13_response_utility.py','scripts/z13_map_utility.py','runs/z13_map_utility_20261018/report.json']]
files += [NATIVE/n for n in ['gemini_request.txt','gemini_submission.json','gemini_answer_accessibility.txt','gemini_answer_capture.json','comparison_report.json','previous_cases_followup.json']]
cases=[dict(video='b1cd1e94-549d0bfe',frame=540,old_id=832,raw_id=31468,kind='own'),dict(video='b23b2649-8349d2a1',frame=544,old_id=232,raw_id=54943,kind='own'),dict(video='b2036451-aa924fd1',frame=937,old_id=574,raw_id=35274,kind='foreign')]
def key(video,sequence):return hashlib.sha256(json.dumps([video,sequence],separators=(',',':')).encode()).hexdigest()[:32]
for case in cases:
    path=NATIVE/'rollouts/C34'/(case['video']+'.events.json.gz');files.append(path)
    with gzip.open(path,'rt',encoding='utf8') as f:data=json.load(f)
    event=next(x for x in data['events'] if x['frame']==case['frame']);q=next(x for x in event['proposals'] if x['old_id']==case['old_id'] and x['raw_id']==case['raw_id']);seq=[[h['frame'],h['crop']] for h in q['history']]
    for sequence in [seq,seq+q['search']]:files += [NATIVE/'features'/case['video']/(key(case['video'],sequence)+'.C34.npy')]
p=dict(registered=now.isoformat(),head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip(),authorization='research_identity_synthesis_20261109/authorization.json',save_stop='2026-11-09T23:54:00+03:00',hard_stop='2026-11-09T23:59:00+03:00',
    sources={path.relative_to(ROOT).as_posix():sha(path) for path in files},axes=read(OUT/'math_checks.json')['axes'],diagnostic_cases=cases,
    question='Can per-channel contrast over shifts improve identity admission while preserving own continuations and not increasing foreign permissions?',
    action='Keep original mean-correlation winner, support,19offsets,geometry,folds,L2lambda1,freeintercept,threshold0. Replace ONLY34selected correlations by34selected-minus-nonneighbor-mean values, hex axial distance>=2. Recover18missing channelvectors; reuse ready winner and scalarprofiles. No core,video,tracker,normalization search or new labels.',
    controls='Matched ordinary34channel spatial/temporal inputs, ready G7 and old per-representation models. Additional per-representation scale-only control multiplies OLD correlations by training-fold SD(contrast)/SD(old); constant-old channels use ratio1 and are disclosed. No heldout information chooses these ratios. Same41features and exact same regularizer. This separates peak information from effective scale/regularization changes.',
    fit='26sameknowntrainingedges,20own/6foreign,20heldoutvideofolds. Reuse oldfoldgeometrynormalization,fit120NEWmodels(3representations x2features x20folds); never refit80oldmodels. Three oldvalcases descriptive only, excluded from fit and gate.',
    acceptance=[dict(criterion='Every old ready mean score/winner agrees within1e-12 with newly completed profiles; fixed support and raw maps unchanged; no dropping samples,undefined-shift imputation or neural calls.',met=None),
        dict(criterion='For each representation separately, contrast must have own_accept>=B,foreign_accept<=B,andstrictlygreaterbalancedaccuracy against BOTH its ready old model and scale-only control. No arm promotion if its gate fails. Point development admission only; no uniqueFlyVis or actualIDclaim.',met=None),
        dict(criterion='Independent all120fit gradient/loss/prediction audit passes; video exclusion and train-only scaling checked. Stop without native rollout if allgatesfalse. A passing arm needs a separate fixed native protocol and matched controls.',met=None)],
    literature='Bolme etal2010§3.5 supports considering peak/sidelobe, but actualPSR divides bystd, excludes11x11pixels and depends on trained response. IEEE link blocked byJSverification; authorPDFread. Our unnormalized34channelcontrast on19shifts is a new engineering hypothesis, not a literature-proven identity method.',
    prior='z13_map_utility alreadyusedpeakz,separatedpeakgap,entropy,spreadon721locationmaps and failed its main criteria; currentrepresentation is per-channel correlation over19alignments. Do not claim new generic peakconfidence. Keep oldstateinconclusive and partial positive observations.',
    strongest_alternative='Wrong objects may produce sharper alignment peaks than the true object; or apparent gain comes only from feature variance and effective regularization. Sameimagecontrols and scale-only controls are required.',
    stop='No automatic rejection from alloldrawscoresnegative: mathcounterexample invalidates that Gemini stoprule. Do not select thresholds,windows,shifts or bestseeds by3knownvalcases. No finalindependenttest at this stage.')
with (OUT/'protocol.json').open('x',encoding='utf8') as f:json.dump(p,f,ensure_ascii=False,indent=2)
identity='shift_profile_contrast_20261109';path=ROOT/'experiments/project_history.json';before=path.read_bytes();cat=json.loads(before);assert all(c['id']!=identity for c in cat['experiments'])
cat['experiments'].append(dict(id=identity,title='Контраст пика совпадения — поканальные сдвиги вместо общего уровня',area='flyvis',state='planned',started='2026-11-09',question=p['question'],hypothesis='Ложное совпадение с общим фоном может быть широким по сдвигам; это ещё не установлено. Альтернатива — чужой объект имеет столь же острый пик или выигрыш даёт только масштаб признаков.',expected=p['action'],acceptance=p['acceptance'],obtained='',interpretation='',decision='',cannot_claim='Это разработческая проверка считывания на прежних26связях. Три старых примера не обучают модель и не определяют допуск. PSR не гарантирует личность; старые проверки пиков не объявляются новыми. Ядро неизменно.',evidence=[dict(kind='file',ref='research_shift_contrast_20261109/protocol.json',claim='Одно изменение и критерии до новых профилей и моделей'),dict(kind='file',ref='research_shift_contrast_20261109/REVIEW.md',claim='Критика Gemini, первоисточник и отличие от прежних попыток')],tags=['fly-problem-P1']))
atomic(path,(json.dumps(cat,ensure_ascii=False,indent=2)+'\n').encode(),before)
path=ROOT/'experiments/obsidian_experiment_topics.json';before=path.read_bytes();topics=json.loads(before);topics['topics'][identity]=['Сохранение личности','Выбор продолжения среди рамок','Контраст пространственного совпадения'];atomic(path,(json.dumps(topics,ensure_ascii=False,indent=2)+'\n').encode(),before)
path=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';before=path.read_bytes();nl=b'\r\n' if b'\r\n' in before else b'\n';marker='## Журнал изменений'.encode()+nl
entry='**'+now.strftime('%d.%m.%Y, %H:%M MSK')+'** — Проверен ответ Gemini и зарегистрирован контраст профиля сдвигов\n\n*Получено.* Полный отрицательный нативный пилот опубликован `10f93b74494bda0587ec58676805316b026ec2f5`; сервер GitHub сверен. Один запрос Gemini отправлен через открытое приложение; весь текст до и после отправки совпал, полный ответ с формулами и конечной меткой сохранён в runs/domain_matched_native_20261109/gemini_answer_accessibility.txt,gemini_answer_capture.json. Предложено вычитать боковой отклик. IEEE-ссылка не открыла статью, но первичный авторскийPDFпрочитан§3.5: https://www.cs.colostate.edu/~draper/papers/bolme_cvpr10.pdf. В статьеPSRделитсянаразброс;простойшаблонможетнеиметьполезногоPSR. ПричинноеобъяснениеGeminiо«плоскойчужой»неподтверждено. Математически опровергнут его универсальный отказ при отрицательных сырых баллах.\n\n*Постановка (`'+identity+'`).* Одназамена:34корреляциина34контраста понесоседнимсдвигам,прежнийвыборсдвигаполностьюсохранён. Тежегеометрия,ячейки,26связей,межвидеовыескладки,штраф1,порог0. Обычныевходные и масштабные контроли обязательны; старые модели не повторяются. Источники research_shift_contrast_20261109/protocol.json,math_checks.json,REVIEW.md. Старые14признаковкартыиотказнаграничномкольце найдены и сопоставлены; идеямаксимуманеназываетсяновой. ТолькоCPUпоготовымкартам;полныйтрекерещёнезапускается.\n\n'
atomic(path,before.replace(marker,marker+nl+entry.replace('\n',nl.decode()).encode(),1),before)
print('SHIFT_CONTRAST_GATE_REGISTERED',len(p['sources']),'sources',flush=True)
