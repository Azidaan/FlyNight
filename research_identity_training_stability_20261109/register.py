"""Register a zero-fit test of training-video sensitivity before new logits."""
import json,hashlib,subprocess,time
from pathlib import Path
from datetime import datetime,timezone,timedelta
ROOT=Path(__file__).resolve().parents[1];OUT=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def atomic(path,data,before):
    temp=path.with_suffix(path.suffix+'.stability.pending');assert not temp.exists();temp.write_bytes(data)
    for _ in range(30):
        assert path.read_bytes()==before
        try:temp.replace(path);return
        except PermissionError:time.sleep(.1)
    raise PermissionError(str(path))
now=datetime.now(timezone(timedelta(hours=3)));inventory=read(OUT/'inventory.json');assert inventory['ready'] and inventory['folds_ready']==80
assert read(ROOT/'research_visible_geometry_head_20261109/final_checks.json')['passed']
sources=dict(inventory['sources'])
files=[OUT/n for n in ['inventory.json','register.py','compare.py','preflight.py','queue.py']]
files += [ROOT/n for n in ['tracking/channel_identity_head.py','tracking/observed_spatial_reader.py','tracking/full_template_correspondence.py','tracking/hex_linear_interpolation.py','tracking/z13_readout.py','runs/z13_local_peak_20261021/geometry.json','research_visible_geometry_head_20261109/preflight.py','research_identity_synthesis_20261109/ПРОВЕРКА_GEMINI_v18.md']]
for row in inventory['records']:
    for request in row['maps']:
        files += [ROOT/m['path'] for m in request['matches']]
sources.update({p.relative_to(ROOT).as_posix():sha(p) for p in files})
p=dict(registered=now.isoformat(),head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip(),sources=sources,
    action='Apply ALL80ready heldout-video heads (20videos x G7,C34,R134spatial,R134temporal) to the same FIVEpredeclared currentnativequeries. No old fits or core maps repeated. C34vectors reused; restore only missing imagecorrelationvectors from10exact ready R1maps withidentical full-relative support and19shiftmeanwinner. No classifier,threshold or tracker changes.',
    mathematics='Use the saved G7fold geometry normalization for every arm in that fold, then z=theta_g*g+theta_c*r+b. Independently reproduce ALL104saved heldout predictions within1e-12. Two foreign-only videos are exact one-foreign deletions; four mixed videos remove additional ownexamples and MUST NOT be described as a pure foreign effect. Every fold retains both classes; allfive queryvideos excludedfromalltraining.',
    question='Can removal of one alreadyexamined trainingvideo change a known incorrect identity decision at the fixed threshold?',
    acceptance=[dict(criterion='All five current histories have both exact ready R1maps withclosedmetadata/receipts; readyC34logit and104heldoutpredictions reproduced within1e-12; no maps/fits/core/video repeated.',met=None),dict(criterion='For each of three baselineC34errors report all20 sign outcomes, including two pureforeigndeletions andfourmixedgroups separately. A sign correction under either pureforeigndeletion admits direct audit of that removed pair, not model adoption. Mixed-only flips admit only group-sensitivity conclusion. No flips => one-video sensitivity does not explain these decisions; NOT proof that sixnegatives suffice or representation lacks information.',met=None)],
    strongest_alternative='Small, unrepresentative trainingcohort can yield stable wrong weights under every deletion. Conversely a flip may reflect valid rarecontrast rather than a bad label. This check chooses between an addressable single-group audit and a different representation/coverage hypothesis; it cannot certify a general cause.',
    limits='Alreadyviewed five valqueries, no independent test, actualIDs not measured, no thresholds tuned. Centroid/fulltensor alternatives fromGemini rejected as non-identifying. No automatic new native pilot.',
    literature='Koh&Liang2017 section2.1 formalizes leave-point-out prediction influence, section2.3 distinguishes influence from feature proximity; https://proceedings.mlr.press/v70/koh17a/koh17a.pdf. We use existing exact video-heldout models, not their Hessian approximation. Raudys/Jain1991 fulltext unavailable via DOI/IEEE; no30-50sample requirement accepted.',
    authorization='research_identity_synthesis_20261109/authorization.json',save_stop='2026-11-09T23:54:00+03:00',hard_stop='2026-11-09T23:59:00+03:00')
with (OUT/'protocol.json').open('x',encoding='utf8') as f:json.dump(p,f,ensure_ascii=False,indent=2)
identity='identity_training_stability_20261109';path=ROOT/'experiments/project_history.json';before=path.read_bytes();cat=json.loads(before);assert all(c['id']!=identity for c in cat['experiments'])
cat['experiments'].append(dict(id=identity,title='Принадлежность машины — устойчивость к составу обучения',area='flyvis',state='planned',started=now.strftime('%Y-%m-%d'),question=p['question'],hypothesis='Рабочее неверное решение может зависеть от отдельного обучающего видео. Устойчивость к его исключению не доказывает достаточность всей выборки.',expected=p['action'],acceptance=p['acceptance'],obtained='',interpretation='',decision='',cannot_claim=p['limits'],evidence=[dict(kind='file',ref='research_identity_training_stability_20261109/protocol.json',claim='Все готовые модели и границы вывода до новых рабочих оценок')],tags=['fly-problem-P1']))
atomic(path,(json.dumps(cat,ensure_ascii=False,indent=2)+'\n').encode(),before)
path=ROOT/'experiments/obsidian_experiment_topics.json';before=path.read_bytes();topics=json.loads(before);topics['topics'][identity]=['Сохранение личности','Выбор продолжения среди рамок','Обучение признака принадлежности'];atomic(path,(json.dumps(topics,ensure_ascii=False,indent=2)+'\n').encode(),before)
path=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';before=path.read_bytes();nl=b'\r\n' if b'\r\n' in before else b'\n';marker='## Журнал изменений'.encode()+nl
entry='**'+now.strftime('%d.%m.%Y, %H:%M MSK')+'** — По накопленной сводке зарегистрирована проверка влияния обучения\n\n*Постановка (`'+identity+'`).* Один запрос Gemini отправлен и ответ сохранён полностью: research_identity_synthesis_20261109/gemini_v18_submission.json; gemini_v18_capture.json. Расстояние до среднего не доказывает сохранность информации, общая корреляция тензора может повторять прежнее энергетическое считывание, объём 30–50 не обоснован. Исходный Raudys/Jain DOI не открылся, IEEE запросил JavaScript; полный текст не считается проверенным. Отдельно проверены Koh и Liang (2017), разделы 2.1 и 2.3, о влиянии исключения обучения. Разбор: ПРОВЕРКА_GEMINI_v18.md.\n\n*Данные.* Найдены все десять точных обычных карт для пяти рабочих запросов и 80 готовых моделей с исключением видео. Два видео содержат только одну чужую связь, четыре — также свои; их эффекты будут различены. Источник: research_identity_training_stability_20261109/inventory.json.\n\n*Решение.* Применить готовые головы к неизменным пяти входам с FlyVis, обычными пространственными/временными признаками и геометрией. Новых обучений, сети, видео, порогов и полного трекера нет. Критерии и остановка записаны до новых баллов в protocol.json. Устойчивый отказ не доказывает отсутствие информации в признаке.\n\n'
assert before.count(marker)==1;atomic(path,before.replace(marker,marker+nl+entry.replace('\n',nl.decode()).encode(),1),before)
print('ZERO_FIT_TRAINING_STABILITY_REGISTERED',len(sources),flush=True)
