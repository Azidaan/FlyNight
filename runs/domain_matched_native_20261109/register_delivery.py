"""Register the new head's numerical delivery check before its counterfactuals."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import json,hashlib,subprocess,time
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
now=datetime.now(timezone(timedelta(hours=3)));head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip()
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def atomic(path,data,before):
    temp=path.with_suffix(path.suffix+'.delivery.pending');assert not temp.exists();temp.write_bytes(data)
    for _ in range(30):
        assert path.read_bytes()==before
        try:temp.replace(path);return
        except PermissionError:time.sleep(.1)
    raise PermissionError(str(path))
assert read(OUT/'delivery_protocol.json')['head']==head and not (OUT/'delivery_report.json').exists()
assert read(ROOT/'research_domain_matched_reader_20261109/final_checks.json')['passed']
identity='domain_identity_delivery_preflight_20261109'
path=ROOT/'experiments/project_history.json';before=path.read_bytes();cat=json.loads(before);assert all(c['id']!=identity for c in cat['experiments'])
cat['experiments'].append(dict(id=identity,title='Новая голова принадлежности — доставка разрешения в общий решатель',area='flyvis',state='planned',started='2026-11-09',
    question='Принимает ли новая голова рёбра стоимости1, которые при вычитании0,2вfloat32остаютсявыше0,8?',
    hypothesis='Смена обучающей области может разрешить ранее отвергавшиеся рёбра и впервые сделать численную границу существенной. Альтернатива — все новые разрешения проходят локальный порог, а их отклоняет только конкуренция.',
    expected='Повторно использовать213готовыхвекторов и полные матрицы старых историй; применить новые окончательные коэффициенты без обучения. Сравнить точнуюfloat64и прежнююfloat32доставку при прежних0,2и0,8.',
    acceptance=[dict(criterion='Все213готовыхвекторов и исходные назначения воспроизведены; чистое приведение исходной матрицы кfloat64не меняет сохранённые первые назначения.',met=None),dict(criterion='Установить число новых разрешений выше границыfloat32и различие глобальных назначений; не выдавать однокадровый контрфакт за новыеID. Любая правка доставки требует одинаковой точности у всех контролей и отдельногоfloat32варианта для отделения эффектов.',met=None)],
    obtained='',interpretation='',decision='',cannot_claim='Старое исключение численной причины относилось к прежнейголове; новый результат всё ещё не полные траектории и не независимая оценка личности.',
    evidence=[dict(kind='file',ref='runs/domain_matched_native_20261109/delivery_protocol.json',claim='Условия до новых вычислений доставки')],tags=['fly-problem-P1']))
atomic(path,(json.dumps(cat,ensure_ascii=False,indent=2)+'\n').encode(),before)
path=ROOT/'experiments/obsidian_experiment_topics.json';before=path.read_bytes();topics=json.loads(before);topics['topics'][identity]=['Сохранение личности','Выбор продолжения среди рамок','Доставка решения в трекер'];atomic(path,(json.dumps(topics,ensure_ascii=False,indent=2)+'\n').encode(),before)
source=ROOT/'tmp/domain_reader_publication/staging_receipt.json';dest=OUT/'publication_receipt_before_native.json';assert not dest.exists();receipt=read(source)
dest.write_text(json.dumps(dict(commit=head,staging_receipt_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),graph=receipt['graph'],physical_active_notes=receipt['physical_active_notes'],physical_tracked_notes_audited=receipt['physical_tracked_notes_audited'],preserved_foreign_files=len(receipt['preserved_working_files']),scientific_ID_goal_complete=False),ensure_ascii=False,indent=2),encoding='utf8')
path=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';before=path.read_bytes();nl=b'\r\n' if b'\r\n' in before else b'\n';marker='## Журнал изменений'.encode()+nl
entry='**'+now.strftime('%d.%m.%Y, %H:%M MSK')+'** — Проверенные рабочие считыватели опубликованы; зарегистрирована доставка новой головы\n\nКоммит `'+head+'` сохраняет рабочую выборку, отрицательный координатный контроль и фиксированную межвидеовую проверку.58тестовкаталога пройдены, локальныйграф1267узлов/8деревьев,81файлгипотез побайтно сохранён. Публикуемаякопия1260узлов сохраняет прежний серверный вид гипотез; чужой календарь и прочие черновики остаются локально. Проверка расширена на все физические заметки индекса: прежняя проверка только членов манифеста пропустила один старый активный дневнойузел. Генератор исправил архивные метки без удаления содержимого; источник `runs/domain_matched_native_20261109/publication_receipt_before_native.json`, подробная проверка tmp/domain_reader_publication/staging_receipt.json. Сбои локальногоимениqueue.pyи поляDOIсохранены, обойдены отдельной проверяемой загрузкой и исправлением метаданных;58успешныхтестов не повторяли.\n\n*Постановка (`'+identity+'`).* До реального пилота проверить новые разрешения на готовых213векторах и полныхматрицах:float32противfloat64при прежнихalpha0,2иmatch0,8. Это технический предварительный этап, не новый результатID. Порог не подбирается; новые карты и метрики не запускаются. Условия `runs/domain_matched_native_20261109/delivery_protocol.json`. Старый отрицательный вывод о границе не переносится автоматически на новую голову.\n\n'
atomic(path,before.replace(marker,marker+nl+entry.replace('\n',nl.decode()).encode(),1),before)
print('DELIVERY_CARD_REGISTERED',flush=True)
