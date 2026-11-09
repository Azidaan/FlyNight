"""Record verified working-domain data and preregister a fixed-weight reader screen."""
import json,hashlib,subprocess
from pathlib import Path
from datetime import datetime,timezone,timedelta
ROOT=Path(__file__).resolve().parents[1];OUT=Path(__file__).resolve().parent
now=datetime.now(timezone(timedelta(hours=3)));head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip()
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def replace(path,data,before):
    tmp=path.with_suffix(path.suffix+'.domain.pending');assert not tmp.exists();tmp.write_bytes(data);assert path.read_bytes()==before;tmp.replace(path)
def update(path,obj,before):replace(path,(json.dumps(obj,ensure_ascii=False,indent=2)+'\n').encode(),before)
r=read(OUT/'results.json');assert r['passed'] and r['readiness'] and r['counts']==dict(unknown=137,own=20,foreign=6)
assert not (ROOT/'research_coordinate_reader_20261109/comparison.json').exists()
facts='В точной рабочей области нативного считывателя на84прежнихfit84видео найдены20своих и6чужих связей на20видео;137рабочихсвязей неизвестны. Всего179предложений,163с прежнейопоройот3ячеек. Оба класса сохраняются после исключения любого целого видео. Все100857кадров и903890строк готового нативного вывода совпали точно. Источник: research_working_domain_identity_20261109/results.json → counts, known_videos, held_out_training_classes, exact_frames, exact_rows.'
decision='Готов новый причинный обучающий пул рабочего допуска, но6чужихпримера не гарантируют перенос. Перед новым обучением отдельно проверить одну геометрическую замену с полностью прежними весами на213готовыхнативныхзапросах и тех же обычных входах. Отсутствие рабочего чужого класса встаромнаборе препятствует обоснованию новой условной головы, но не запрещает фиксированное вмешательство в координаты без обучения. Эти вопросы не смешивать.'
path=ROOT/'experiments/project_history.json';before=path.read_bytes();catalog=json.loads(before)
c=next(c for c in catalog['experiments'] if c['id']=='working_domain_identity_inventory_20261109')
c.update(state='infrastructure',done='2026-11-09',obtained=facts,
    interpretation='Получены подтверждённые чужие именно там, где геометрия разрешает связь. Это готовность данных, не успешный считыватель и не сохранённыеID. Все84видео и20видео нового известного пула разработческие, а не новый независимый тест.',decision=decision)
for a in c['acceptance']:a.update(met=True,evidence='research_working_domain_identity_20261109/results.json; videos/*.json; protocol.json')
c['evidence'] += [dict(kind='file',ref='research_working_domain_identity_20261109/results.json',claim='Рабочие свои/чужие, неизвестные и точные нативные выходы')]
identity='coordinate_frozen_reader_screen_20261109';assert all(c['id']!=identity for c in catalog['experiments'])
catalog['experiments'].append(dict(id=identity,title='Согласованные координаты карт — проверка с прежними весами',area='flyvis',state='planned',started='2026-11-09',
    question='Изменит ли одно согласование координат неверные решения без переобучения головы и изменения порога?',
    hypothesis='Текущий пространственный признак может портиться из-за растяжения видимой части на всю старую рамку. Сильнейшая альтернатива — неверны прогноз масштаба или обученные веса; согласование не устранит отказ.',
    expected='Все213запросов исходногоC34, те же прошлые/текущие карты дляFlyVis и двух обычных34канальныхвходов, полностью прежние веса. Готовые признаки переиспользуются.',
    acceptance=[dict(criterion='До новых баллов три первых фиксированных старых C34вектора и сдвига воспроизводятся с ошибкой не более1e-10; источники неизменны; ядро/видео/порог/обучение не запускаются.',met=None),
        dict(criterion='Сохранить все прежде принятые известные свои, исправить хотя бы один из2известныхошибочныхотказов и не увеличить число принятых известных чужих. Это только допуск к реальному пилоту, не независимое доказательство или сохранённыеID.',met=None)],
    obtained='',interpretation='',decision='',cannot_claim='Постфактум диагностическая проверка на20просмотренныхvalвидео,207личностейнеизвестны. Координаты и сигналы оцениваются до назначения; успех потребует реальныхID и независимого теста.',
    evidence=[dict(kind='file',ref='research_coordinate_reader_20261109/protocol.json',claim='Единственная замена и критерии до новых баллов')],tags=['fly-problem-P1']))
update(path,catalog,before)
path=ROOT/'experiments/obsidian_experiment_topics.json';before=path.read_bytes();topics=json.loads(before)
topics['topics'][identity]=['Сохранение личности','Выбор продолжения среди рамок','Фиксированные веса и координаты']
update(path,topics,before)
path=ROOT/'experiments/system_problems.json';before=path.read_bytes();sdata=json.loads(before);s=next(x for x in sdata['problems'] if x['id']=='С2')
for field,value in [('fact',s['fact']+'\n\n'+facts+' Актуальная сводка: research_identity_synthesis_20261109/СВОДКА_v12.md.'),('next_check',decision)]:
    s.setdefault('revisions',[]).append(dict(date='2026-11-09',commit=head,field=field,old=s[field],reason='Рабочий пул восстановлен; отделена проверка координат с прежними весами от нового обучения.'));s[field]=value
s['updated']=now.isoformat();s['cards'] += [identity]
for source,claim in [('research_geometry_consistency_20261109/coordinate_report.json','Координатное расхождение на213нативныхзапросах'),('research_geometry_consistency_20261109/training_states_complete.json','Старый обучающий пул не содержит чужих внутри необходимого геометрического допуска'),('research_working_domain_identity_20261109/results.json','Новый рабочий пул20своих и6чужих с сохранением неизвестных')]:
    if not any(x.get('source')==source for x in s['evidence']):s['evidence'].append(dict(source=source,sha256=sha(ROOT/source),claim=claim))
for evidence in s['cases'][-1].get('evidence',[]):
    if 'path' in evidence:evidence['source']=evidence.pop('path');evidence['claim']='Первичный координатный или обучающий аудит'
s['cases'].append(dict(date='2026-11-09',component='Подтверждённые связи внутри рабочего допуска',fact=facts,
    hypothesis='Исправление распределения обучения и исправление координат — отдельные вмешательства; ни одно пока не признано причиной или решением провала.',
    discriminating_check=decision,limitation='6чужихна6видео — достаточность классов для процедуры, а не статистическая гарантия качества.',
    evidence=[dict(source='research_working_domain_identity_20261109/results.json',sha256=sha(OUT/'results.json'),claim='Точная рабочая выборка без новых карт или метрик')]))
update(path,sdata,before)
text='''# Сохранение личности: рабочие примеры найдены

1. Нерешённая цель

Своей машине после пропуска нужен прежний ID; чужой рамке нельзя передавать его. Последний полный20видеопилот отрицательный, версия не принята. Ошибка ядра не установлена. См. runs/full_template_native_20261109/РЕЗУЛЬТАТ.md; прежние версии1–11сохранены.

2. Установленные ограничения прежней проверки

В207из213запросов различаются геометрия допуска и координаты чтения. Видимой опоры достаточно; два ошибочных своих отказа сохраняют94и66ячеек. Это измеренное расхождение, но не доказанная неправильность физического масштаба. Источник research_geometry_consistency_20261109/coordinate_report.json.

Из91старойобучающейпары только6своих и0чужих проходят необходимый геометрический допуск. Поэтому прежняя межвидеовая точность описывает другой условный пул. Доказано ограничение переноса вывода, а не причинная роль этого ограничения. Источник training_states_complete.json в той же папке;182487публичныхстрок совпали точно.

3. Новый полезный материал

'''+facts+'''

Новый отбор требует именно нативную потерянную строку без первого назначения, две реальные рамки, ровно одну геометрически допустимую, тот же пропуск5–6кадров и прежнюю опору. Разметка не выбирает кандидата или позу. В отличие от прежнего исторического пула, метка второй рамки вообще не нужна. Прошлая опора и текущая сырая рамка прямо подтверждены на реальных аннотационных кадрах; промежуточные кадры не додуманы.

4. Разделяем два действия

'''+decision+'''

У фиксированной координатной замены критерий до баллов: сохранить прежние известные свои разрешения, исправить хотя бы один издвухотказов и не добавить известных чужих разрешений. При провале не менять веса, сдвиги или порог по этим случаям. При успехе нужен полный нативный пилот с теми же входными контролями и прежними защитными критериями. Источник research_coordinate_reader_20261109/protocol.json.

5. Границы знания

Геометрический прогноз может быть неверен. Веса и пространственные признаки могут не переноситься даже после согласования. История может содержать промежуточную чужую машину: прямые метки установлены для двух наблюдений, а не всех кадров. Большинство рабочих связей неизвестно. Конкуренция и удаление дубликатов остаются отдельными механизмами потери номера. Новые данные и математическая исправность ещё не являются улучшенным трекером.
'''
dest=ROOT/'research_identity_synthesis_20261109/СВОДКА_v12.md';assert not dest.exists();dest.write_text(text,encoding='utf8')
path=ROOT/'research_identity_synthesis_20261109/АКТУАЛЬНОЕ.md';before=path.read_bytes();replace(path,('# Актуальное накопленное заключение\n\n[Версия12 — подтверждённые примеры точного рабочего допуска](СВОДКА_v12.md).\n\nНайдены20своих и6чужих на20обучающихвидео;137рабочихсвязей неизвестны. Полный пилот остаётся отрицательным. Следующая зарегистрированная проверка — только согласование координат с прежними весами на213готовыхзапросах и одинаковыми обычными входами. Новое обучение является отдельным действием. Прежние версии1–11сохранены.\n').encode(),before)
path=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';before=path.read_bytes();nl=b'\r\n' if b'\r\n' in before else b'\n';marker='## Журнал изменений'.encode()+nl
entry='**'+now.strftime('%d.%m.%Y, %H:%M MSK')+'** — Найдены свои и чужие в точном рабочем допуске\n\n*Получено (`working_domain_identity_inventory_20261109`).* '+facts+'\n\n*Вывод.* Это готовность материала, не решение личности. Исторические84видео ранее использовались другими считывателями; независимым тестом не являются.\n\n*Решение.* '+decision+'\n\n*Постановка (`'+identity+'`).* Одно согласование координат с прежними весами, без обучения и подбора. Все213готовыхC34запросов и идентичные истории для обычных входов. Проверка первых3старыхвекторов до новыхбаллов; остальные готовые C34и совпадающиевходныепризнаки переиспользуются. Критерий сохранения своих и отсутствия добавленных известных чужих зарегистрирован до баллов в `research_coordinate_reader_20261109/protocol.json`. Раздел7плана не меняется; сводкаv12.\n\n'
replace(path,before.replace(marker,marker+nl+entry.replace('\n',nl.decode()).encode(),1),before)
with (OUT/'recording.json').open('x',encoding='utf8') as f:json.dump(dict(recorded=now.isoformat(),head=head,version=12,next_card=identity,goal_achieved=False),f,indent=2)
print('RECORDED_DOMAIN_DATA_AND_FROZEN_READER_SCREEN',flush=True)
