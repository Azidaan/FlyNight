"""Close the fixed-weight coordinate version without changing its failed gate."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import json,hashlib,subprocess
ROOT=Path(__file__).resolve().parents[1];OUT=Path(__file__).resolve().parent
now=datetime.now(timezone(timedelta(hours=3)));head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def update(path,obj,before):
    temp=path.with_suffix(path.suffix+'.coord_result.pending');assert not temp.exists()
    temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf8');assert path.read_bytes()==before;temp.replace(path)
p=json.loads((OUT/'comparison.json').read_text(encoding='utf8'));assert p['passed'] and not p['gate']
facts='Одна замена координат с полностью прежними весами не исправила ни одного издвухошибочныхотказов FlyVis. Своя540: балл −0,272990→−0,177607; своя544: −0,119333→−0,142575; чужая937: +0,792016→+0,788944. Из5своих разрешены3 до и после, чужая1разрешена до и после. Источник: research_coordinate_reader_20261109/comparison.json → known, accepted_known, old_own_false_refusals_recovered. Пространственный обычный вход разрешил4своих/1чужую, временной5/1, геометрия5/1 на тех же историях; это разрешения до решателя, не ID. Все213запросов проверены,207личностейнеизвестны;633готовыхвектора переиспользованы, восстановлены только2недостающихобычныхкарты изготовыхрецепторов. Три старых контрольныхвектора совпали точно.'
decision='Не запускать полный нативный пилот этой фиксированной координатной версии: зарегистрированный допуск не выполнен. Не менять порог или веса по шести проверочным меткам. Следующая отдельная правка — обучение того же прежнего считывателя в действующем рабочем допуске на новых20своих/6чужихсвязях с одинаковыми геометрическим и обычными входными контролями. Координатный переход оставить экспериментальным, вне нативного исполнителя.'
path=ROOT/'experiments/project_history.json';before=path.read_bytes();c=json.loads(before);card=next(x for x in c['experiments'] if x['id']=='coordinate_frozen_reader_screen_20261109')
card.update(state='completed_negative',done='2026-11-09',obtained=facts,interpretation='Согласованность координат сама по себе с прежними весами не устранила два отказа. Это не доказательство неправильности согласованного перехода или бесполезности пространственных карт; отдельно остаются обучение вне рабочего допуска и ошибки прогноза масштаба.',decision=decision)
card['acceptance'][0].update(met=True,evidence='research_coordinate_reader_20261109/comparison.json → old_map_feature_parity; нулевыеошибки всех3свидетелей')
card['acceptance'][1].update(met=False,evidence='research_coordinate_reader_20261109/comparison.json → gate=false, old_own_false_refusals_recovered=0')
card['evidence'].append(dict(kind='file',ref='research_coordinate_reader_20261109/comparison.json',claim='Отрицательный контроль с теми же весами и согласованными входами'))
update(path,c,before)
path=ROOT/'experiments/system_problems.json';before=path.read_bytes();sdata=json.loads(before);s=next(x for x in sdata['problems'] if x['id']=='С2')
for field,value in [('fact',s['fact']+'\n\n'+facts+' Актуальная сводка: research_identity_synthesis_20261109/СВОДКА_v13.md.'),('next_check',decision)]:
    s.setdefault('revisions',[]).append(dict(date='2026-11-09',commit=head,field=field,old=s[field],reason='Прежние веса и координатная замена проверены; критерий не пройден.'));s[field]=value
s['updated']=now.isoformat();s['evidence'].append(dict(source='research_coordinate_reader_20261109/comparison.json',sha256=sha(OUT/'comparison.json'),claim='Согласование координат с прежними весами не исправило оба своих отказа'))
s['cases'].append(dict(date='2026-11-09',component='Координаты считывателя с прежними весами',fact=facts,hypothesis='Ошибочная геометрия чтения не является достаточным объяснением двух отказов с этой головой.',discriminating_check=decision,limitation='Постфактум диагностика на просмотренных видео; не полные траектории и не независимый тест.',evidence=[dict(source='research_coordinate_reader_20261109/comparison.json',sha256=sha(OUT/'comparison.json'),claim='Фиксированный неуспешный контроль')]))
update(path,sdata,before)
text='''# Сохранение личности: координатной замены недостаточно

1. Что остаётся целью

Нужно принимать свою машину после пропуска, отвергать чужую и фактически сохранять прежний ID. Последний полный20видеопилот отрицательный; текущая версия не принята. Дефект ядра FlyVis не установлен. Первичный полный итог: runs/full_template_native_20261109/РЕЗУЛЬТАТ.md.

2. Проверенный координатный кандидат

'''+facts+'''

Кандидат проверен именно без обучения: изменение данных или коэффициентов не могло скрыть его эффект. Несмотря на физически заметное расхождение координат и достаточную опору, оба собственных отказа сохранились. Математическая корректность преобразования не означает полезный признак личности. Критерий до баллов не меняется; полный пилот этой версии не запускается.

3. Более релевантный обучающий материал

Из старых91пары только6своих и0чужих проходят необходимый геометрический допуск. Новый причинный рабочий отбор на84прежнихfit84видео дал20своих и6чужих на20видео,137рабочихсвязей неизвестны. Оба класса остаются после исключения каждого видео. Все100857кадров и903890готовыхпубличныхстрок совпали точно. Источники: research_geometry_consistency_20261109/training_states_complete.json; research_working_domain_identity_20261109/results.json. Это ограничение прежней проверки и готовность нового материала, а не доказанная причина провала или улучшенный трекер.

4. Одно следующее вмешательство

'''+decision+'''

Новый контроль меняет рабочую область обучения, сохраняя прежнее относительное полное считывание,34канала,19сдвигов,штраф1,порог0 и процедуру межвидеовой проверки. Он не совмещается с неудачной координатной заменой. Сначала наличие точных сохранённых входных последовательностей и старых карт; восстановить только недостающее. К реальнымID переходить лишь при заранее зарегистрированном преимуществе над геометрией без потери своих. Шесть чужих — лишь обучаемость процедуры, не гарантия переноса. Все эти данные разработческие; окончательный успех требует независимых полных траекторий.

5. Что не решено

Могут не переноситься и сами признаки, и обученные веса. Промежуточная история может содержать чужую машину: прямые метки подтверждают прошлое и настоящее наблюдения, а не все кадры между ними. Правильное локальное разрешение всё ещё может потеряться при общем назначении или удалении дубликатов. Unknown не превращается в свой или чужой. Старыеверсии1–12сохранены.
'''
dest=ROOT/'research_identity_synthesis_20261109/СВОДКА_v13.md';assert not dest.exists();dest.write_text(text,encoding='utf8')
path=ROOT/'research_identity_synthesis_20261109/АКТУАЛЬНОЕ.md';path.write_text('# Актуальное накопленное заключение\n\n[Версия13 — одной координатной замены недостаточно](СВОДКА_v13.md).\n\nПрежние веса с согласованными координатами не исправили два своих отказа; версия остановлена по прежнему критерию. Новое обучение — отдельное вмешательство: прежний считыватель на20своих/6чужих в точном рабочем допуске, с одинаковыми контролями. Полный пилот остаётся отрицательным, цель открыта. Прежние версии1–12сохранены.\n',encoding='utf8')
path=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';before=path.read_bytes();nl=b'\r\n' if b'\r\n' in before else b'\n';marker='## Журнал изменений'.encode()+nl
entry='**'+now.strftime('%d.%m.%Y, %H:%M MSK')+'** — Координатная замена с прежними весами не исправила свои отказы\n\n*Получено (`coordinate_frozen_reader_screen_20261109`).* '+facts+'\n\n*Вывод.* Согласованность координат не является достаточным исправлением этой головы.\n\n*Решение.* '+decision+' Полный20видеопрогон не запускается; готовые ядро/карты/метрики не повторялись. Связи и критерий сохранены, сводкаv13.\n\n'
tmp=path.with_suffix('.coord_result.pending');assert not tmp.exists();tmp.write_bytes(before.replace(marker,marker+nl+entry.replace('\n',nl.decode()).encode(),1));assert path.read_bytes()==before;tmp.replace(path)
with (OUT/'recording.json').open('x',encoding='utf8') as f:json.dump(dict(recorded=now.isoformat(),version=13,goal_achieved=False,full_pilot_stopped_by_gate=True),f,indent=2)
print('RECORDED_NEGATIVE_FIXED_COORDINATE_READER',flush=True)
