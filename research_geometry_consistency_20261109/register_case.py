"""Register the bounded coordinate diagnostic before it reads measured records."""
from pathlib import Path
import json,hashlib
from datetime import datetime,timezone,timedelta
ROOT=Path(__file__).resolve().parents[1];OUT=Path(__file__).resolve().parent
STAMP=datetime.now(timezone(timedelta(hours=3)))
def update(path,obj,before):
    temp=path.with_suffix(path.suffix+'.coordinate.pending');assert not temp.exists()
    temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    assert path.read_bytes()==before;temp.replace(path)
assert (OUT/'coordinate_protocol.json').exists() and not (OUT/'coordinate_report.json').exists()
path=ROOT/'experiments/project_history.json';before=path.read_bytes();c=json.loads(before)
identity='geometry_coordinate_gate_20261109';assert all(x['id']!=identity for x in c['experiments'])
c['experiments'].append(dict(id=identity,title='Геометрический допуск и чтение карт — согласованность координат',area='flyvis',
    state='planned',started='2026-11-09',
    question='Насколько различаются координаты допуска и чтения на сохранённых запросах и хватает ли общей видимой опоры для одной согласованной замены?',
    hypothesis='Растяжение прошлой полной рамки на нынешнюю видимую часть может менять соответствие участков. Сильнейшая альтернатива: прогноз полного размера также неверен; существующие веса не различают экземпляры при любом из этих совмещений.',
    expected='Только математическая и координатная проверка на всех готовых запросах C34. Нет новых карт, обучения, подбора порогов, назначений или метрик.',
    acceptance=[dict(criterion='Проверить обратимость и вырожденное равенство двух преобразований до разбора реальных запросов.',met=None),
        dict(criterion='Восстановить параметры из фактических mean/covariance/history, воспроизвести прежнее число ячеек и покрыть все 213 оценённых запросов; неизвестные личности сохранить.',met=None),
        dict(criterion='До следующей реализации установить расхождение в единицах шага сетки и доступность минимум трёх видимых ячеек без пересечения рамки на всех19сдвигах. Это критерий готовности, не критерий улучшения ID.',met=None)],
    obtained='',interpretation='',decision='',cannot_claim='Координатное расхождение не доказывает ошибку физического масштаба, улучшение личности или пользу ядра FlyVis. Проверочные20видео уже просмотрены. Готовые сигналы и выходы не повторяются.',
    evidence=[dict(kind='file',ref='research_geometry_consistency_20261109/coordinate_protocol.json',claim='Условия до чтения измеренных координат'),
        dict(kind='file',ref='research_geometry_consistency_20261109/ОСНОВАНИЕ.md',claim='Формулы, первичная литература и прежний8канальныйопыт')],
    tags=['fly-problem-P1']))
update(path,c,before)
path=ROOT/'experiments/obsidian_experiment_topics.json';before=path.read_bytes();topics=json.loads(before)
topics['topics'][identity]=['Сохранение личности','Выбор продолжения среди рамок','Согласованность пространственного соответствия']
update(path,topics,before)
path=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';before=path.read_bytes();nl=b'\r\n' if b'\r\n' in before else b'\n';marker='## Журнал изменений'.encode()+nl
assert before.count(marker)==1
body='**'+STAMP.strftime('%d.%m.%Y, %H:%M MSK')+'** — Зарегистрирована координатная проверка допуска и чтения карт\n\n'
body+='*Постановка (`'+identity+'`).* На всех 213 готовых оценённых запросах C34 восстановить масштаб и смещение geometric_match из сохранённых mean/covariance/history, проверить обратимость и прежнее число ячеек. Сравнить координаты с растяжением рамок; оценить видимую опору при тех же 19 сдвигах, без считывания сигналов и новых назначений. Правило минимум трёх ячеек прежнее. Условия до проверки: `research_geometry_consistency_20261109/coordinate_protocol.json`. Старый G_FLYVIS с обратным геометрическим переходом уже проверялся в другой постановке; его не объявлять новой идеей. Изменений плана раздела7, порогов и ядра нет. Причинная роль масштаба пока неизвестна.\n\n'
temp=path.with_suffix('.coordinate.pending');assert not temp.exists();temp.write_bytes(before.replace(marker,marker+nl+body.replace('\n',nl.decode()).encode(),1));assert path.read_bytes()==before;temp.replace(path)
with (OUT/'registration.json').open('x',encoding='utf8') as f:json.dump(dict(id=identity,registered=STAMP.isoformat(),
    protocol_sha256=hashlib.sha256((OUT/'coordinate_protocol.json').read_bytes()).hexdigest(),
    catalog_sha256=hashlib.sha256((ROOT/'experiments/project_history.json').read_bytes()).hexdigest()),f,ensure_ascii=False,indent=2)
print('REGISTERED_CASE',identity)
