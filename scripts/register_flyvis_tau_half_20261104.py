"""Заранее фиксирует причинное сравнение без обучения и переоценки готового базиса."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib,json,os,shutil
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'runs/flyvis_y1_latency_fp64_20261104';OUT=ROOT/'runs/flyvis_tau_half_20261104'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    OUT.mkdir(exist_ok=False)
    previous=json.loads((BASE/'final_verification.json').read_text(encoding='utf-8'));assert previous['complete']
    p=json.loads((BASE/'protocol.json').read_text(encoding='utf-8'))
    p.update(id='flyvis_tau_half_v1',registered_msk=datetime.now(ZoneInfo('Europe/Moscow')).isoformat(),
        intervention='nodes_time_const *= 0.5 after checkpoint load; effective tau=max(raw,dt), unchanged Euler step and all remaining model/decoder parameters',
        training=False,only_one_parameter_group_changed=True,
        primary='16 ordinary-contrast reversal conditions; full coverage, mean paired stock-fast delay reduction >=1 video frame and 95% cluster bootstrap lower bound >0',
        noninferiority='All16 low-contrast reversals identified, mean delay not worse; mean nominal EPE not worse at either contrast; no threshold tuning.',
        baseline='Reused original CPU-FP64 stock outputs, input SHA and independent audit; no retraining, detector or GPU inference.',
        limitations='Four previously studied TRAIN-scene backgrounds, one model and artificial global motion; no HOTA, adaptive time constants, or independent final test. One fixed intervention; dt lower bound may limit effective change.',
        audit='All input/output SHA and independent scalar metric implementations; network replay is not independent and is not claimed.')
    for name in ['scripts/flyvis_tau_half_20261104.py','scripts/audit_flyvis_tau_half_20261104.py']:
        p['frozen_sources'][name]=sha(ROOT/name)
    (OUT/'protocol.json').write_text(json.dumps(p,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (OUT/'authorization.json').write_text(json.dumps(dict(message='Давай решать',scope='Одна фиксированная правка временных констант без обучения; та же первая проверка движения',
       previous_audit='runs/flyvis_y1_latency_fp64_20261104/independent_audit.json'),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (OUT/'inputs').mkdir()
    for file in (BASE/'inputs').iterdir():os.link(file,OUT/'inputs'/file.name)
    for name in ['geometry.npz','textures.npz','textures.json']:shutil.copy2(BASE/name,OUT/name)
    (OUT/'inputs_completed.json').write_text(json.dumps(dict(complete=True,cases=64,protocol_sha256=sha(OUT/'protocol.json'),reused_inputs=True),indent=2)+'\n')
    path=ROOT/'experiments/project_history.json';shutil.copy2(path,OUT/'project_history_before.json')
    data=json.loads(path.read_text(encoding='utf-8'));assert not any(c['id']==p['id'] for c in data['experiments'])
    data['experiments'].append(dict(id=p['id'],title='Я1 — Проверить уменьшение временных констант на задержке движения',area='flyvis',state='planned',started='2026-11-04',tags=['Y1','motion-diagnostic'],
       question='Сокращает ли фиксированное уменьшение временных констант задержку разворота без ухудшения слабого контраста и ошибки перемещения?',
       hypothesis='Сокращение временных констант способно ускорить переходный отклик ядра; ограничение численного шага может не позволить части констант измениться достаточно.',
       expected='Различить вклад этой группы параметров от других причин; ускорение может сопровождаться потерей точности или устойчивости.',
       acceptance=[dict(criterion=p['primary'],met=None),dict(criterion=p['noninferiority'],met=None)],
       done='',obtained='',interpretation='',decision='',cannot_claim='',metrics=[],
       evidence=[dict(kind='json',ref='runs/flyvis_tau_half_20261104/protocol.json',claim='Условия до расчёта'),dict(kind='json',ref='runs/flyvis_y1_latency_fp64_20261104/independent_audit.json',claim='Проверенный штатный контроль')]))
    temp=path.with_suffix('.tau_half.next');temp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');temp.replace(path)
    plan=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';shutil.copy2(plan,OUT/'plan_before.md');raw=plan.read_bytes();mark='## Журнал изменений'.encode();nl=b'\r\n' if mark+b'\r\n' in raw else b'\n';now=datetime.now(ZoneInfo('Europe/Moscow'))
    entry=f'\n**{now:%d.%m.%Y, %H:%M} MSK** — Автор разрешил проверять решение задержки FlyVis\n\nАвтор: «Давай решать». До расчёта зарегистрирован `flyvis_tau_half_v1`: одна фиксированная правка группы временных констант ×0,5 с неизменным шагом1/60с и штатным ограничением effective=max(tau,dt); остальные веса и считыватель неизменны. Проверяются все64ранее подготовленных входа при обычном и слабом контрасте; старый штатный результат переиспользуется по SHA. Первичный гейт: уменьшение задержки при развороте не менее1кадра и нижняя95%граница выше0; отдельно без ухудшения слабого контраста и номинальной ошибки. Обучения и HOTA нет. Протокол `runs/flyvis_tau_half_20261104/protocol.json`. Ограничение шага может обрезать изменение многих констант; отсутствие эффекта не закрывает адаптивную динамику.\n\n'
    plan.write_bytes(raw.replace(mark+nl,mark+nl+entry.encode(),1))
    (OUT/'status.json').write_text(json.dumps(dict(stage='registered',done=0,total=64)))
    print('Протокол и карточка зарегистрированы.')
if __name__=='__main__':main()
