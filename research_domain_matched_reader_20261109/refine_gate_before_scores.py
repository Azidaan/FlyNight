"""Symmetric goal-faithful admission, explicitly registered before any scores."""
import json,hashlib,subprocess,time
from pathlib import Path
from datetime import datetime,timezone,timedelta
ROOT=Path(__file__).resolve().parents[1];OUT=Path(__file__).resolve().parent
now=datetime.now(timezone(timedelta(hours=3)));head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip()
for name in ['execution_protocol.json','features.json','predictions.json','summary.json']:assert not (OUT/name).exists(),name
p=json.loads((OUT/'protocol.json').read_text(encoding='utf8'))
old=p['primary'];p['primary']='C34versusG7onidenticalheldout-videoedges. Native-pilot admission: own accepts>=G7, foreign accepts<=G7, and balanced accuracy>G7. Thus either more own at no extra foreign, fewer foreign at no own loss, or both. Same zero threshold and fixed models. This is admission only, not confirmed core benefit or ID retention.'
p['amended_before_scores']=dict(date=now.isoformat(),old=old,reason='The objective includes recovering own AND rejecting foreign. Requiring fewer foreign alone would reject an own-recovery improvement whenever geometry already rejects every foreign. Use symmetric non-worsening of both outcomes and strict improvement of at least one; no score or fit exists, no threshold changed.',
    original_protocol_sha256=hashlib.sha256((OUT/'protocol.json').read_bytes()).hexdigest(),no_features_fits_predictions_or_execution_passport_exist=True)
with (OUT/'protocol_v2.json').open('x',encoding='utf8') as f:json.dump(p,f,ensure_ascii=False,indent=2);f.write('\n')
path=ROOT/'experiments/project_history.json';before=path.read_bytes();catalog=json.loads(before);card=next(c for c in catalog['experiments'] if c['id']=='domain_matched_identity_head_20261109')
criterion=card['acceptance'][1]['criterion'];card.setdefault('revisions',[]).append(dict(date='2026-11-09',commit=head,field='acceptance[1].criterion',old=criterion,reason='Уточнено до признаков и обучения: учитываются обе части исходной цели, а не только уменьшение чужих. Порог модели не меняется.'))
card['acceptance'][1]['criterion']='Дополнено до прогона: C34 принимает не меньше своих, не больше чужих и имеет строго большую сбалансированную точность, чем G7. Это допуск к реальному пилоту, не решениеID. Прежнее требование только уменьшить чужие сохранено в revisions и исходном протоколе.'
card['evidence'].append(dict(kind='file',ref='research_domain_matched_reader_20261109/protocol_v2.json',claim='Симметричный критерий исходной цели до каких-либо признаков, обучения или баллов'))
temp=path.with_suffix('.matched_gate.pending');assert not temp.exists();temp.write_text(json.dumps(catalog,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
for _ in range(30):
    assert path.read_bytes()==before
    try:temp.replace(path);break
    except PermissionError:time.sleep(.1)
else:raise PermissionError(str(path))
path=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';before=path.read_bytes();nl=b'\r\n' if b'\r\n' in before else b'\n';marker='## Журнал изменений'.encode()+nl
entry='**'+now.strftime('%d.%m.%Y, %H:%M MSK')+'** — До баллов уточнён симметричный допуск рабочего считывателя\n\nУ доменной правки допускается больше своих без добавленных чужих либо меньше чужих без потери своих. Обе величины не ухудшаются, сбалансированная точность строго растёт относительно G7. Иначе вариант не проходит. Прежний критерий требовал только уменьшения чужих и мог отвергать полезное возвращение своих при нуле чужих у геометрии. Уточнение сделано до любых признаков, обучения и баллов; источники `research_domain_matched_reader_20261109/protocol_v2.json`, исходный protocol.json и revisionsкарточки. Порог0,штраф1,состав26пар и четыре модели прежние. Раздел7плана не меняется.\n\n'
temp=path.with_suffix('.matched_gate.pending');assert not temp.exists();temp.write_bytes(before.replace(marker,marker+nl+entry.replace('\n',nl.decode()).encode(),1))
for _ in range(30):
    assert path.read_bytes()==before
    try:temp.replace(path);break
    except PermissionError:time.sleep(.1)
else:raise PermissionError(str(path))
print('SYMMETRIC_GATE_RECORDED_BEFORE_SCORES',flush=True)
