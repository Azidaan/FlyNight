"""Add only the omitted source metadata; preserve the failed graph receipt."""
import json,hashlib,time
from pathlib import Path
from datetime import datetime,timezone,timedelta
ROOT=Path(__file__).resolve().parents[1];OUT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def atomic(path,data,before):
    temp=path.with_suffix(path.suffix+'.doi.pending');assert not temp.exists();temp.write_bytes(data)
    for _ in range(30):
        assert path.read_bytes()==before
        try:temp.replace(path);return
        except PermissionError:time.sleep(.1)
    raise PermissionError(str(path))
path=ROOT/'experiments/useful_literature.json';before=path.read_bytes();data=json.loads(before);entry=next(x for x in data['sources'] if x['id']=='koh2017_training_influence');assert 'doi' not in entry
entry['doi']='не указан на официальной странице PMLR';atomic(path,(json.dumps(data,ensure_ascii=False,indent=2)+'\n').encode(),before)
after=json.loads(path.read_bytes());del next(x for x in after['sources'] if x['id']=='koh2017_training_influence')['doi'];assert after==json.loads(before)
changed={str(path.relative_to(ROOT)):dict(before=hashlib.sha256(before).hexdigest(),after=sha(path),change='Exactly one omitted doi field added; no DOI invented.')}
now=datetime.now(timezone(timedelta(hours=3)));path=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';before=path.read_bytes();nl=b'\r\n' if b'\r\n' in before else b'\n';marker='## Журнал изменений'.encode()+nl
text='**'+now.strftime('%d.%m.%Y, %H:%M MSK')+'** — Исправлено обязательное поле записи литературы перед сборкой графа\n\nПроверки каталога и 58 тестов после результата identity_training_stability_20261109 прошли, но генератор копии графа остановился на KeyError: doi для новой статьи Koh и Liang. Прежний отрицательный код сохранён: runs/flyvis_repairs_20261106/unification_20261107/uncapped_checks/0042932f54414430b91299af5b65d890/memory_guard.worker.log. В реестре добавлено единственное пропущенное поле: DOI не указан на официальной странице PMLR; номер не выдуман. Основное хранилище до сбоя не изменялось. Научные файлы и каталог не изменены. Повторяется только проверка исправленной записи и сборка графа с сохранением гипотез; готовые научные расчёты и 58 тестов каталога переиспользуются при сверке их исходников. Протокол: research_identity_training_stability_20261109/graph_recovery_protocol.json.\n\n'
assert before.count(marker)==1;updated=before.replace(marker,marker+nl+text.replace('\n',nl.decode()).encode(),1);atomic(path,updated,before)
changed[str(path.relative_to(ROOT))]=dict(before=hashlib.sha256(before).hexdigest(),after=sha(path),change='Append one factual infrastructure journal entry; historical text preserved.')
receipt=json.loads((OUT/'final_catalog_tests_receipt.json').read_text(encoding='utf8'));assert receipt['passed'] and receipt['tests']==58
for name,h in receipt['stable_sources'].items():
    if name in changed:assert changed[name]['before']==h,name
    else:assert sha(ROOT/name)==h,name
sources={n:sha(OUT/n) for n in ['comparison.json','audit.json','recording.json','РЕЗУЛЬТАТ.md','final_catalog_tests_receipt.json','final_catalog_tests.log']}
with (OUT/'graph_recovery_protocol.json').open('x',encoding='utf8') as f:json.dump(dict(registered=now.isoformat(),changed=changed,scientific_sources=sources,tests_reused=58,failure_guard='0042932f54414430b91299af5b65d890',scope='Only source-schema repair and protected graph reconstruction. Reuse unchanged catalog tests; rerun catalog validator. No fits/core/video/scientific results repeated.'),f,ensure_ascii=False,indent=2)
print('LITERATURE_DOI_SCHEMA_REPAIRED',flush=True)
