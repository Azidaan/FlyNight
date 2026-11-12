from common import *
receipt=read(OUT/'publication_repair.json');assert sha(ROOT/'experiments/useful_literature.json')==receipt['after_sha256']
plan=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';raw=plan.read_bytes();assert len(raw)>2_000_000 and raw.decode('utf8').count('## Журнал изменений')==1
entry='Служебная поправка записи литературы для image_correspondence_probe_20261112: добавлен DOI 10.48550/arXiv.1603.03590 после ошибки предварительной сборки графа. Научные данные не пересчитывались. Первая запись этой поправки в план получила OSError22 при открытии файла; исходный план сохранён и прочитан полностью, запись отсутствовала. Повтор выполняется с проверкой неизменности. Источники research_image_correspondence_probe_20261112/publication_repair.json, publication_resume.json.'
assert 'Служебная поправка записи литературы для image_correspondence_probe_20261112' not in raw.decode('utf8')
journal(entry)
text=(OUT/'publish.py').read_text(encoding='utf8').replace('graph_live_v2','graph_live_v3').replace('graph_publication_v2','graph_publication_v3').replace('graph_baseline_v2','graph_baseline_v3')
text=text.replace("'finish.py','publish.py'","'finish.py','publish.py','publish_v3.py','repair_publication.py','resume_publication.py'")
text=text.replace("'.gitignore','protocol.json'","'.gitignore','protocol.json','publication_repair.json','publication_resume.json','prepare_failure.log'")
assert not (OUT/'publish_v3.py').exists();(OUT/'publish_v3.py').write_text(text,encoding='utf8')
save('publication_resume.json',dict(created=now(),prior_error='OSError [Errno22] Invalid argument opening dissertation/ЕДИНЫЙ_ПЛАН.md for writing; subsequent missing publish_v3.py was a dependency failure.',plan_before_sha256=hashlib.sha256(raw).hexdigest(),plan_before_bytes=len(raw),prior_entry_absent=True,resumed_entry_recorded=True,scientific_reruns=0,underlying_OS_error_cause='Not determined; a subsequent unchanged-content-checked write succeeded. Do not infer model or experimental failure.'))
print('RESUMED_PUBLICATION_METADATA')
