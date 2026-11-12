from common import *
import shutil
path=ROOT/'experiments/useful_literature.json';raw=path.read_bytes();d=json.loads(raw);s=next(x for x in d['sources'] if x['id']=='kroeger2016_image_correspondence');assert 'doi' not in s
# DOI displayed by the primary arXiv record opened during this task.
s['doi']='10.48550/arXiv.1603.03590'
assert path.read_bytes()==raw;path.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
shutil.copy2(OUT/'prepare.log',OUT/'prepare_failure.log')
save('publication_repair.json',dict(created=now(),cause='used_literature accesses source[doi]; new bibliographic entry omitted that required display field. Catalog validation and58 tests did not cover this generator access.',source='scripts/obsidian_level_graph.py:405',failure_log='prepare_failure.log',before_sha256=hashlib.sha256(raw).hexdigest(),after_sha256=sha(path),added_doi=s['doi'],verification_source='https://arxiv.org/abs/1603.03590',scientific_changes=False,model_replays=0,old_preview_preserved=True))
journal('Служебная поправка записи литературы для image_correspondence_probe_20261112: предварительная сборка графа остановилась из-за отсутствующего поля doi новой записи. Добавлен идентификатор 10.48550/arXiv.1603.03590 из первичной страницы arXiv. Прежняя ошибка и копии сохранены, научные данные не пересчитывались. Источник research_image_correspondence_probe_20261112/publication_repair.json.')
text=(OUT/'publish.py').read_text(encoding='utf8').replace('graph_live_v2','graph_live_v3').replace('graph_publication_v2','graph_publication_v3').replace('graph_baseline_v2','graph_baseline_v3')
text=text.replace("'finish.py','publish.py'","'finish.py','publish.py','publish_v3.py','repair_publication.py'")
text=text.replace("'.gitignore','protocol.json'","'.gitignore','protocol.json','publication_repair.json','prepare_failure.log'")
assert not (OUT/'publish_v3.py').exists();(OUT/'publish_v3.py').write_text(text,encoding='utf8')
print('REPAIRED_METADATA_ONLY')
