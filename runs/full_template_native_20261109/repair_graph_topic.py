"""Add only this completed experiment to the existing semantic routing table."""
from common import *
path=ROOT/'experiments/obsidian_experiment_topics.json';before=path.read_bytes();data=read(path)
key='full_template_native_20261109';assert key not in data['topics']
with (OUT/'obsidian_topics_before_native.json').open('xb') as f:f.write(before)
route=['Сохранение личности','Выбор продолжения среди рамок','Проверка реальных назначений']
data['topics'][key]=route;tmp=path.with_suffix('.json.native_topic.pending')
with tmp.open('x',encoding='utf8',newline='\n') as f:json.dump(data,f,ensure_ascii=False,indent=2);f.write('\n')
assert path.read_bytes()==before;tmp.replace(path)
write('graph_topic_recovery.json',dict(passed=True,source=path.relative_to(ROOT).as_posix(),sha256=sha(path),
    route=route,key=key,previous_snapshot='runs/full_template_native_20261109/obsidian_topics_before_native.json',
    reason='Preview correctly stopped on missing topics[cardid]. Only one own card routing key added; no foreign routing rule, hypothesis, catalogue or numerical result changed. Original preview failed before touching the live vault.'))
source=(OUT/'finish_checks.py').read_text(encoding='utf8')
assert source.count("'checks_recovery_protocol.json'")==1
source=source.replace("'checks_recovery_protocol.json'","'checks_recovery2_protocol.json'")
source=source.replace("assert not (OUT/'final_checks.json').exists()", "assert not (OUT/'final_checks.json').exists()\ntopic=read(OUT/'graph_topic_recovery.json');assert topic['passed']\ntopic_digest=sha(ROOT/topic['source']);assert topic_digest==topic['sha256']")
source=source.replace('assert hypotheses(vault)==hyp','assert hypotheses(vault)==hyp\nassert sha(ROOT/topic[\'source\'])==topic_digest')
source=source.replace('graph=graph,hypotheses_unchanged=True','graph=graph,graph_topic_source_sha256=topic_digest,hypotheses_unchanged=True')
with (OUT/'finish_checks_v2.py').open('xb') as f:f.write(source.encode('utf8'))
print('One own experiment route registered; graph-only continuation saved',flush=True)
