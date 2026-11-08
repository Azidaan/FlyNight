import hashlib,json
from pathlib import Path
from datetime import datetime,timedelta,timezone
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
p=read(OUT/'protocol.json');done=read(OUT/'completed.json');assert done['passed'] and done['sequences']==399
for file,digest in p['sources'].items():assert sha(ROOT/file)==digest,file
for file,digest in done['files'].items():assert sha(ROOT/file)==digest,file
obtained='Восстановлены399наборов по26отброшенных карт активности, являющихся частью34входных карт штатного декодера FlyVis. Все готовые8T4/T5 воспроизведены точно: maxerror0; R1по34каналам также совпал точно. Источники `runs/'+OUT.name+'/completed.json`, `runs/'+OUT.name+'/input_readiness.json`. ОдинGPU,219,55с, без обработки видео и обучения ядра; сохранены только отсутствующие26каналов.'
path=ROOT/'experiments/project_history.json';cat=read(path);card=next(c for c in cat['experiments'] if c['id']==OUT.name);assert card['state']=='running'
card.update(title='Восстановление26 карт активности из34 входов штатного декодера FlyVis',question='Могло ли отбрасывание26 из34 входных карт декодера терять сведения об экземпляре?',
    state='infrastructure',done='2026-11-08',obtained=obtained,interpretation='Входы и состояние воспроизведены без изменения готового восьмиканального сигнала. Это восстановление данных; польза дополнительных карт для личности не измерена.',
    decision='Перейти к заранее фиксируемому34-канальному сравнению с равными временными и пространственными контролями входа.',cannot_claim='Не новый полный FlyVis и не готовый трекер. 34карты — активности клеток на входе штатного декодера; его выход потока имеет другую размерность. Веса ядра не изменялись, ID и HOTA не оценены.')
card['acceptance'][0].update(met=True,evidence='runs/'+OUT.name+'/completed.json')
card['evidence'] += [dict(kind='file',ref=f'runs/{OUT.name}/{file}',claim='Проверенные восстановленные данные') for file in ['completed.json','input_readiness.json']]
save(path,cat)
path=ROOT/'experiments/obsidian_experiment_topics.json';topics=read(path);topics['topics'][OUT.name]=topics['topics']['singleton_support_size_20261108'];save(path,topics)
path=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';text=path.read_text(encoding='utf8');now=datetime.now(timezone(timedelta(hours=3)))
entry=f'**{now:%d.%m.%Y, %H:%M} MSK** — Отброшенные26 карты восстановлены с точным совпадением прежних8.\n\n*Получено (`{OUT.name}`).* {obtained}\n\n*Уточнение формулировки.* В первоначальном названии карточки неточно сказано «карты полного декодера». Это34карты активности клеток, подаваемые на штатный декодер, а не34канала его выхода. Название уточнено.\n\n*Решение.* Данные готовы для34-канального сравнения. Научного вывода о личности по восстановлению не делаем.\n\n'
path.write_text(text.replace('## Журнал изменений\n','## Журнал изменений\n\n'+entry,1),encoding='utf8')
save(OUT/'execution_receipt.json',dict(passed=True,checksums={file:sha(OUT/file) for file in ['restore.py','finish.py','protocol.json','input_readiness.json','inputs.npz','completed.json']}))
print(obtained)
