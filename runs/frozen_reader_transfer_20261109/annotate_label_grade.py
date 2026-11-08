"""Preserve conditional results and explicitly record ambiguous past identity."""
import hashlib,json
from pathlib import Path
from datetime import datetime,timedelta,timezone
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(n,v):
    with (OUT/n).open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n')
audit=read(OUT/'audit.json');rows=audit['direct_current_and_prior_native_target_anchors'];ambiguous=[r for r in rows if not r['past_identity_unique']];assert len(ambiguous)==1 and ambiguous[0]['old']==405
note='У чужой405→417прошлаярамка378кадра соответствует двумGTприпрежнемIoU≥0,5:00008216(0,560993)и00008218(0,724182). Текущая384кадра однозначно00008217. Метка чужая устойчива к двум квалифицированным альтернативам, но точную старую личность нельзя объявлять однозначной. При строгом требовании уникальной прошлойGTв новомсрезе5своихи0чужих; AUROCтакогострогогосрезанеопределена. Последний кадр шаблона6основныхпар не имеет прямой разметки.'
write('label_grade.json',dict(passed=True,conditional_scores_preserved=True,ambiguous_previous_identity=ambiguous,strict_unique_previous_counts=dict(own=5,foreign=0,auroc=None),current_raw_unique_count=6,latest_reference_direct_count=0,note=note,sources={'runs/'+OUT.name+'/audit.json':sha(OUT/'audit.json')}))
path=ROOT/'experiments/project_history.json';cat=read(path);card=next(c for c in cat['experiments'] if c['id']==OUT.name);old=card['cannot_claim'];card.setdefault('revisions',[]).append(dict(date=datetime.now(timezone(timedelta(hours=3))).isoformat(),field='cannot_claim',old=old,reason='Явно сохранена неоднозначность прошлойGTединственнойчужойпары; бинарнаяметкаи оценки не менялись.'));card['cannot_claim']=old+' '+note;card['evidence'].append(dict(kind='file',ref=f'runs/{OUT.name}/label_grade.json',claim='Неоднозначность прошлой личности и неопределённая строгаяAUROC'));path.write_text(json.dumps(cat,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
path=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';s=path.read_text(encoding='utf8');now=datetime.now(timezone(timedelta(hours=3)));entry=f'**{now:%d.%m.%Y, %H:%M} MSK** — Уточнена степень подтверждения единственной чужой пары переноса.\n\n*Получено (`{OUT.name}`).* {note} Источник `runs/{OUT.name}/label_grade.json`, исходныеIoUв`audit.json`.\n\n*Вывод и решение.* Не заменять условный45/6-парныйдиагностическийрезультат утверждением о строгой независимой проверке личности. Все оценки и исходные метки сохранены. В карточке прежнее ограничение перенесено вrevisions[], новое дополнило его; статусне повышен.\n\n';path.write_text(s.replace('## Журнал изменений\n','## Журнал изменений\n\n'+entry,1),encoding='utf8')
print('Recorded label-grade limitation without changing any score')
