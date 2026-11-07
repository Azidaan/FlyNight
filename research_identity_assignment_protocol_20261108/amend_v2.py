"""Amend an unexecuted draft, preserving the complete first draft and receipts."""
import hashlib
import json
from datetime import datetime,timedelta,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=Path(__file__).resolve().parent


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p,value):
    with Path(p).open('x',encoding='utf-8',newline='\n') as f:json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n')


def main():
    stamp=datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=3)))
    p=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))
    p['id']='identity_geometry_competition_protocol_v2_20261108'
    p['prepared']=stamp.isoformat()
    p['supersedes']=dict(path='protocol.json',sha256=sha(OUT/'protocol.json'),
        reason='Before any scientific run: add full-tracker comparison G_FLYVIS-B0, so a feature variant cannot be accepted merely by beating a harmful geometry variant. Strengthen identity/harm guards against every primary comparator.')
    p['inference']['primary_comparisons'].insert(1,['G_FLYVIS','B0'])
    p['inference']['family_size']=8;p['inference']['ci_levels_percent']=[95,99.375]
    p['inference']['system_flyvis_acceptance']='Corrected lower HOTA bound G_FLYVIS-B0>0, in addition to all6 content-control comparisons.'
    p['inference']['harm_guard']='G vsB0; G_FLYVIS vsB0 and each of its6 content controls: nonnegative mean AssA/IDF1 changes and no increase in total IDSw,FP,FN.'
    write(OUT/'protocol_v2.json',p)
    doc=(OUT/'ПРОТОКОЛ.md').read_text(encoding='utf-8')
    old='Заранее задана семья из семи сравнений HOTA: G−B0 и G_FLYVIS минус G, G_SHIFT, G_IMAGE, G_R1, G_DIS, G_PERM.'
    new='Заранее задана семья из восьми сравнений HOTA: G−B0, G_FLYVIS−B0 и G_FLYVIS минус G, G_SHIFT, G_IMAGE, G_R1, G_DIS, G_PERM.'
    assert doc.count(old)==1;doc=doc.replace(old,new).replace('99,285714%','99,375%')
    old='Для принятия варианта дополнительно требуются неотрицательные средние разности AssA и IDF1 и отсутствие роста общего числа IDSw, FP и FN относительно соответствующего основного контроля.'
    new='Для принятия версии с FlyVis в стенд также требуется положительная скорректированная нижняя граница G_FLYVIS−B0. Для G относительно B0, а для G_FLYVIS относительно B0 и каждого из шести содержательных контролей дополнительно требуются неотрицательные средние разности AssA и IDF1 и отсутствие роста общего числа IDSw, FP и FN.'
    assert doc.count(old)==1;doc=doc.replace(old,new)
    doc=doc.replace('Это подготовленный протокол следующей проверки,','Это версия 2 подготовленного протокола следующей проверки,',1)
    doc=doc.replace('## 1. Статус и вопрос','Версия 2 заменяет первоначальный черновик до регистрации и до любых опытов. Изменение критериев и причина сохранены в `protocol_v2.json/supersedes`.\n\n## 1. Статус и вопрос',1)
    with (OUT/'ПРОТОКОЛ_v2.md').open('x',encoding='utf-8',newline='\n') as f:f.write(doc)
    write(OUT/'CURRENT.json',dict(current_protocol='protocol_v2.json',current_document='ПРОТОКОЛ_v2.md',
        status='prepared_not_executed_not_launch_ready',previous_protocol='protocol.json',previous_document='ПРОТОКОЛ.md',
        scientific_runs=0,reason=p['supersedes']['reason']))
    entry=f"**{stamp.strftime('%d.%m.%Y, %H:%M MSK')}** — До регистрации и запуска уточнён черновик протокола сохранения ID: действующая версия `research_identity_assignment_protocol_20261108/protocol_v2.json`, `ПРОТОКОЛ_v2.md`, указатель `CURRENT.json`. Добавлено обязательное сравнение G_FLYVIS с исходным B0, чтобы выигрыш над вредной геометрической версией не считался принятием в стенд; семья теперь из 8 сравнений HOTA, интервалы Бонферрони 99,375%. Ограничения вреда по AssA/IDF1/IDSw/FP/FN заданы относительно каждого основного контроля. Первая версия, её проверки и входные манифесты сохранены. Новых научных расчётов нет; каталог, очередь, код трекера и граф не менялись.\n\n"
    write(OUT/'journal_v2_entry.json',dict(entry=entry))
    path=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';text=path.read_text(encoding='utf-8');anchor='## Журнал изменений\n\n'
    assert text.count(anchor)==1;path.write_text(text.replace(anchor,anchor+entry,1),encoding='utf-8')
    files={p.name:sha(p) for p in OUT.iterdir() if p.is_file() and p.name!='checksums_v2.json'}
    write(OUT/'checksums_v2.json',dict(created=stamp.isoformat(),files=files))


if __name__=='__main__':main()
