"""Record a retrospective evidence review, preserving all earlier decisions."""
from pathlib import Path
import sys,json,re,hashlib,subprocess
from datetime import datetime
from zoneinfo import ZoneInfo
ROOT=Path(__file__).resolve().parents[1]; OUT=Path(__file__).resolve().parent
ID='identity_weekly_causal_review_20261112'
sys.path.insert(0,str(ROOT))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def save(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT)

def main():
    assert not (OUT/'recording.json').exists()
    report=OUT/'РАЗБОР.md';text=report.read_text(encoding='utf8')
    refs=sorted(set(re.findall(r'\]\((E:/Dissertation/[^)]+)\)',text)))
    manifest={}
    for ref in refs:
        p=Path(ref); assert p.is_file(),p
        manifest[p.relative_to(ROOT).as_posix()]=sha(p)
    inv=read(OUT/'inventory.json'); assert inv['count']==97
    assert read(OUT/'receipt.json')['passed']
    assert read(OUT/'event_witnesses.json')['passed']
    assert read(OUT/'decision_decomposition.json')['passed']
    assert read(OUT/'gate_inventory.json')['reasons']['outside_observed_training_gap']==dict(unknown=2081,own=12,foreign=12)
    before=OUT/'before';before.mkdir()
    paths=['experiments/project_history.json','experiments/obsidian_experiment_topics.json',
           'experiments/system_problems.json','dissertation/ЕДИНЫЙ_ПЛАН.md']
    original={}
    for n in paths:
        p=ROOT/n; original[n]=p.read_bytes();(before/p.name).write_bytes(original[n])
    stamp=datetime.now(ZoneInfo('Europe/Moscow'))
    summary=('Разобраны 97 карточек за 05–12.11: 42 отрицательные, 24 неопределённые, 1 положительная и 30 инфраструктурных '
             '(research_weekly_causal_audit_20261112/inventory.json). Сохранённые оценки пяти случаев разложены без обучения '
             '(decision_decomposition.json). В C34 рабочий допуск давности исключил 12 известных своих и 12 чужих '
             '(gate_inventory.json). В двух контрольных вариантах найдена пограничная стоимость 0,800000011920929>0,8; '
             'у 1102 на кадре 264 чужая рамка прошла штатное назначение, которое проверка личности пропускает '
             '(event_witnesses.json; runs/domain_matched_native_20261109/id_fate.json). '
             'Источники и пределы каждого вывода: research_weekly_causal_audit_20261112/РАЗБОР.md.')
    interpretation=('Разные подтверждённые причины относятся к разным событиям и компонентам. Не установлена единая причина всех '
                    'проблем; причина неверного зрительного порядка и первый момент части подмен остаются открытыми. '
                    'Численная корректность границы, расширение рабочих пар и критерий продолжения исследования — разные вопросы.')
    decision=('Опыт согласия остаётся остановленным с отрицательным исходом. Пороги, ядро и трекер не менять и не запускать. '
              'По просьбе автора отдельно рассмотрены рабочие ограничения и требования продолжения; рекомендации нового опыта '
              'не являются изменением прежних критериев или разрешением полного запуска.')
    cannot=('Не доказаны абсолютно все причины недели, полезность ослабления допуска, преимущество ядра FlyVis или новый HOTA. '
            'Неизвестные пары не размечены автоматически. Перечень карточек не означает повтор всех первичных расчётов. '
            'Это ретроспективный разбор, а не заранее зарегистрированный новый эксперимент.')
    d=json.loads(original[paths[0]]);assert all(x['id']!=ID for x in d['experiments'])
    d['experiments'].append(dict(id=ID,title='Причины потери личности за неделю — вычисления, назначения и строгость допусков',
        area='flyvis',state='infrastructure',started=stamp.date().isoformat(),done=stamp.date().isoformat(),
        question='Какие причины научных и технических проблем 5–12 ноября подтверждены, а какие остаются гипотезами?',
        hypothesis='Ретроспективный вопрос без новой экспериментальной гипотезы.',
        expected='Разделить точные механизмы и незакрытые версии; отдельно оценить рабочие и исследовательские допуски.',
        acceptance=[dict(criterion='Ретроспективная полнота: перечислены все карточки периода и адреса доказательств.',met=True,evidence='research_weekly_causal_audit_20261112/inventory.json'),
                    dict(criterion='Готовые оценки восстановлены; новые карты, обучение, подбор порогов и полный трекер не запускались.',met=True,evidence='research_weekly_causal_audit_20261112/receipt.json; research_weekly_causal_audit_20261112/decision_decomposition.json'),
                    dict(criterion='Все причины всех научных и технических проблем установлены без неопределённости.',met=False,evidence='research_weekly_causal_audit_20261112/РАЗБОР.md, раздел 8')],
        obtained=summary,interpretation=interpretation,decision=decision,cannot_claim=cannot,
        evidence=[dict(kind='file',ref='research_weekly_causal_audit_20261112/'+n,claim=c) for n,c in [
            ('РАЗБОР.md','Механизмы, рабочие версии и отдельное рассмотрение допусков'),
            ('inventory.json','Все карточки выбранного периода'),('decision_decomposition.json','Арифметика текущих пяти оценок'),
            ('event_witnesses.json','Сохранённые матрицы, назначения и пограничная стоимость'),
            ('gate_inventory.json','Известные свои и чужие вне рабочего допуска'),('evidence_manifest.json','Снимок источников при завершении разбора')]]))
    topics=json.loads(original[paths[1]])
    topics['topics'][ID]=['Стенд и вычисления','Данные и воспроизводимость','Причины отказов и границы доказательств']
    problems=json.loads(original[paths[2]])
    for ident in ['С1','С2','С5','С6']:
        p=next(x for x in problems['problems'] if x['id']==ident)
        p['cases'].append(dict(date=stamp.date().isoformat(),component='Вся авторская система; ретроспективный причинный разбор',
            fact=summary,hypothesis='Оставшиеся гипотезы разделены в разделе 8 отчёта; единственная общая причина не установлена.',
            discriminating_check='Для подмены истории нужны прямые промежуточные метки и матрица первого ошибочного назначения; для зрительного порядка — отдельная проверка соответствия физических деталей.',
            limitation=cannot,evidence=['research_weekly_causal_audit_20261112/РАЗБОР.md','research_weekly_causal_audit_20261112/event_witnesses.json'],source_sha256=sha(report)))
        if ID not in p['cards']:p['cards'].append(ID)
        p['updated']=stamp.date().isoformat()
    entry='\n**'+stamp.strftime('%d.%m.%Y, %H:%M MSK')+'** — Причины проблем недели и строгость допусков.\n\n'
    entry+='*Постановка.* После отрицательного опыта автор поручил разобрать весь путь, включая технические ошибки, и отдельно рассмотреть рабочие ограничения и требования продолжения. Это разбор сохранённых данных, без нового эксперимента.\n\n'
    entry+='*Получено (`'+ID+'`).* '+summary+'\n\n*Вывод.* '+interpretation+' '+cannot+'\n\n*Решение.* '+decision+'\n'
    plan=original[paths[3]].decode('utf8');marker='## Журнал изменений\n';assert plan.count(marker)==1
    plan=plan.replace(marker,marker+entry,1)
    changed=[d,topics,problems]
    for n,new in zip(paths[:3],changed):
        assert (ROOT/n).read_bytes()==original[n];save(ROOT/n,new)
    assert (ROOT/paths[3]).read_bytes()==original[paths[3]];(ROOT/paths[3]).write_text(plan,encoding='utf8')
    (OUT/'journal_addition.txt').write_text(entry,encoding='utf8')
    save(OUT/'evidence_manifest.json',dict(created_msk=stamp.isoformat(),files=manifest,
        scope='Hashes at completion of this review, not retroactively assigned to historical runs. Original run receipts remain their provenance.'))
    save(OUT/'recording.json',dict(passed=True,id=ID,head=git('rev-parse','HEAD').decode().strip(),
        sources_at_start={n:hashlib.sha256(raw).hexdigest() for n,raw in original.items()},
        report_sha256=sha(report),scientific_sentence_review=dict(component_vs_object=True,no_redundant_sections=True,observed_vs_proposed_explicit=True),
        all_causes_established=False,history_preserved=True))
    print('RECORDED_REVIEW',len(refs),'source links')

if __name__=='__main__':main()
