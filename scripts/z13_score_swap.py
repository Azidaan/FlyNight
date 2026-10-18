"""Saved-row factorial check: swap confidence without changing any predicted center."""
import argparse
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime,timedelta,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
RUN=ROOT/'runs/z13_score_swap_20261018'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def dump(path,value):
    from tracking.z13_checkpoint import atomic_json
    atomic_json(path,value)


def key(row):return row['video'],str(row['label']),row['kind']


def mix(centers,scores):
    score_by_key={key(r):r['p_present'] for r in scores}
    if len(score_by_key)!=len(scores) or len({key(r) for r in centers})!=len(centers) or {key(r) for r in centers}!=set(score_by_key):
        raise ValueError('Unmatched or duplicate rows')
    return [dict(r,p_present=score_by_key[key(r)]) for r in centers]


def selected_keys(rows,q):
    count=max(1,math.ceil(len(rows)*q))
    # Stable tie handling as in the existing coverage table. No outcomes used.
    return {key(r) for r in sorted(rows,key=lambda r:-r['p_present'])[:count]}


def assess(variants):
    from scripts.z13_continue import paired,per_video_useful
    out={}
    for q in (.01,.02,.05):
        summary={};pv={}
        for name,rows in variants.items():
            chosen=selected_keys(rows,q);selected=[r for r in rows if key(r) in chosen];known=[r for r in selected if r['kind']>=0]
            useful=sum(bool(r.get('useful')) for r in known)
            summary[name]=dict(selected=len(selected),known=len(known),unknown=len(selected)-len(known),useful=useful,known_nonuseful=len(known)-useful,
                precision_known=useful/len(known) if known else None)
            gated=[dict(r,useful=bool(r.get('useful')) and key(r) in chosen) for r in rows]
            pv[name]=per_video_useful(gated)
        comparisons={}
        for baseline in ('old_center_old_score','new_center_new_score'):
            comparisons['new_center_old_score-'+baseline]=paired(pv['new_center_old_score'],pv[baseline])
        out[f'{q:.2f}']=dict(summary=summary,paired_selected_success=comparisons)
    out['primary_pass']=all(p['ci95'][0]>0 for p in out['0.01']['paired_selected_success'].values())
    return out


def prepare():
    if RUN.exists():raise RuntimeError('Already prepared; inspect instead of overwriting')
    paths={'train':dict(old='runs/z13_continue_20261017/baseline/rows.json',new='runs/z13_target_center_20261018/D/center_rows.json',r1='runs/z13_target_center_20261018/R1/center_rows.json'),
           'dev':dict(old='runs/z13_target_center_dev_20261018/aggregate/D_prior_rows.json',new='runs/z13_target_center_dev_20261018/aggregate/D_center_rows.json',r1='runs/z13_target_center_dev_20261018/aggregate/R1_center_rows.json')}
    files={str(p):hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for group in paths.values() for p in group.values()}
    files['scripts/z13_score_swap.py']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    now=datetime.now(timezone(timedelta(hours=3)))
    protocol=dict(id='z13_score_swap_v1',created_at=now.isoformat(),sources=paths,sha256=files,
        action='Retain new D centers, replace only p_present with the old D readout score. Full 2x2 factorial; no fitting.',
        primary='At fixed global 1% rank coverage: lower95%video-bootstrap CI of selected useful centers per all positive cases must exceed0 against both old-center/old-score and new-center/new-score. Secondary2%/5% cannot rescue failure of primary.',
        metric='selected_success_fraction_positive: unselected positive cases count as no useful intervention; known negative and unknown selections reported separately. Not HOTA or calibrated risk.',
        selection='Train26 only for advancement. Open dev outcome only if the fixed train criterion passes; dev already used elsewhere, so exploratory. No parameter/threshold search.',
        controls='Old center/new score completes factorial. R1 center/own score is contextual no-FlyVis reference only: lacks an old spatial head, hence not a matched control of this two-readout intervention.',
        counterargument='Old presence score may recognize the object while failing to assess the accuracy of a different center; sparse1% selections may yield uncertain intervals.',
        decision_value='Decide whether a simple internal score reuse is worth a full matched intervention check; failure does not diagnose the whole representation or close FlyVis.',
        literature=['https://arxiv.org/abs/1811.07628','https://arxiv.org/abs/2003.12565'],
        scope='Saved predictions only, CPU, no core training, GPU, videos, HOTA, heldout75 or model scheduling.',budget_seconds=120)
    RUN.mkdir();dump(RUN/'protocol.json',protocol);dump(RUN/'status.json',dict(state='ready',stage='saved-row test prepared',done=0,total=1))
    gpath=ROOT/'experiments/metrics_glossary.json';glossary=read(gpath)
    glossary['metrics']['selected_success_fraction_positive']=dict(title='Доля положительных случаев с полезным выбранным ответом',ru='Для каждого видео: число выбранных полезных ответов, делённое на число всех положительных случаев; затем среднее по видео. Невыбранный случай не даёт полезного вмешательства. Отбор задаётся баллом без разметки.',better='higher',unit='доля',range=[0,1],tool='scripts/z13_score_swap.py',caveat='Диагностика выбора; отдельно считать выбранные отрицательные и неизвестные. Не точность среди отобранных, не HOTA и не оценка полного риска.',source='runs/z13_score_swap_20261018/protocol.json')
    gpath.write_text(json.dumps(glossary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    path=ROOT/'experiments/project_history.json';cat=read(path)
    cat['experiments'].append(dict(id=protocol['id'],title='З13: прежний балл при новых центрах D',area='flyvis',state='ready',started=now.date().isoformat(),
        question='Поможет ли прежняя оценка присутствия выбирать полезные центры новой D?',hypothesis='Старый выход присутствия может ранжировать новые центры лучше высоты нового пика; это гипотеза, а не установленная причина.',
        expected='Сохранённые train26: факторный контроль2×2 по источнику центра и балла. Основной отбор1%, дополнительные2%/5%; оба нижних повидеоДИ выше0. Без подбора параметров и повторного извлечения. Dev читается для результата только после прохождения train-критерия. R1 — ориентир, не равный по ёмкости двухголовый контроль.',
        acceptance=[dict(criterion='Строки совпадают, замена балла не меняет центры и исходы',met=None),dict(criterion='На train26 при отборе1% оба основных повидеоДИ выше0',met=None)],
        tags=['Z13','fly-problem-P8'],runs=['runs/z13_score_swap_20261018'],files=['scripts/z13_score_swap.py'],
        evidence=[dict(kind='json',ref='runs/z13_score_swap_20261018/protocol.json',claim='Условия дешёвой проверки до расчёта')]))
    path.write_text(json.dumps(cat,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    journal('постановка', 'Автор разрешил продолжать тесты. Одно действие — прежний балл присутствия при новых центрах D; условие и факторные контроли записаны до расчёта в runs/z13_score_swap_20261018/protocol.json. Основной отбор1%, вторичные2%/5%. Дешёвая проверка по готовым строкам; dev-результат не открывается при непрохождении train. R1 не является равным по ёмкости контролем двух считывателей, поэтому вклад FlyVis этим тестом не доказывается. CPU до120с; без новых обучения и обработки видео. Принцип разделения локализации/оценки качества поддерживают ATOM и PrDiMP, величина эффекта из них не переносится.')


def journal(title,text):
    path=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';plan=path.read_text(encoding='utf-8');anchor='## Журнал изменений\n';assert anchor in plan
    now=datetime.now(timezone(timedelta(hours=3)))
    path.write_text(plan.replace(anchor,anchor+f'\n**{now:%d.%m.%Y, %H:%M} MSK** — {title} `z13_score_swap_v1`\n\n'+text+'\n',1),encoding='utf-8')


def run():
    protocol=read(RUN/'protocol.json')
    if (RUN/'report.json').exists():raise RuntimeError('Completed; do not repeat')
    for name,digest in protocol['sha256'].items(): assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    results={}
    for split in ('train','dev'):
        if split=='dev' and not results['train']['primary_pass']:break
        dump(RUN/'status.json',dict(state='running',stage=split+' saved rows',done=0,total=1))
        paths=protocol['sources'][split];old,new,r1=[read(ROOT/paths[k]) for k in ('old','new','r1')]
        variants=dict(old_center_old_score=old,new_center_new_score=new,new_center_old_score=mix(new,old),old_center_new_score=mix(old,new),R1_reference=r1)
        assert {key(r) for r in r1}=={key(r) for r in new}
        results[split]=assess(variants)
    report=dict(results=results,dev_evaluated='dev' in results,decision='advance_to_matched_check' if results['train']['primary_pass'] else 'score_reuse_not_confirmed_at_primary_coverage',limitations=protocol['controls']+' '+protocol['metric'])
    dump(RUN/'report.json',report);dump(RUN/'status.json',dict(state='completed',stage='saved-row report written',done=1,total=1))


def finish():
    if (RUN/'recorded.json').exists():return
    rep=read(RUN/'report.json');path=ROOT/'experiments/project_history.json';cat=read(path)
    card=next(c for c in cat['experiments'] if c['id']=='z13_score_swap_v1')
    counts=rep['results']['train']['0.01']['summary'];pairs=rep['results']['train']['0.01']['paired_selected_success']
    source='runs/z13_score_swap_20261018/report.json'
    obtained=source+' (results.train.0.01.summary): '+'; '.join(f"{k}: полезных {v['useful']}, известных {v['known']}, неизвестных {v['unknown']}, выбрано {v['selected']}" for k,v in counts.items())+'. '
    obtained+=source+' (results.train.0.01.paired_selected_success): '+json.dumps(pairs,ensure_ascii=False)+'. Dev-результат '+('проверен после train-критерия.' if rep['dev_evaluated'] else 'не открывался, поскольку train-критерий не выполнен.')
    passed=rep['results']['train']['primary_pass']
    card.update(state='completed_positive' if passed else 'inconclusive',done=datetime.now().date().isoformat(),obtained=obtained,
        interpretation='Дешёвый критерий повторного использования прежнего балла '+('выполнен.' if passed else 'не выполнен. Это не диагноз представления и не закрытие линии.'),
        decision=rep['decision'],cannot_claim='Нет HOTA, доказательства пользы стенду, полной оценки риска на неизвестных исходах или равного по ёмкости контроля без FlyVis. Доли2%/5% не заменяют основной критерий1%.')
    for a,met in zip(card['acceptance'],[True,passed]):a.update(met=met,evidence=source)
    card['evidence'].append(dict(kind='json',ref=source,claim='Полная факторная проверка по сохранённым строкам'))
    path.write_text(json.dumps(cat,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    journal('итог','*Получено.* '+obtained+'\n\n*Вывод.* '+card['interpretation']+'\n\n*Решение.* '+card['decision']+'. HOTA не измерена.')
    dump(RUN/'recorded.json',dict(recorded=True))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','run','finish']);args=ap.parse_args()
    {'prepare':prepare,'run':run,'finish':finish}[args.mode]()
