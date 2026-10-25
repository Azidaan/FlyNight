"""Preregister and evaluate saved-action bounds before selecting a real pilot."""
from pathlib import Path
import sys
import json
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.z13_candidate_reserve_20261021 import read, write, sha, now
from scripts.z13_continue import paired
from tracking.z13_action_feasibility import conditional_net, required_recovery
RUN = ROOT / 'runs/z13_math_priority_20261025'
IDENT = 'z13_math_priority_v1'


def journal(title, text):
    from scripts import z13_candidate_reserve_20261021 as m
    old = m.IDENT
    try:
        m.IDENT = IDENT; m.journal(title, text)
    finally:
        m.IDENT = old


def prepare():
    assert not RUN.exists()
    files = ['runs/z13_proposal_geometry_20261025/recovery1/report.json',
        'runs/z13_candidate_identity_20261021/report.json',
        'runs/z13_candidate_ranker_20261021/report.json',
        'runs/z13_selected_box_value_20261025/report.json',
        'scripts/z13_math_priority_20261025.py', 'tracking/z13_action_feasibility.py']
    p = dict(id=IDENT, created=now().isoformat(), research_only=True, device='cpu',
        authorization='Автор25.10: ранжировать варианты, провести математические проверки, затем настоящий прогон.',
        action='Choose one different internal FlyVis readout action before another learner.',
        sources={n:sha(ROOT/n) for n in files},
        proxies={'KeepTrack_three_peaks':'Dpeak3', 'region_center':'centers_eye',
            'full_box_readout':'boxes_eye', 'wide_search_full_box':'boxes_frame'},
        interpretation='These are saved one-step action-space ceilings using truth, not implementations or predictions of named papers. Full learned association over all real regions shares the box ceiling, not the three-peak ceiling. Broad box support is optimistic relative to strict trainable masks.',
        conditional='For video reserve O, assumed recovery r and assumed harm frequency f among other known cases: G=r*O-f*(1-O). r/f are scenarios, not estimated future skills. Report means and paired-video95CI under assumptions.',
        criteria='Admit a non-temporal full-box label/model pilot only when saved eye-box ceiling mean>=.02 and lower95CI>0 and increment over eye-center lower>0; real strict-support gate before training must separately satisfy>=.02 and lower>0 excluding NULL-only reserve. No HOTA forecast. No acceptance threshold changes.',
        rank='1 actual full-box supervision with existing internal head;2 learned temporal association over real regions;3 spatially transferred state;4 wider input. First candidate has cached features and untrained earlier attempt; temporal/spatial variants lack verified matched sequential feature data. Existing three-peak implementations stay failed.',
        assumptions='Repeated train26, one-step baseline-state outcomes, perfect choice can abstain, no closed-loop feedback. Conditional math has no measured learned r/f and is not a confidence interval for a future model.')
    write(RUN/'protocol.json', p)
    c=read(ROOT/'experiments/project_history.json')
    assert not any(x['id']==IDENT for x in c['experiments'])
    c['experiments'].append(dict(id=IDENT,title='Предварительная математика резерва действий и цены ошибок',area='flyvis',state='planned',started=now().strftime('%Y-%m-%d'),
        question='Какой доступный выход имеет резерв и какой точности выбора потребует полезное вмешательство?',hypothesis='Полная рамка имеет больший резерв, но потребует подавления вреда; три старых пика могут не достигать прежнего порога даже при идеальном выборе.',expected=p['conditional'],
        acceptance=[dict(criterion=p['criteria'],met=None)],obtained='',interpretation='',decision='',cannot_claim=p['assumptions'],tags=['fly-problem-P8'],
        evidence=[dict(kind='file',ref='runs/z13_math_priority_20261025/protocol.json',claim='Условия и пределы математической проверки до расчёта')]))
    write(ROOT/'experiments/project_history.json',c)
    journal('постановка математической проверки перед новым прогоном',p['action']+' '+p['criteria']+' '+p['conditional']+' '+p['assumptions'])


def run():
    p=read(RUN/'protocol.json')
    for n,h in p['sources'].items():assert sha(ROOT/n)==h
    g=read(ROOT/'runs/z13_proposal_geometry_20261025/recovery1/report.json')
    results={};scenarios={};zeros={v:0. for v in g['per_video']['Dpeak3']}
    for name,key in p['proxies'].items():
        values=g['per_video'][key];r=paired(values,zeros);mean=r['mean_per_video']
        results[name]=dict(source_key='per_video.'+key,**r,required_recovery_for_2pp_at_1pct_false_harm=required_recovery(mean),
            cannot_reach_2pp_even_with_perfect_selection=mean<.02)
        scenarios[name]=[]
        for recovery in (.25,.5,.75,1.):
            for harm in (0.,.005,.01,.02):
                val={v:float(conditional_net(x,recovery,harm)) for v,x in values.items()}
                scenarios[name].append(dict(assumed_recovery=recovery,assumed_false_harm=harm,**paired(val,zeros)))
    diff=paired(g['per_video']['boxes_eye'],g['per_video']['centers_eye'])
    b=results['full_box_readout'];admit=b['mean_per_video']>=.02 and b['ci95'][0]>0 and diff['ci95'][0]>0
    report=dict(complete=True,diagnostic_only=True,not_model_prediction=True,not_hota=True,action_bounds=results,
        box_minus_center=diff,conditional_scenarios=scenarios,admit_strict_box_label_pilot=admit,
        selected_next='actual full-box supervision' if admit else 'no pilot admitted',limits=p['assumptions'])
    write(RUN/'report.json',report)
    lines=['# 1. Приоритет после математической проверки','',
        'Это границы сохранённых действий и условные сценарии. Качество ещё не обученной модели и HOTA не предсказываются.','',
        '| Действие | Идеальный резерв, п.п. [95% ДИ] | Нужная доля использования резерва при 1% вреда среди остальных |',
        '|---|---|---|']
    for name,r in results.items():
        lines.append(f"| {name} | {r['mean_per_video']*100:.3f} [{r['ci95'][0]*100:.3f}; {r['ci95'][1]*100:.3f}] | {r['required_recovery_for_2pp_at_1pct_false_harm']*100:.1f}% |")
    lines+=['','## 1.1. Решение','',p['rank'],'',
        'KeepTrack на всех реальных областях не ограничен тремя прежними пиками. Математика трёх пиков не отвергает весь KeepTrack. Выше ставится опыт с уже готовыми признаками и ещё не проверенными полными рамочными метками.',
        '', 'Реальная проверка: новый запрос автора допускает отдельный бюджет после прежней остановки по цене. Старое окно не возобновляется; прежние результаты и ошибки сохраняются. Повторно используются четыре готовых видео. Перед обучением требуется резерв именно строгого набора областей, без вклада одного NULL.',
        '',p['assumptions']]
    (RUN/'РЕЗУЛЬТАТ.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    cat=read(ROOT/'experiments/project_history.json');card=next(x for x in cat['experiments'] if x['id']==IDENT)
    card.update(state='completed_positive' if admit else 'completed_negative',done=now().strftime('%Y-%m-%d'),
        obtained='runs/z13_math_priority_20261025/report.json: action_bounds='+str(results)+';box_minus_center='+str(diff),
        interpretation=p['interpretation'],decision=report['selected_next']+'; перед обучением проверить строгий набор. Значения r/f сценарные, не прогноз качества.')
    card['acceptance'][0].update(met=admit,evidence='runs/z13_math_priority_20261025/report.json')
    card['evidence'].append(dict(kind='json',ref='runs/z13_math_priority_20261025/report.json',claim='Парные границы и условная чувствительность'))
    write(ROOT/'experiments/project_history.json',cat)
    journal('итог математической проверки действий',card['obtained']+'\n\n*Вывод.* '+card['interpretation']+'\n\n*Решение.* '+card['decision'])
    print(json.dumps({k:round(v['mean_per_video']*100,4) for k,v in results.items()}))


if __name__=='__main__':
    if '--prepare' in sys.argv:prepare()
    elif '--run' in sys.argv:run()
