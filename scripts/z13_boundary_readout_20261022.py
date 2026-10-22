"""One predeclared response-support refusal check on immutable train26 maps."""
from pathlib import Path
from datetime import datetime
import sys
import time
import math
import subprocess
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_center_hota_20261021 import read,write,sha,now
RUN=ROOT/'runs/z13_boundary_readout_20261022';SOURCE=ROOT/'runs/z13_query_memory_20261021'
IDENT='z13_boundary_readout_v1';T0=time.monotonic()


def journal(title,text):
    p=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';s=p.read_text(encoding='utf-8');mark='## Журнал изменений\n';assert s.count(mark)==1
    p.write_text(s.replace(mark,mark+f'\n**{now():%d.%m.%Y, %H:%M} MSK** — {title} `{IDENT}`\n\n'+text+'\n',1),encoding='utf-8')


def prepare():
    if RUN.exists():raise RuntimeError('Existing run must be preserved')
    files=['tracking/z13_boundary_readout.py','tests/test_z13_boundary_readout.py','scripts/z13_boundary_readout_20261022.py',
           'scripts/z13_bounded_stage_20261022.py','scripts/z13_stage_progress_20261022.ps1',
           'scripts/z13_center_hota_20261021.py','scripts/z13_candidate_reserve_20261021.py','scripts/z13_hota_deadline_guard.py',
           'tracking/z13_ridge_localizer.py','tracking/z13_readout.py','scripts/z13_target_model.py',
           'scripts/z13_score_swap.py','scripts/z13_continue.py','scripts/z13_query_memory_20261021.py',
           'runs/z13_local_peak_20261021/geometry.json','runs/z13_first_divergence_20261022/report.json']
    hashes={}
    for arm in ('C','D','R1'):
        for name in ('last_rows.json','evaluate/labels.json','cache_manifest.json'):files.append(f'runs/z13_query_memory_20261021/{arm}/{name}')
        files.extend([f'runs/z13_parallel_next_20261018/utility/{arm}_rows.json',f'runs/z13_map_utility_20261018/{arm}/predicted_scores.json'])
        hashes.update({n:h for n,h in read(SOURCE/arm/'cache_manifest.json').items() if '/evaluate/' in n})
    hashes.update({n:sha(ROOT/n) for n in files})
    RUN.mkdir()
    protocol=dict(id=IDENT,created=now().isoformat(),authorization='Author22.10 evening: Продолжи. Работай в автономном режиме. Existing autonomous necessary work and publication authorization renewed.',
        device='cpu',budget_seconds=600,save_at='2026-10-23T07:35:00+03:00',hard_stop='2026-10-23T07:39:30+03:00',
        action='Inside the readout replace unconditional availability by refusal if the winning hexagonal column lacks any of its six neighbors. Preserve the original center and raw peak score; do not move the winner or change the core.',
        method='One fixed outer ring only, degree<6 with self-filled neighbor entries excluded. No grid, label fitting, new lambda or normalization. On each arm8326saved rows: select the same84answers by original score among eligible rows; raw/geometry6/map14 controls also select84. Stable existing tie order; eligible count must suffice. No pure coverage reduction is counted as success.',
        data='Existing repeated exploratory train26 only. No dev35 calculation, no HOTA, no protected75, no model training/extraction.',
        primary=['All C/D/R1 raw centers<1e-6px and scores<1e-9 to saved last rows; all maps/hash unchanged.',
                 'D boundary-D raw, D boundary-D geometry6 and D boundary-D map14: all paired-video95%lower>0 for selected_success_fraction_positive at1%equal quota.',
                 'D boundary-R1 boundary paired-video95%lower>0. Controls keep the same history and maps; no anatomy claim.'],
        counterargument='A real target can be at the boundary. Removing difficult answers may lower coverage or improve only R1; equal quota, prior stronger selectors, absolute useful/known-negative/unknown counts and C/D/R1 gains are required. One R1 dev event did not establish a D failure.',
        decision='Advance only if all main criteria pass; otherwise stop this one-ring refusal candidate, retaining centered readout. No dev retuning or claim all uncertainty modeling fails. This gate alone is not a new neural representation or confirmed tracker improvement.',
        alternatives_ranked=['Fixed one-ring response-support refusal: current cheapest isolated test.',
                             'Peak-to-sidelobe confidence or distractor-aware refusal: separate future design, no simultaneous tuning here.',
                             'Probabilistic/weighted target-model fitting: changes model objective; requires separate preregistration and compute estimate.'],
        references=[dict(url='https://github.com/visionml/pytracking/blob/master/pytracking/tracker/dimp/dimp.py',section='localize_advanced',supports='Explicit not_found/uncertain states and separated distractor handling. This code does not establish or validate our one-ring veto.'),
                    dict(url='https://arxiv.org/html/2003.12565v1',section='Abstract and sections2–4',supports='Scalar confidence lacks direct probabilistic interpretation; probabilistic target-state estimation is an alternative, not implemented here.')],
        sha256=hashes)
    write(RUN/'protocol.json',protocol);write(RUN/'status.json',dict(state='ready',stage='Фиксированная проверка внешнего кольца',done=0,total=3,updated=now().isoformat()))
    catalog=read(ROOT/'experiments/project_history.json');assert not any(c['id']==IDENT for c in catalog['experiments'])
    catalog['experiments'].append(dict(id=IDENT,title='З13: отказ от граничного пика при равной доле выбранных ответов',area='flyvis',state='ready',started='2026-10-22',done='',tags=['fly-problem-P8'],
        question='Поможет ли один фиксированный отказ на внешнем кольце выбирать полезные центры D сверх прежнего балла и прежних селекторов?',
        hypothesis='Неполная пространственная поддержка максимума может выделять часть ошибочных ответов; для D это ещё не установлено.',
        expected='Только CPU по сохранённым картам train26: по84ответа для каждого варианта; одна граница без подбора. Прежние центры/карты/ядро сохраняются.',
        acceptance=[dict(criterion=c,met=None) for c in protocol['primary']],summary='Один механизм отказа зарегистрирован до расчёта.',obtained='',interpretation='',decision='',
        cannot_claim='Не самостоятельная новизна FlyVis, не улучшение HOTA, не калиброванная вероятность. Один dev-пример R1 определил вопрос; повторно используемый train26 даёт поисковый результат. Главный критерий90%/50 не заменён.',
        metrics=[],commits=[],files=files[:3],evidence=[dict(kind='file',ref='runs/z13_boundary_readout_20261022/protocol.json',claim='Критерии/контроли/порядок до исхода')]))
    write(ROOT/'experiments/project_history.json',catalog)
    journal('постановка фиксированного отказа на внешнем кольце','*Постановка.* '+protocol['action']+'\n\n'+protocol['method']+'\n\nКритерии до исходов: '+str(protocol['primary'])+'\n\nГлавный контраргумент: '+protocol['counterargument']+'\n\nОпора: DiMP localize_advanced и PrDiMP2003.12565 поддерживают отделение локализации от неопределённости, но не доказывают наш запрет внешнего кольца. Предыдущие обучаемые14признаков и геометрия6 — обязательные сохранённые контроли, не повторяются. CPUдо600с; ближайшее окно автора заканчивается23.10 07:40МСК, контроллер ограничивает текущую стадию ещё раньше. Рабочие файлы/хеши зафиксированы; .gitignore по-прежнему отложен.')


def worker():
    from tracking.z13_readout import HexGeometry,to_frame
    from tracking.z13_boundary_readout import read_response,outer_ring
    from scripts.z13_query_memory_20261021 import load_maps
    from scripts.z13_continue import paired,per_video_useful
    from scripts.z13_score_swap import key,selected_keys
    p=read(RUN/'protocol.json')
    def check():
        if time.monotonic()-T0>p['budget_seconds']-10 or (RUN/'stop.request').exists() or now()>=datetime.fromisoformat(p['save_at']):raise TimeoutError('CPU budget/author window')
    def verify():
        for n,h in p['sha256'].items():check();assert sha(ROOT/n)==h,n
    verify();g=read(ROOT/'runs/z13_local_peak_20261021/geometry.json');geom=HexGeometry(np.asarray(g['points'],np.float32),g['eye_size']);edge=outer_ring(geom);assert edge.sum()==90
    summaries={};per={};reproduction={};manifest={}
    for ai,arm in enumerate(('C','D','R1')):
        write(RUN/'status.json',dict(state='running',stage='Сохранённые карты и равный отбор84',arm=arm,done=ai,total=3,updated=now().isoformat()))
        maps,hashes=load_maps(SOURCE/arm/'evaluate');manifest.update(hashes)
        labels=read(SOURCE/arm/'evaluate/labels.json');raw=read(SOURCE/arm/'last_rows.json');by={r['label']:r for r in raw};assert len(by)==len(labels)==len(maps)==8326
        boundary=[];dc=ds=0.;eligible=0
        for l in labels:
            check();score=maps[l['label']][-1];center,confidence,allowed=read_response(score,geom)
            c=np.asarray(to_frame(center,l['crop'],g['eye_size']));r=by[l['label']]
            assert r['video']==l['video'] and r['kind']==l['kind']
            dc=max(dc,float(np.max(np.abs(c-r['center']))));ds=max(ds,abs(confidence-r['p_present']))
            eligible+=allowed;boundary.append(dict(r,p_present=confidence if allowed else -1e30,boundary_refused=not allowed))
        assert dc<1e-6 and ds<1e-9 and eligible>=84
        reproduction[arm]=dict(rows=len(raw),max_center_px=dc,max_score=ds,eligible=eligible,outer_ring_columns=int(edge.sum()))
        geo=read(ROOT/f'runs/z13_parallel_next_20261018/utility/{arm}_rows.json')['geometry']
        oldscores={(r['video'],str(r['label'])):r['score'] for r in read(ROOT/f'runs/z13_map_utility_20261018/{arm}/predicted_scores.json')}
        full=[dict(r,p_present=oldscores[r['video'],str(r['label'])]) for r in raw]
        variants=dict(raw=raw,boundary=boundary,geometry6=geo,map14=full)
        expected={key(r) for r in raw};assert all({key(r) for r in rows}==expected for rows in variants.values())
        for rows in variants.values():
            assert all(np.max(np.abs(np.asarray(r['center'])-by[r['label']]['center']))<.5 and bool(r.get('useful'))==bool(by[r['label']].get('useful')) for r in rows)
        for mode,rows in variants.items():
            chosen=selected_keys(rows,.01);assert len(chosen)==84
            sel=[r for r in rows if key(r) in chosen];known=[r for r in sel if r['kind']>=0]
            if mode=='boundary':assert not any(r['boundary_refused'] for r in sel)
            summaries[arm+'_'+mode]=dict(selected=84,known=len(known),unknown=84-len(known),useful=sum(bool(r.get('useful')) for r in known),
                known_nonuseful=sum(not bool(r.get('useful')) for r in known),
                strictly_worse_center=sum(r['kind']==1 and r['err']>r['err_pred'] for r in sel),absent=sum(r['kind']==0 for r in sel))
            per[arm+'_'+mode]=per_video_useful([dict(r,useful=bool(r.get('useful')) and key(r) in chosen) for r in rows])
        write(RUN/arm/'boundary_rows.json',boundary)
        chosen=selected_keys(raw,.01);rejected=[r for r in boundary if key(r) in chosen and r['boundary_refused']]
        summaries[arm+'_original_selected_boundary']=dict(rejected=len(rejected),useful=sum(r['kind']>=0 and bool(r.get('useful')) for r in rejected),
            unknown=sum(r['kind']<0 for r in rejected),absent=sum(r['kind']==0 for r in rejected),strictly_worse_center=sum(r['kind']==1 and r['err']>r['err_pred'] for r in rejected))
        del maps
    comparisons={}
    for arm in ('C','D','R1'):
        for mode in ('raw','geometry6','map14'):comparisons[arm+'_boundary-'+arm+'_'+mode]=paired(per[arm+'_boundary'],per[arm+'_'+mode])
    for arm in ('C','D'):comparisons[arm+'_boundary-R1_boundary']=paired(per[arm+'_boundary'],per['R1_boundary'])
    dg={v:per['D_boundary'][v]-per['D_raw'][v] for v in per['D_raw']};rg={v:per['R1_boundary'][v]-per['R1_raw'][v] for v in per['R1_raw']}
    comparisons['D_gain-R1_gain']=paired(dg,rg)
    criteria=dict(reproduction=True,selection=all(comparisons['D_boundary-D_'+m]['ci95'][0]>0 for m in ('raw','geometry6','map14')),fly_control=comparisons['D_boundary-R1_boundary']['ci95'][0]>0)
    verify();write(RUN/'reproduction.json',reproduction);write(RUN/'cache_manifest.json',manifest)
    write(RUN/'report.json',dict(primary_pass=all(criteria.values()),criteria=criteria,paired=comparisons,summary=summaries,per_video=per,device='cpu',hota_measured=False,dev_evaluated=False,
        limitations='Exploratory repeated train26,post-hoc mechanism motivated by oneR1devcase. Equal84selection does not prove deployable calibrated refusal. No network/core/anatomy or HOTA improvement claim; old90%/50criterion unchanged.'))
    write(RUN/'status.json',dict(state='completed',stage='Полный итог train26 сохранён',done=3,total=3,updated=now().isoformat()))
    print('Boundary test complete',criteria,flush=True)


if __name__=='__main__':
    if '--prepare' in sys.argv:prepare()
    elif '--worker' in sys.argv:
        try:worker()
        except Exception as e:write(RUN/'status.json',dict(state='failed',stage=str(e),updated=now().isoformat()));raise
    else:raise RuntimeError('Explicit prepare/worker required')
