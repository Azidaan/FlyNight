"""Separate scientific confidence replacement, using the fixed previous assessment."""
from pathlib import Path
from datetime import datetime
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_center_hota_20261021 import read,write,sha,now
RUN=ROOT/'runs/z13_absolute_confidence_20261022';BASE=ROOT/'runs/z13_boundary_readout_20261022';IDENT='z13_absolute_confidence_v1'


def journal(title,text):
    path=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';content=path.read_text(encoding='utf-8');anchor='## Журнал изменений\n';assert content.count(anchor)==1
    path.write_text(content.replace(anchor,anchor+f'\n**{now():%d.%m.%Y, %H:%M} MSK** — {title} `{IDENT}`\n\n'+text+'\n',1),encoding='utf-8')


def prepare():
    if RUN.exists():raise RuntimeError('Existing run')
    old=read(BASE/'protocol.json');assert not read(BASE/'report.json')['primary_pass']
    files=['tracking/z13_absolute_confidence.py','tests/test_z13_absolute_confidence.py','scripts/z13_absolute_confidence_20261022.py',
           'runs/z13_boundary_readout_20261022/report.json','scripts/z13_boundary_readout_20261022.py']
    p=dict(old);p.update(id=IDENT,created=now().isoformat(),mode='absolute',
        action='Replace query-normalized peak z=(max-mean)/std by raw max of the same observed-center ridge response. Center/model/lambda/history unchanged; no boundary veto.',
        method='One scalar confidence replacement only, no fitted coefficients or grids. The same84responses per arm are ranked by max(raw response). Reference max-z is reconstructed unchanged, then compare original,geometry6,map14 and R1 under identical1%quota.',
        counterargument='Raw ridge amplitudes can vary across target models and scene contrast; removing normalization may worsen ranking. Score is not a calibrated probability. Previous geometry6/map14 controls must be beaten, not only the weak max-z baseline.',
        alternatives_ranked=['Raw response amplitude: current isolated CPU check motivated by absent-target failures of max-z.','Template-referenced confidence: future, needs observed self-map cache.','Probabilistic or hard-negative target fitting: separate objective change, not performed here.'],
        decision='Advance only if all equal-quota D criteria pass. If not, stop scalar confidence replacement; do not keep tuning this selector on train26. Retain original centered readout and move to a separately specified model-fitting change. No dev tuning or HOTA rescue.',
        sha256={**old['sha256'],**{n:sha(ROOT/n) for n in files}})
    p['primary']=[old['primary'][0].replace('scores','original z-scores'),old['primary'][1].replace('boundary','absolute'),old['primary'][2].replace('boundary','absolute')]
    p['data']=old['data'];p['references']=[dict(url='https://github.com/visionml/pytracking/blob/master/pytracking/tracker/dimp/dimp.py',section='localize_advanced: max_score1 and not_found/uncertain thresholds',supports='Official DiMP compares raw learned classifier responses to thresholds; ours is a ridge map, not a DiMP reproduction or transferred threshold.'),old['references'][1]]
    RUN.mkdir();write(RUN/'protocol.json',p);write(RUN/'status.json',dict(state='ready',stage='Исходная высота отклика вместо max-z',done=0,total=3,updated=now().isoformat()))
    cat=read(ROOT/'experiments/project_history.json');assert not any(c['id']==IDENT for c in cat['experiments'])
    cat['experiments'].append(dict(id=IDENT,title='З13: абсолютная высота отклика вместо нормировки текущей картой',area='flyvis',state='ready',started='2026-10-22',done='',tags=['fly-problem-P8'],
        question='Улучшит ли исходная высота отклика выбор полезных центров D сверх max-z и прежних селекторов?',
        hypothesis='Деление на разброс текущей карты может усиливать пики при отсутствии цели; исходный масштаб отклика может быть полезен, но это не установлено.',
        expected='Фиксированная замена одного балла, прежние центры/ядро и26train-видео; одинаковый отбор84,CPUдо600с.',
        acceptance=[dict(criterion=c,met=None) for c in p['primary']],summary='До исходов зарегистрирована одна замена балла.',obtained='',interpretation='',decision='',
        cannot_claim='Не калиброванная вероятность, не HOTA и не преимущество анатомии/обучения ядра. Повторно используемые train26; один R1dev-пример мотивирует вопрос, но не задаёт параметры. Критерий90%/50 сохраняется.',
        metrics=[],commits=[],files=files[:3],evidence=[dict(kind='file',ref='runs/z13_absolute_confidence_20261022/protocol.json',claim='Одно изменение и прежние контроли до расчёта'),dict(kind='file',ref='runs/z13_boundary_readout_20261022/report.json',claim='Предыдущий запрет внешнего кольца не улучшил D')]))
    write(ROOT/'experiments/project_history.json',cat)
    journal('постановка абсолютной высоты отклика '+IDENT,'*Постановка.* '+p['action']+'\n\n'+p['method']+'\n\nКритерии до исходов: '+str(p['primary'])+'\n\nКонтраргумент: '+p['counterargument']+'\n\nПо первичному итогу предыдущего опыта D max-z и отказ от края выбрали2полезных/84;62известных отрицательных случая связаны с отсутствием цели,20неизвестны. Источник: runs/z13_boundary_readout_20261022/report.json → summary.D_raw/D_boundary. Ухудшение координат при присутствующей цели в этих выбранных строках не наблюдалось. Это ограничение отбора, не доказательство причины всей HOTA. Опора — официальный DiMP localize_advanced использует абсолютный max_score1; его модель и пороги не переносятся. CPUдо600с, без новых извлечений или обучения. Каталог новой карточки '+IDENT+' содержит самостоятельные критерии.')


def worker():
    from tracking.z13_readout import HexGeometry,to_frame
    from tracking.z13_absolute_confidence import read_response
    from tracking.z13_ridge_localizer import map_center
    from scripts.z13_query_memory_20261021 import load_maps
    from scripts.z13_continue import paired,per_video_useful
    from scripts.z13_score_swap import key,selected_keys
    started=time.monotonic();p=read(RUN/'protocol.json');source=ROOT/'runs/z13_query_memory_20261021'
    def check():
        if time.monotonic()-started>p['budget_seconds']-10 or (RUN/'stop.request').exists() or now()>=datetime.fromisoformat(p['save_at']):raise TimeoutError('CPU budget/author window')
    def verify():
        for name,digest in p['sha256'].items():check();assert sha(ROOT/name)==digest,name
    verify();g=read(ROOT/'runs/z13_local_peak_20261021/geometry.json');geom=HexGeometry(np.asarray(g['points'],np.float32),g['eye_size'])
    summaries={};per={};reproduction={};manifest={}
    for ai,arm in enumerate(('C','D','R1')):
        write(RUN/'status.json',dict(state='running',stage='Исходная высота отклика, равный отбор84',arm=arm,done=ai,total=3,updated=now().isoformat()))
        maps,hashes=load_maps(source/arm/'evaluate');manifest.update(hashes)
        labels=read(source/arm/'evaluate/labels.json');raw=read(source/arm/'last_rows.json');by={r['label']:r for r in raw};assert len(by)==len(labels)==len(maps)==8326
        absolute=[];dc=ds=0.
        for label in labels:
            check();score=maps[label['label']][-1];center,confidence,allowed=read_response(score,geom)
            c=np.asarray(to_frame(center,label['crop'],g['eye_size']));r=by[label['label']]
            assert r['video']==label['video'] and r['kind']==label['kind'] and allowed
            dc=max(dc,float(np.max(np.abs(c-r['center']))));ds=max(ds,abs(map_center(score,geom)[1]-r['p_present']))
            absolute.append(dict(r,p_present=confidence))
        assert dc<1e-6 and ds<1e-9
        reproduction[arm]=dict(rows=len(raw),max_center_px=dc,max_original_z_score=ds,confidence='raw maximum replaces z score; no veto')
        geo=read(ROOT/f'runs/z13_parallel_next_20261018/utility/{arm}_rows.json')['geometry']
        oldscores={(r['video'],str(r['label'])):r['score'] for r in read(ROOT/f'runs/z13_map_utility_20261018/{arm}/predicted_scores.json')}
        full=[dict(r,p_present=oldscores[r['video'],str(r['label'])]) for r in raw]
        variants=dict(raw=raw,absolute=absolute,geometry6=geo,map14=full)
        expected={key(r) for r in raw};assert all({key(r) for r in rows}==expected for rows in variants.values())
        for rows in variants.values():
            assert all(np.max(np.abs(np.asarray(r['center'])-by[r['label']]['center']))<.5 and bool(r.get('useful'))==bool(by[r['label']].get('useful')) for r in rows)
        for mode,rows in variants.items():
            chosen=selected_keys(rows,.01);assert len(chosen)==84
            selected=[r for r in rows if key(r) in chosen];known=[r for r in selected if r['kind']>=0]
            summaries[arm+'_'+mode]=dict(selected=84,known=len(known),unknown=84-len(known),useful=sum(bool(r.get('useful')) for r in known),known_nonuseful=sum(not bool(r.get('useful')) for r in known),strictly_worse_center=sum(r['kind']==1 and r['err']>r['err_pred'] for r in selected),absent=sum(r['kind']==0 for r in selected))
            per[arm+'_'+mode]=per_video_useful([dict(r,useful=bool(r.get('useful')) and key(r) in chosen) for r in rows])
        write(RUN/arm/'absolute_rows.json',absolute);del maps
    comparisons={}
    for arm in ('C','D','R1'):
        for mode in ('raw','geometry6','map14'):comparisons[arm+'_absolute-'+arm+'_'+mode]=paired(per[arm+'_absolute'],per[arm+'_'+mode])
    for arm in ('C','D'):comparisons[arm+'_absolute-R1_absolute']=paired(per[arm+'_absolute'],per['R1_absolute'])
    dg={v:per['D_absolute'][v]-per['D_raw'][v] for v in per['D_raw']};rg={v:per['R1_absolute'][v]-per['R1_raw'][v] for v in per['R1_raw']}
    comparisons['D_gain-R1_gain']=paired(dg,rg)
    criteria=dict(reproduction=True,selection=all(comparisons['D_absolute-D_'+m]['ci95'][0]>0 for m in ('raw','geometry6','map14')),fly_control=comparisons['D_absolute-R1_absolute']['ci95'][0]>0)
    verify();write(RUN/'reproduction.json',reproduction);write(RUN/'cache_manifest.json',manifest)
    write(RUN/'report.json',dict(primary_pass=all(criteria.values()),criteria=criteria,paired=comparisons,summary=summaries,per_video=per,device='cpu',hota_measured=False,dev_evaluated=False,limitations='Exploratory repeated train26. Raw ridge magnitude is not calibrated probability and may depend on target/contrast. Equal84selection is a selector diagnostic, not HOTA or core/anatomy improvement; old90%/50criterion unchanged.'))
    write(RUN/'status.json',dict(state='completed',stage='Полный итог train26 сохранён',done=3,total=3,updated=now().isoformat()))
    print('Absolute confidence test complete',criteria,flush=True)


if __name__=='__main__':
    if '--prepare' in sys.argv:prepare()
    elif '--worker' in sys.argv:
        try:worker()
        except Exception as e:write(RUN/'status.json',dict(state='failed',stage=str(e),updated=now().isoformat()));raise
    else:raise RuntimeError('Explicit prepare/worker required')
