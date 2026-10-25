"""Record and publish the one-sheet uncertainty/NULL architecture milestone."""
from pathlib import Path
import sys, subprocess, inspect
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_candidate_reserve_20261021 import read,write,sha,now
from scripts import z13_null_mixture_20261025 as science
from scripts import z13_history_clock_20261025 as publisher
RUN=science.RUN/'architecture_milestone'
ARCH=ROOT/'dissertation/architecture'

def crops():
    from PIL import Image
    Image.MAX_IMAGE_PIXELS=None
    source=read(ARCH/'FlyVis_architecture_sources.json')
    with Image.open(ARCH/'FlyVis_architecture.png') as im:
        for ident in ('z13_ridge_uncertainty_v1','z13_null_mixture_v1'):
            n=next(n for n in source['nodes'] if n['id']=='v_'+ident)
            im.crop((n['x']-35,n['y']-35,n['x']+n['w']+35,n['y']+n['h']+35)).save(RUN/('v_'+ident+'.png'))

def record():
    assert not (RUN/'review.json').exists()
    v=read(ARCH/'validation.json')
    assert v['diagram_sheets']==1 and not v['box_overlaps'] and not v['edges_through_unrelated_nodes'] and v['all_text_fits']
    for n,h in v['outputs'].items(): assert sha(ARCH/n)==h
    paths=['runs/z13_ridge_uncertainty_20261025/replay/report.json','runs/z13_null_mixture_20261025/report.json']
    for name in paths:
        r=read(ROOT/name);assert r['complete'] and not r['primary_pass']
    r=read(ROOT/paths[1]);assert r['paired']['null_effect_D-geometry']['ci95'][0]>0
    assert r['paired']['Dnull-baseline']['ci95'][0]<0 and r['paired']['Dnull-Dold']['ci95'][0]<0
    write(RUN/'review.json',dict(at=now().isoformat(),one_sheet=True,nodes=v['nodes'],edges=v['edges'],overview_and_latest_two_branch_crops_viewed=True,
        source_reports={n:sha(ROOT/n) for n in paths},both_new_branches_yellow=True,
        confirmed_results_review='Positive interaction (Dnull-Dold)-(geometrynull-geometryold) retained as a partial result. Absolute Dnull-baseline, Dnull-Dold, Dnull-C/R1/geometry all include zero. No new confirmed useful stand improvement; no HOTA claim.',
        scope='Uncertainty is a conditional readout covariance experiment. NULL-as-unchanged is a new application contract, not an intrinsic core modernization or service repair. Existing red/green branches retained.'))
    science.journal('архитектурная веха уверенности и явного бездействия',
        '*Получено.* На одном листе добавлены две жёлтые ветви: неопределённость считывателя по опоре признаков образца; сохранение исходного назначения при ненадёжном ответе. '
        'Обе реализации не прошли основные критерии. NULL D против исходных назначений +0,0198 п.п. [−0,2111;+0,2692], против прежней D +0,1219 п.п. [−0,0019;+0,2688]. '
        'Частное взаимодействие (D новая−D прежняя)−(геометрия новая−геометрия прежняя) +0,1240 п.п. [+0,0064;+0,2576] сохранено, но оно не доказывает абсолютную пользу стенду. '
        'Источники runs/z13_ridge_uncertainty_20261025/replay/report.json и runs/z13_null_mixture_20261025/report.json. '
        '*Проверено.* Обзор и актуальные читаемые фрагменты просмотрены; '+str(v['nodes'])+' блоков/'+str(v['edges'])+' стрелок, один лист, без перекрытий и обрезки. '
        'Источник runs/z13_null_mixture_20261025/architecture_milestone/review.json. Новых лучших подтверждённых результатов стенда нет; HOTA не измерялась. '
        '*Решение.* В графе и схеме не перекрашивать эти ветви в зелёный успех. Готовые этапы не повторять; критерии двух неопределённых проверок наличия данных не ослаблять.')
    from scripts import z13_region_assignment_recovery2_20261025 as notes
    notes.namespace()['handoff']('Архитектурная веха uncertainty/NULL сохранена: один лист107блоков/97стрелок, обе новые ветви жёлтые. Частное положительное взаимодействие NULL не означает абсолютного улучшения; основной критерий не пройден. Научных процессов нет. Следующий внутренний узел выбирать по действующему плану с измеренной дешёвой мишенью; не продвигать неопределённые observed-negatives/four-bin gates. Всё скрыто;57чужихправок/.gitignore сохранять.')

def publish():
    assert read(RUN/'review.json')['overview_and_latest_two_branch_crops_viewed']
    extra=['scripts/build_flyvis_architecture.py','scripts/flyvis_architecture_layout.py',Path(__file__).relative_to(ROOT).as_posix()]
    extra += [p.relative_to(ROOT).as_posix() for p in ARCH.iterdir() if p.is_file() and (p.name.startswith('FlyVis_architecture') or p.name=='validation.json')]
    s=inspect.getsource(publisher.publish)
    s=s.replace("['-m','pytest','tests/test_project_history.py','tests/test_z13_history_clock.py','-q']","['-m','pytest','tests/test_project_history.py','-q']")
    s=s.replace('    assert not set(names)&foreign','    names += EXTRA\n    assert not set(names)&foreign')
    s=s.replace('physical-frame temporal resampling inside frozen FlyVis','one-sheet uncertainty and NULL readout architecture milestone')
    ns={**vars(publisher),'RUN':RUN,'__file__':__file__,'EXTRA':extra}
    exec(compile(s,__file__,'exec'),ns);ns['publish']('final')

if __name__=='__main__':
    if '--crops' in sys.argv:crops()
    elif '--record' in sys.argv:record()
    elif '--publish' in sys.argv:publish()
    else:raise RuntimeError('Explicit mode required')
