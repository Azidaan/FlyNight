"""Preserve the agent's failed layout, record the visually reviewed final milestone."""
from pathlib import Path
import sys
import subprocess
import shutil
import inspect
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_candidate_reserve_20261021 import read,write,sha,now
from scripts import z13_history_clock_20261025 as science
from scripts import z13_history_clock_recovery1_20261025 as recovery
OUT=science.RUN/'architecture_milestone'
ARCH=ROOT/'dissertation/architecture'


def repair_owned_build():
    # Before this build both architecture generators/exports were Git-clean.
    # The generator wrote its own exports before rejecting a crossing arrow.
    # Preserve that failed attempt; restore only the exact accepted diagram so
    # its existing receipt can authorize the next generation without bypassing it.
    folder=OUT/'failed_layout';assert not folder.exists()
    path=ARCH/'FlyVis_architecture.drawio';receipt=read(ARCH/'validation.json')
    old=subprocess.check_output(['git','show','HEAD:dissertation/architecture/FlyVis_architecture.drawio'],cwd=ROOT)
    import hashlib
    if hashlib.sha256(old).hexdigest()!=receipt['outputs'][path.name]:
        old=old.replace(b'\r\n',b'\n').replace(b'\n',b'\r\n')
    assert hashlib.sha256(old).hexdigest()==receipt['outputs'][path.name]
    text=path.read_text(encoding='utf-8')
    assert 'v_z13_history_clock_v1' in text and 'v_z13_region_semantic_v1' in text
    folder.mkdir(parents=True)
    for p in ARCH.iterdir():
        if p.is_file() and (p.name.startswith('FlyVis_architecture') or p.name=='validation.json'):shutil.copy2(p,folder/p.name)
    write(OUT/'layout_recovery.json',dict(at=now().isoformat(),old_accepted_diagram_sha256=hashlib.sha256(old).hexdigest(),failed_agent_diagram_sha256=sha(path),
        reason='Own generated temporal branch moved the future-memory box into the old wrap corridor; structural check rejected it before a new receipt. Failed exports preserved. Restored exact accepted diagram bytes only to satisfy the unchanged overwrite guard. Move lower row and wrap corridor; no scientific source changed.'))
    path.write_bytes(old)


def record():
    assert not (OUT/'review.json').exists()
    v=read(ARCH/'validation.json')
    assert v['diagram_sheets']==1 and not v['box_overlaps'] and not v['edges_through_unrelated_nodes'] and v['all_text_fits']
    for n,h in v['outputs'].items():assert sha(ARCH/n)==h
    reports=['runs/z13_region_semantic_20261025/training/report.json','runs/z13_history_clock_20261025/replay/report.json']
    assert all(read(ROOT/n)['complete'] and not read(ROOT/n)['primary_pass'] for n in reports)
    write(OUT/'review.json',dict(at=now().isoformat(),one_sheet=True,nodes=v['nodes'],edges=v['edges'],overview_and_readable_two_branch_crops_viewed=True,
        both_new_branches_yellow=True,source_reports={n:sha(ROOT/n) for n in reports},
        confirmed_results_review='Neither semantic-target nor actual-frame-clock version passes the assignment gate; no new best confirmed improvement. Clock D-old lower bound equals0, not positive. Existing useful-center results remain separate from HOTA.',
        problem_status='Reliable intervention remains unresolved. Temporal mismatch is observed; correcting it by this linear-interpolation implementation did not establish a useful effect.'))
    science.journal('архитектурная веха семантической цели и хода времени',
        '*Получено.* На единственном листе добавлены две жёлтые альтернативы: цель обучения правильной области вместо редкой полезной поправки; ход состояния по реальному интервалу кадров вместо одного шага на запись. '
        'Обе версии не прошли основные критерии. Семантическая D против исходных назначений −0,0054 п.п. [−0,0162;0]; '
        'временная D против прежней D +0,0143 п.п. [0;+0,0429], против исходных назначений −0,0878 п.п. [−0,2839;+0,0895]. '
        'Источники: runs/z13_region_semantic_20261025/training/report.json и runs/z13_history_clock_20261025/replay/report.json. '
        '*Проверено.* Обзор и читаемые фрагменты обеих ветвей просмотрены; '+str(v['nodes'])+' блоков/'+str(v['edges'])+' стрелок, один лист, нет перекрытий/проходов через посторонние блоки/обрезки текста. '
        'Первую собственную неудачную компоновку сохранили в architecture_milestone/failed_layout; коридор основного пути и нижний ряд опущены. '
        'Новых сильных подтверждённых результатов для блока11нет; полезные центры не HOTA. Источник runs/z13_history_clock_20261025/architecture_milestone/review.json. '
        '*Решение.* Не переносить эти реализации в HOTA; следующий внутренний узел требует измеренной мишени, равного контроля и собственного критерия до прогона.')


def publish():
    assert read(OUT/'review.json')['overview_and_readable_two_branch_crops_viewed']
    names=['scripts/build_flyvis_architecture.py','scripts/flyvis_architecture_layout.py',Path(__file__).relative_to(ROOT).as_posix(),
        'scripts/z13_history_clock_recovery1_20261025.py','scripts/z13_publish_history_clock_start_20261025.py']
    names += [p.relative_to(ROOT).as_posix() for p in ARCH.iterdir() if p.is_file() and (p.name.startswith('FlyVis_architecture') or p.name=='validation.json')]
    code=inspect.getsource(science.publish);old='    assert not set(names)&foreign';assert code.count(old)==1
    code=code.replace(old,'    names += MILESTONE_FILES\n'+old)
    ns=dict(vars(science));ns['MILESTONE_FILES']=names
    exec(compile(code,str(Path(__file__)),'exec'),ns);ns['publish']('final')


if __name__=='__main__':
    if '--repair-owned-build' in sys.argv:repair_owned_build()
    elif '--record' in sys.argv:record()
    elif '--publish' in sys.argv:publish()
    else:raise RuntimeError('Explicit mode required')
