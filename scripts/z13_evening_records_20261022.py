"""Review receipts for our evening records; never include pre-existing graph edits."""
from pathlib import Path
import sys
import subprocess
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_center_hota_20261021 import read,write,sha,now
RUN=ROOT/'runs/z13_absolute_confidence_20261022'


def own_graph():
    foreign={k.replace('\\','/') for k in read(ROOT/'runs/z13_boundary_readout_20261022/vault_finish.json')['protected_existing_edits']}
    names=set()
    for args in (['diff','--name-only','-z','HEAD','--','ObsidianFlyVis'],['ls-files','--others','--exclude-standard','-z','--','ObsidianFlyVis']):
        names.update(n for n in subprocess.check_output(['git','--no-optional-locks',*args],cwd=ROOT).decode('utf-8').split('\0') if n)
    return sorted(names-foreign)


def receipt():
    write(RUN/'owned_graph_receipt.json',{n:sha(ROOT/n) for n in own_graph() if (ROOT/n).is_file() and Path(n).suffix=='.md'})


def architecture():
    from PIL import Image
    nodes=read(ROOT/'dissertation/architecture/FlyVis_architecture_sources.json')['nodes'];node=next(n for n in nodes if n['id']=='v_z13_absolute_confidence_v1')
    image=Image.open(ROOT/'dissertation/architecture/FlyVis_architecture.png')
    image.crop((node['x']-70,node['y']-30,node['x']+node['w']+50,node['y']+node['h']+40)).save(RUN/'architecture_detail.png')
    plan=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';content=plan.read_text(encoding='utf-8');anchor='## Журнал изменений\n';assert content.count(anchor)==1
    event=f'\n**{now():%d.%m.%Y, %H:%M} MSK** — архитектурная веха: испытанные замены уверенности\n\nВ единственный лист добавлена ветвь двух отдельных проверок: отказ от края и исходная высота пика. Обе не приняты; зелёного улучшения HOTA нет. Контроли,84ответа,26train и источники указаны. Сохранена прежняя схема FlyVis_architecture_427f983.drawio. Проверка:85узлов,75стрелок,1лист, перекрытий/пересечений чужих блоков нет; текст помещается. Источник: dissertation/architecture/validation.json.\n'
    assert 'архитектурная веха: испытанные замены уверенности' not in content
    plan.write_text(content.replace(anchor,anchor+event,1),encoding='utf-8')


def manifest():
    files=['experiments/project_history.json','dissertation/ЕДИНЫЙ_ПЛАН.md',
        'tracking/z13_boundary_readout.py','tests/test_z13_boundary_readout.py','scripts/z13_boundary_readout_20261022.py',
        'scripts/z13_bounded_stage_20261022.py','scripts/z13_stage_progress_20261022.ps1','scripts/z13_record_cpu_readout_20261022.py',
        'tracking/z13_absolute_confidence.py','tests/test_z13_absolute_confidence.py','scripts/z13_absolute_confidence_20261022.py',
        'tracking/z13_balanced_ridge.py','tests/test_z13_balanced_ridge.py','scripts/z13_balanced_ridge_20261022.py',
        'scripts/z13_cpu_isolated_recovery_20261022.py','scripts/z13_bounded_recovery_controller_20261022.py',
        'scripts/z13_prepare_balanced_recovery_20261022.py','scripts/z13_record_balanced_ridge_20261022.py',
        'scripts/z13_balanced_progress_20261022.ps1',
        'scripts/build_flyvis_architecture.py','scripts/flyvis_architecture_layout.py','scripts/z13_evening_records_20261022.py']
    files.extend(own_graph())
    for run in ('z13_boundary_readout_20261022','z13_absolute_confidence_20261022'):
        files.extend(p.relative_to(ROOT).as_posix() for p in (ROOT/'runs'/run).rglob('*.json') if 'owned_backup' not in p.as_posix() and p.name!='publication_manifest.json')
    for name in ('protocol.json','launch.json','technical.json'):
        path=ROOT/'runs/z13_balanced_ridge_20261022'/name
        if path.exists():files.append(path.relative_to(ROOT).as_posix())
    files.extend(path.relative_to(ROOT).as_posix() for path in (ROOT/'runs/z13_balanced_ridge_20261022/recovery_cpu').glob('*.json'))
    files.extend(path.relative_to(ROOT).as_posix() for path in (ROOT/'runs/z13_balanced_ridge_20261022').glob('*.json') if path.name not in ('status.json','publication_manifest.json','watch_launch.json','publish_result.json'))
    for arm in ('D','C','R1'):
        files.extend(path.relative_to(ROOT).as_posix() for path in (ROOT/'runs/z13_balanced_ridge_20261022'/arm).glob('*.json'))
    for name in ('FlyVis_architecture.drawio','FlyVis_architecture.svg','FlyVis_architecture.png','FlyVis_architecture_preview.png','FlyVis_architecture_detail.png','FlyVis_architecture_sources.json','FlyVis_architecture_427f983.drawio','validation.json'):
        files.append('dissertation/architecture/'+name)
    write(RUN/'publication_manifest.json',{name:sha(ROOT/name) for name in sorted(set(files)) if (ROOT/name).is_file()})
    print('Explicit publication files',len(read(RUN/'publication_manifest.json')))


if __name__=='__main__':
    if '--receipt' in sys.argv:receipt()
    elif '--architecture' in sys.argv:architecture()
    elif '--manifest' in sys.argv:manifest()
    else:raise ValueError('Explicit mode required')
