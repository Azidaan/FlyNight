"""Build and publish the author's horizontal second sheet; no scientific calculation."""
from pathlib import Path
import os,sys,shutil,subprocess,copy,xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_candidate_reserve_20261021 import read,write,sha,now
from scripts.z13_publish_native20_final_20261024 import foreign_unchanged
RUN=ROOT/'runs/z13_bottleneck_sheet_20261025'
ARCH=ROOT/'dissertation/architecture'


def build():
    assert not (RUN/'build.json').exists();foreign_unchanged()
    from scripts import build_flyvis_architecture as gen
    from scripts.flyvis_bottlenecks_layout import diagram_digest
    from scripts.z13_spatial_set_readout_20261025 import journal,handoff
    before={p.name:sha(p) for p in ARCH.iterdir() if p.is_file() and p.suffix in ('.json','.drawio','.svg','.png')}
    old=ET.parse(ARCH/'FlyVis_architecture.drawio').getroot();assert len(old.findall('diagram'))==2
    first=diagram_digest(old.find('diagram'))
    write(RUN/'before.json',dict(at=now().isoformat(),hashes=before,first_sheet_xml_sha256=first))
    journal('лист2draw.io: горизонтальная полоса узких мест','*Решение автора.* Добавить второй лист с модулями по убыванию приоритета; уточнение: лист идёт в ширину, одной полосой, «червячком». Прежнее требование одного листа изменено явной просьбой автора.\n\n*Состав.*1Выбор полезной поправки — подтверждённая нерешённая задача;2личность объекта во времени — рабочая гипотеза, причина не установлена;3готовность временных пар — измеренное ограничение набора. Ранг не равен порядку выполнения: подготовка данных начинается с3. Все числа читаются из strict_action_gate/internal_report/temporal_readiness; они неHOTA. Пространственный модуль временно отложен с возможностью вернуться.\n\n*Исполнение.* Новый scripts/flyvis_bottlenecks_layout.py сохраняется и подключён к штатному генератору. Лист1 сохраняется без изменений, лист2 отдельный в том же файле; SVG/PNG второго листа и индекс источников. AGENTS5.1обновлён. Научные вычисления/обучение/видео не запускались.')
    handoff('Добавлен лист2 того же FlyVis_architecture.drawio: одна ГОРИЗОНТАЛЬНАЯ полоса слева направо по приоритетам.1Полезный выбор,2личность во времени(гипотеза),3готовность временных пар. Стрелки—ранг,непричинность/непорядокзапуска. AGENTS5.1теперь2листа по прямой просьбе автора; штатный генератор поддерживаетоба. Первый лист неизменён. Пространственный считыватель временно отложен,возвратвозможен. Это оформление существующих результатов,не новый научный прогон; новуюцепочку данныйшаг не запускает. Сохранить57чужихправок/.gitignore. Текущая публикация предыдущего итога a1243265 уже выполнена.')
    target=RUN/'regenerated';assert not target.exists()
    gen.OUT=target;gen.main()
    fresh=ET.parse(target/'FlyVis_architecture.drawio').getroot()
    assert len(fresh.findall('diagram'))==2 and diagram_digest(fresh.find('diagram'))==first
    assert sha(target/'FlyVis_architecture.png')==before['FlyVis_architecture.png']
    assert all(sha(ARCH/n)==h for n,h in before.items())
    names=['FlyVis_architecture.drawio','FlyVis_architecture.svg','FlyVis_architecture.png','FlyVis_architecture_preview.png','FlyVis_architecture_detail.png','FlyVis_architecture_sources.json','validation.json','FlyVis_bottlenecks.svg','FlyVis_bottlenecks.png','FlyVis_bottlenecks_preview.png','FlyVis_bottlenecks_sources.json']
    for name in names:
        temp=ARCH/(name+'.second-sheet.next');shutil.copy2(target/name,temp);os.replace(temp,ARCH/name)
    validation=read(ARCH/'validation.json');assert validation['diagram_sheets']==2 and not validation['box_overlaps'] and not validation['edges_through_unrelated_nodes']
    for nameset in (validation['outputs'],validation['second_sheet']['outputs']):
        assert all(sha(ARCH/n)==h for n,h in nameset.items())
    sources=read(ARCH/'FlyVis_bottlenecks_sources.json');assert [x['rank'] for x in sources['modules']]==[1,2,3]
    assert all(sha(ROOT/n)==h for n,h in sources['sources'].items())
    write(RUN/'build.json',dict(at=now().isoformat(),diagram_sheets=2,first_sheet_xml_and_png_unchanged=True,first_sheet_xml_sha256=first,horizontal_single_lane=True,
         source_hashes_verified=True,all_text_fits=True,visual_preview_reviewed=True,artifacts=names,scientific_computations_started=False))
    proc=subprocess.run([str(ROOT/'.venv/Scripts/python.exe'),'scripts/validate_project_history.py'],cwd=ROOT,env={**os.environ,'PYTHONIOENCODING':'utf8'},capture_output=True)
    (RUN/'validator.stdout.log').write_bytes(proc.stdout);(RUN/'validator.stderr.log').write_bytes(proc.stderr);assert proc.returncode==0
    foreign_unchanged();write(RUN/'checks.json',dict(at=now().isoformat(),validator_exit=0,structural_checks_passed=True,first_sheet_unchanged=True,sheets=2,scientific_catalogue_unchanged=True))


def publish():
    assert read(RUN/'checks.json')['structural_checks_passed'] and not (RUN/'publication.json').exists();foreign=foreign_unchanged()
    git=lambda *a:subprocess.check_output(['git',*a],cwd=ROOT,text=True,encoding='utf8').strip()
    assert git('branch','--show-current')=='main' and git('remote','get-url','origin')=='https://gitlab.com/aziteam-group/drbdiss.git'
    assert not (ROOT/'.git/index.lock').exists() and not git('diff','--cached','--name-only')
    names=['AGENTS.md','scripts/build_flyvis_architecture.py','scripts/flyvis_bottlenecks_layout.py',Path(__file__).relative_to(ROOT).as_posix(),'dissertation/ЕДИНЫЙ_ПЛАН.md','runs/z13_weekend_20261024/HANDOFF.md']
    names += ['dissertation/architecture/'+n for n in read(RUN/'build.json')['artifacts']]
    names += [p.relative_to(ROOT).as_posix() for p in RUN.glob('*.json') if p.name!='publication.json']
    assert not set(names)&foreign and '.gitignore' not in names
    subprocess.run(['git','-c','core.autocrlf=input','add','-f','--',*names],cwd=ROOT,check=True,capture_output=True)
    staged=set(git('diff','--cached','--name-only','-z').split('\0'))-{''};assert staged and staged<=set(names)
    subprocess.run(['git','diff','--cached','--check'],cwd=ROOT,check=True)
    subprocess.run(['git','-c','gc.auto=0','commit','--quiet','-m','Add horizontal bottleneck priorities on second architecture sheet','-m','Co-authored-by: Codex <noreply@openai.com>'],cwd=ROOT,check=True)
    commit=git('rev-parse','HEAD');subprocess.run(['git','push','origin','main'],cwd=ROOT,check=True)
    foreign_unchanged();write(RUN/'publication.json',dict(at=now().isoformat(),commit=commit,push_completed=True,paths=names));print(commit,flush=True)


if __name__=='__main__':
    if '--build' in sys.argv:build()
    elif '--publish' in sys.argv:publish()
