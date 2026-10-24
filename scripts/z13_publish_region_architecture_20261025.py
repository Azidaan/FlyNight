"""Publish the reviewed single-sheet architecture milestone without rerunning science."""
from pathlib import Path
import os
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.z13_region_readout_20261025 import W,read,write,sha,now,journal


def main():
    from scripts.z13_publish_native20_final_20261024 import foreign_unchanged,graph_rows
    foreign=foreign_unchanged();prefix='region_architecture'
    receipt=W/(prefix+'_publication.json');assert not receipt.exists()
    run=ROOT/'runs/z13_region_readout_20261025/recovery1'
    result=read(run/'report.json');assert not result['primary_pass']
    finish=read(run/'finish_recorded.json');assert sha(run/'report.json')==finish['report_sha256']
    val=read(ROOT/'dissertation/architecture/validation.json')
    assert val['diagram_sheets']==1 and not val['box_overlaps'] and not val['edges_through_unrelated_nodes'] and val['all_text_fits']
    for name,digest in val['outputs'].items():assert sha(ROOT/'dissertation/architecture'/name)==digest
    journal('архитектурная веха: региональное считывание без подтверждённого улучшения',
        '*Сделано.* На единственном листе dissertation/architecture/FlyVis_architecture.drawio добавлена жёлтая ветвь z13_region_readout_v1: чтение отклика по реальным областям и передача оценок в сопоставление вместо переноса центра. '
        'Итог связан с runs/z13_region_readout_20261025/recovery1/report.json:12 полезных/14 вредных, средний повидео сдвиг −0,10 п.п., ДИ[−0,30;+0,08]. Преимущество над C/R1/геометрией не подтверждено; это не HOTA.\n\n'
        '*Проверено.* validation.json: один лист,96 узлов,86 связей, перекрытий блоков и проходов стрелок через чужие блоки нет, текст помещается. Просмотрены общий вид и читаемый фрагмент новой ветви. Источники и прежняя геометрия альтернатив сохранены; повторных научных расчётов нет. '
        'Каталог и первичные результаты уже проверены70 тестами и опубликованы ff5a807; эта правка оформляет завершённую архитектурную веху.')
    header='# Актуально — '+now().strftime('%d.%m.%Y %H:%M')+' MSK\n\n'
    header+='Региональный декодер завершён и опубликован ff5a807, первичные строки/интервалы/46 исходных выходов проверены;70 тестов пройдены. Критерии не пройдены, версию не переносить в HOTA. Однолистовая архитектура обновлена по этой вехе, общий вид и новая жёлтая ветвь просмотрены, validation.json зелёный. Публикация оформления: region_architecture_publication.json. Научные расчёты этой версии окончены; нового готового исполнителя сейчас нет. Следующий узел требует собственного обоснования и дешёвого критерия, а не повторения неудачного alpha или изменения порогов. Авторизация до26.10 07:40МСК сохраняется.57 чужих графправок и .gitignore сохранены.\n\nНиже прежние передачи сохранены.\n\n'
    handoff=W/'HANDOFF.md';handoff.write_text(header+handoff.read_text(encoding='utf-8'),encoding='utf-8')
    env={**os.environ,'PYTHONIOENCODING':'utf-8','CUDA_VISIBLE_DEVICES':''}
    checked=subprocess.run([str(ROOT/'.venv/Scripts/python.exe'),'scripts/validate_project_history.py'],cwd=ROOT,env=env,capture_output=True)
    (W/(prefix+'_catalog.log')).write_bytes(checked.stdout);assert checked.returncode==0
    owned=W/(prefix+'_owned_graph.json')
    write(owned,{n:sha(ROOT/n) for _,n in graph_rows() if n not in foreign and (ROOT/n).is_file() and (ROOT/n).suffix=='.md'})
    subprocess.run([str(ROOT/'.venv/Scripts/python.exe'),'scripts/z13_build_vault_preserving_edits.py','--report',str(W/(prefix+'_vault.json')),'--refresh-owned',str(owned)],cwd=ROOT,env=env,check=True,capture_output=True)
    foreign_unchanged()
    names=[Path(__file__).relative_to(ROOT).as_posix(),'scripts/build_flyvis_architecture.py','dissertation/ЕДИНЫЙ_ПЛАН.md','runs/z13_weekend_20261024/HANDOFF.md']
    for suffix in ('.drawio','.svg','.png','_preview.png','_detail.png','_sources.json'):
        names.append('dissertation/architecture/FlyVis_architecture'+suffix)
    names+=['dissertation/architecture/validation.json',owned.relative_to(ROOT).as_posix(),(W/(prefix+'_vault.json')).relative_to(ROOT).as_posix()]
    for state,name in graph_rows():
        if name not in foreign and ((ROOT/name).is_file() or 'D' in state):names.append(name)
    names=sorted(set(names));assert not set(names)&foreign and '.gitignore' not in names
    assert not (ROOT/'.git/index.lock').exists()
    assert subprocess.check_output(['git','branch','--show-current'],cwd=ROOT).decode().strip()=='main'
    assert subprocess.check_output(['git','remote','get-url','origin'],cwd=ROOT).decode().strip()=='https://gitlab.com/aziteam-group/drbdiss.git'
    staged=set(subprocess.check_output(['git','diff','--cached','--name-only','-z'],cwd=ROOT).decode().split(chr(0)))-{''}
    assert staged<=set(names)
    for left in range(0,len(names),25):subprocess.run(['git','add','-f','--',*names[left:left+25]],cwd=ROOT,check=True,capture_output=True)
    subprocess.run(['git','diff','--cached','--check'],cwd=ROOT,check=True,capture_output=True)
    subprocess.run(['git','commit','--quiet','-m','Document unaccepted region readout on the single-sheet FlyVis architecture','-m','Co-authored-by: Codex <noreply@openai.com>'],cwd=ROOT,check=True,capture_output=True)
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip()
    subprocess.run(['git','push','origin','main'],cwd=ROOT,check=True)
    foreign_unchanged();assert sha(run/'report.json')==finish['report_sha256']
    write(receipt,dict(at=now().isoformat(),commit=head,push_completed=True,paths=names,
        single_sheet_and_visual_review=True,scientific_sources_unchanged=True,prior_science_tests=70))
    print(head)


if __name__=='__main__':main()
