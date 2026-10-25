"""Save the reviewed single-sheet milestone after three completed readout trials."""
from pathlib import Path
import sys
import subprocess
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_candidate_reserve_20261021 import read,write,sha,now
from scripts.z13_publish_native20_final_20261024 import foreign_unchanged,graph_rows
OUT=ROOT/'runs/z13_region_architecture_20261025'


def main():
    assert not OUT.exists()
    assert read(ROOT/'runs/z13_region_value_20261025/publication.json')['push_completed']
    audit=read(ROOT/'dissertation/architecture/validation.json')
    assert audit['diagram_sheets']==1 and audit['nodes']==100 and not audit['box_overlaps'] and not audit['edges_through_unrelated_nodes'] and audit['all_text_fits']
    for name,digest in audit['outputs'].items():assert sha(ROOT/'dissertation/architecture'/name)==digest
    reports=['runs/z13_region_assignment_20261025/recovery2/training/report.json','runs/z13_region_balanced_20261025/training/report.json','runs/z13_region_value_20261025/training/report.json']
    assert all(read(ROOT/n)['complete'] and not read(ROOT/n)['primary_pass'] for n in reports)
    write(OUT/'review.json',dict(at=now().isoformat(),single_sheet=True,overview_and_readable_three_branch_detail_viewed=True,
        three_rejected_variants_yellow=True,scientific_reports={n:sha(ROOT/n) for n in reports},
        confirmed_results_review='No strong improvement against the existing readout/stand: all primary baseline intervals cross0 and the2pp gate fails. Signed versus rejected balanced loss is a positive but small recovery contrast; fully retained in primary report and chronology, not promoted as a best stand result.',
        partial_positive=read(ROOT/reports[-1])['paired']['D-value-D-balanced'],problem_P8='Remains partial; reliable intervention and stand benefit not established.'))
    from scripts import z13_localization_labels_20261023 as notes
    notes.IDENT='z13_region_value_v1'
    notes.journal('архитектурная веха трёх проверенных региональных замен',
       '*Получено.* На одном листе добавлены три жёлтые альтернативы: обучаемый выбор областей, равный вес запросов, знаковая полезность. '
       'Все основные критерии не пройдены. Источники: runs/z13_region_assignment_20261025/recovery2/training/report.json; '
       'runs/z13_region_balanced_20261025/training/report.json; runs/z13_region_value_20261025/training/report.json. '
       'Последняя D против исходных назначений +0,066 п.п. [−0,023; +0,195], не HOTA. '
       'Отдельно Dvalue−Dbalanced +0,257 п.п. [положительная нижняя граница0,000193; +0,639] — восстановление после '
       'неудачной балансировки, не подтверждённая польза против исходного стенда. '
       '*Проверено.* 100блоков/90стрелок, один лист, без перекрытий/пересечения чужих блоков, текст помещается; '
       'просмотрены общий вид и читаемый фрагмент трёх новых ветвей. Источник dissertation/architecture/validation.json '
       'и runs/z13_region_architecture_20261025/review.json. P8 остаётся частичной; новых сильных сравнений для блока11нет. '
       '*Решение.* Не продолжать перебор скалярных потерь; следующий проверяемый узел — отсутствующий наблюдаемый контекст назначения.')
    with (OUT/'validation.log').open('w',encoding='utf-8') as out:subprocess.run([str(ROOT/'.venv/Scripts/python.exe'),'scripts/validate_project_history.py'],cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,check=True)
    foreign=foreign_unchanged();receipt=OUT/'owned_graph_before.json';write(receipt,{n:sha(ROOT/n) for _,n in graph_rows() if (ROOT/n).is_file() and n not in foreign})
    subprocess.run([str(ROOT/'.venv/Scripts/python.exe'),'scripts/z13_build_vault_preserving_edits.py','--report',str(OUT/'vault.json'),'--refresh-owned',str(receipt)],cwd=ROOT,check=True,capture_output=True)
    names=[Path(__file__).relative_to(ROOT).as_posix(),'scripts/build_flyvis_architecture.py','dissertation/ЕДИНЫЙ_ПЛАН.md']
    names += [p.relative_to(ROOT).as_posix() for p in (ROOT/'dissertation/architecture').iterdir() if p.is_file()]
    names += [p.relative_to(ROOT).as_posix() for p in OUT.glob('*.json')]
    names += [n for _,n in graph_rows() if n not in foreign and (ROOT/n).is_file()]
    names=sorted(set(names));assert not set(names)&foreign and '.gitignore' not in names
    staged=set(subprocess.check_output(['git','diff','--cached','--name-only','-z'],cwd=ROOT).decode().split('\0'))-{''};assert staged<=set(names)
    assert not (ROOT/'.git/index.lock').exists()
    assert subprocess.check_output(['git','remote','get-url','origin'],cwd=ROOT).decode().strip()=='https://gitlab.com/aziteam-group/drbdiss.git'
    for i in range(0,len(names),25):subprocess.run(['git','add','-f','--',*names[i:i+25]],cwd=ROOT,check=True,capture_output=True)
    subprocess.run(['git','diff','--cached','--check'],cwd=ROOT,check=True)
    subprocess.run(['git','commit','--quiet','-m','Record three rejected regional readout replacements on one architecture sheet','-m','Co-authored-by: Codex <codex@openai.com>'],cwd=ROOT,check=True)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip();subprocess.run(['git','push','origin','main'],cwd=ROOT,check=True)
    foreign_unchanged();write(OUT/'publication.json',dict(at=now().isoformat(),commit=commit,push_completed=True,paths=names))


if __name__=='__main__':main()
