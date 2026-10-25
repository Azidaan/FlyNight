"""Publish the fixed context protocol and active native queue launch."""
from pathlib import Path
import sys
import subprocess
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_candidate_reserve_20261021 import read,write,sha,now
from scripts.z13_publish_native20_final_20261024 import foreign_unchanged,graph_rows
RUN=ROOT/'runs/z13_region_context_20261025'


def main():
    assert not (RUN/'start_publication.json').exists()
    assert '61 passed' in (RUN/'prepare_tests.log').read_text(encoding='utf-8')
    assert 'OK:' in (RUN/'prepare_validation.log').read_text(encoding='utf-8')
    until=time.monotonic()+30
    while not (RUN/'cache/controller.json').exists():
        assert time.monotonic()<until,'Native cache controller did not appear';time.sleep(.5)
    p=read(RUN/'protocol.json');h=ROOT/'runs/z13_weekend_20261024/HANDOFF.md'
    h.write_text('# Актуально — '+now().strftime('%d.%m.%Y %H:%M')+' MSK\n\n'
       'ТЕКУЩИЙ z13_region_context_v1: runs/z13_region_context_20261025, scripts/z13_region_context_20261025.py --pipeline. '
       'Реальный pipeline: '+str(read(RUN/'pipeline.json'))+'; первый контроллер: '+str(read(RUN/'cache/controller.json'))+'. '
       'Запуск/консоль: launch.json. Локальная очередь автоматически cache130CPU1200с -> training24CPU1800с -> '
       'первичные хеши/строки/парныеCI/карточка/журнал/проверки/сохраняющий граф/scoped push. '
       'Один новый узел:6наблюдаемых признаков исходного назначения и конкуренции других треков. '
       'Равный blank-контроль обнуляет6входов при той же94входной архитектуре; D/C/R1/геометрия,трисидa, '
       'прежняя знаковая цель и калибровка fit20. Никаких GPU/новых весов ядра/dev/HOTA/heldout75/загрузок. '
       'Технический контроль130float32-exact базовых повторов и точных37,296UID/метаданных; старые признаки/метки не пересчитываются. '
       'Новое извлечение нужно только для ранее не сохранённой матрицы назначения. '
       'Общий save '+p['pipeline_save_at']+' / hard '+p['pipeline_hard_stop']+', меньшие бюджеты фаз не продлевать. '
       'Первый2замер/полный паритет обязательны; при провале не переходить к моделям. '
       'Читать status.json, pipeline_failure.json, cache/training controller_result.json и worker.stderr.log. '
       'При известной служебной ошибке — немедленная отдельная версия с прежним сроком и сохранением готового; не запускать дубль. '
       'После итогового gate: проверить confirmed_results/архитектурную веху; проваленный узел кHOTA не продвигать. '
       'Три предыдущие региональные версии отрицательны, их карточки и однолистовая архитектура опубликованы1414cfe3; '
       'положительное восстановление против неудачной балансировки не является пользой стенду. '
       'Авторизация до26.10 07:40МСК,57чужихграфправок и.gitignore сохранять.\n\nНиже прежние передачи сохранены.\n\n'+h.read_text(encoding='utf-8'),encoding='utf-8')
    write(RUN/'prepare_verified.json',dict(at=now().isoformat(),tests61_passed=True,validator_exit0=True,
        real94_input_forward_backward_verified=True,protocol_sha256=sha(RUN/'protocol.json'),tests_sha256=sha(RUN/'prepare_tests.log')))
    foreign=foreign_unchanged();receipt=RUN/'start_owned_graph.json'
    write(receipt,{n:sha(ROOT/n) for _,n in graph_rows() if (ROOT/n).is_file() and n not in foreign})
    subprocess.run([str(ROOT/'.venv/Scripts/python.exe'),'scripts/z13_build_vault_preserving_edits.py','--report',str(RUN/'vault_start.json'),'--refresh-owned',str(receipt)],cwd=ROOT,check=True,capture_output=True)
    names=[Path(__file__).relative_to(ROOT).as_posix(),'scripts/z13_region_context_20261025.py','scripts/z13_region_context_console.ps1',
       'tracking/z13_region_assignment_context.py','tests/test_z13_region_assignment_context.py','experiments/project_history.json',
       'dissertation/ЕДИНЫЙ_ПЛАН.md','runs/z13_weekend_20261024/HANDOFF.md']
    names += [(RUN/n).relative_to(ROOT).as_posix() for n in ('protocol.json','smoke.json','launch.json','pipeline.json','cache/protocol.json','cache/controller.json','prepare_verified.json','start_owned_graph.json','vault_start.json')]
    names += [n for _,n in graph_rows() if n not in foreign and (ROOT/n).is_file()]
    names=sorted(set(names));assert not set(names)&foreign and '.gitignore' not in names
    staged=set(subprocess.check_output(['git','diff','--cached','--name-only','-z'],cwd=ROOT).decode().split('\0'))-{''};assert staged<=set(names)
    assert not (ROOT/'.git/index.lock').exists()
    assert subprocess.check_output(['git','remote','get-url','origin'],cwd=ROOT).decode().strip()=='https://gitlab.com/aziteam-group/drbdiss.git'
    for i in range(0,len(names),25):subprocess.run(['git','add','-f','--',*names[i:i+25]],cwd=ROOT,check=True,capture_output=True)
    subprocess.run(['git','diff','--cached','--check'],cwd=ROOT,check=True)
    subprocess.run(['git','commit','--quiet','-m','Register matched observable assignment-context readout queue','-m','Co-authored-by: Codex <codex@openai.com>'],cwd=ROOT,check=True)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip();subprocess.run(['git','push','origin','main'],cwd=ROOT,check=True)
    foreign_unchanged();write(RUN/'start_publication.json',dict(at=now().isoformat(),commit=commit,push_completed=True,paths=names))


if __name__=='__main__':main()
