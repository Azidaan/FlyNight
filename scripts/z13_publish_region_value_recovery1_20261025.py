"""Publish verified namespace recovery without changing scientific training."""
from pathlib import Path
import sys
import subprocess
import hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_candidate_reserve_20261021 import read,write,sha,now
from scripts import z13_region_value_recovery1_20261025 as recovery
from scripts.z13_publish_native20_final_20261024 import foreign_unchanged
RUN=recovery.CONTROL


def main():
    p=read(RUN/'protocol.json');assert read(RUN/'service_smoke.json')['predict_and_policy_and_gc_passed']
    assert hashlib.sha256(recovery.science.training_source().encode()).hexdigest()==p['executed_training_sha256']
    h=ROOT/'runs/z13_weekend_20261024/HANDOFF.md'
    h.write_text('# Актуально — '+now().strftime('%d.%m.%Y %H:%M')+' MSK\n\n'
        'ТЕКУЩИЙ z13_region_value_v1: научные результаты и status остаются в runs/z13_region_value_20261025; '
        'АКТУАЛЬНЫЕ controller/monitor/worker.logs находятся в recovery1. Старые parent controller_result exit1 '
        'и monitor_failure относятся к NameError отсутствующего predict после сохранения1150шагов, не к текущему состоянию. '
        'Новая служебная версия подключила прежние predict/policy_rows/gc/load_split без изменения SHA научной функции. '
        'Dseed0 сохранённые1150шагов не повторены; реально уже идут C и новые модели. '
        'Текущий controller '+str(read(RUN/'controller.json'))+'; консоль '+str(read(RUN/'console_launch.json'))+'. '
        'Исходные save '+p['save_at']+' / hard '+p['hard_stop']+' не продлены. '
        'Финализатор recovery1 перенаправляет только чтение квитанции controller_result, все научные файлы прежние. '
        'Он проверит12моделей/первичные строки/CI/каталог/граф и scoped push. '
        'При новом сбое немедленно исправлять проверенную служебную причину в том же окне, не повторять completed. '
        'После итога проверить отдельные подтверждённые сравнения и однолистовую архитектуру. '
        'Следующий узел при научном провале — отсутствующий наблюдаемый контекст назначения; не перебирать скалярные потери. '
        'Авторизация до26.10 07:40МСК,57чужихграфправок/.gitignore сохранять.\n\nНиже прежние передачи сохранены.\n\n'+h.read_text(encoding='utf-8'),encoding='utf-8')
    with (RUN/'service_validation.log').open('w',encoding='utf-8') as out:subprocess.run([str(ROOT/'.venv/Scripts/python.exe'),'scripts/validate_project_history.py'],cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,check=True)
    names=[Path(__file__).relative_to(ROOT).as_posix(),'scripts/z13_region_value_recovery1_20261025.py','scripts/z13_region_value_recovery1_console.ps1',
        'dissertation/ЕДИНЫЙ_ПЛАН.md','runs/z13_weekend_20261024/HANDOFF.md']
    names += [p.relative_to(ROOT).as_posix() for p in RUN.glob('*.json')]
    names += [(recovery.science.RUN/n).relative_to(ROOT).as_posix() for n in ('controller_result.json','monitor_failure.json')]
    names=sorted(set(names));foreign_unchanged()
    staged=set(subprocess.check_output(['git','diff','--cached','--name-only','-z'],cwd=ROOT).decode().split('\0'))-{''};assert staged<=set(names)
    assert not (ROOT/'.git/index.lock').exists()
    assert subprocess.check_output(['git','remote','get-url','origin'],cwd=ROOT).decode().strip()=='https://gitlab.com/aziteam-group/drbdiss.git'
    subprocess.run(['git','add','-f','--',*names],cwd=ROOT,check=True)
    subprocess.run(['git','diff','--cached','--check'],cwd=ROOT,check=True)
    subprocess.run(['git','commit','--quiet','-m','Resume signed-utility readout with original prediction helpers','-m','Co-authored-by: Codex <codex@openai.com>'],cwd=ROOT,check=True)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip();subprocess.run(['git','push','origin','main'],cwd=ROOT,check=True)
    foreign_unchanged();write(RUN/'start_publication.json',dict(at=now().isoformat(),commit=commit,push_completed=True,paths=names))


if __name__=='__main__':main()
