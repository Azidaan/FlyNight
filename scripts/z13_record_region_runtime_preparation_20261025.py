"""Record and publish inactive runtime preparation, without touching active science."""
from pathlib import Path
import sys
import subprocess

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.z13_candidate_reserve_20261021 import read,write,sha,now
from scripts import z13_localization_labels_20261023 as journal_module
from scripts.z13_publish_native20_final_20261024 import foreign_unchanged

RUN=ROOT/'runs/z13_region_runtime_preparation_20261025'
SOURCES=['tracking/z13_region_runtime.py','tests/test_z13_region_runtime.py',
         'scripts/z13_record_region_runtime_preparation_20261025.py']


def main():
    assert not RUN.exists(),'Preserve earlier preparation receipt'
    active=read(ROOT/'runs/z13_region_assignment_20261025/recovery2/status.json')
    assert active['state']=='running' and active['phase']=='features','Avoid concurrent final recording'
    RUN.mkdir()
    with (RUN/'tests.log').open('w',encoding='utf-8') as out:
        test=subprocess.run([str(ROOT/'.venv/Scripts/python.exe'),'-m','pytest','tests/test_z13_region_runtime.py','-q'],cwd=ROOT,stdout=out,stderr=subprocess.STDOUT)
    assert test.returncode==0,'Runtime adapter checks failed'
    write(RUN/'verified.json',dict(at=now().isoformat(),kind='inactive_runtime_preparation',
        scientific_trial='z13_region_assignment_v1',test_exit_code=test.returncode,
        test_log_sha256=sha(RUN/'tests.log'),sources={n:sha(ROOT/n) for n in SOURCES},
        checks=['First association only; restore flag on exception','Identical hard detection/unmatched/no-op costs',
                'Two receptor updates preserve original state, centers and scores exactly'],
        activation='Disabled: separate primary gate, model, calibration, seed, arm and engine hashes required',
        limitations='No actual neural model activated, no closed-loop inference or HOTA measured; unit checks do not establish scientific benefit.'))
    journal_module.IDENT='z13_region_assignment_v1'
    journal_module.journal('подготовлено неактивное подключение регионального считывателя',
        '*Сделано.* tracking/z13_region_runtime.py подключает выбор реальной области или отказа только к первому сопоставлению; '
        'координаты прогноза и последующие стадии трекера не изменяет. Три проверки прошли: ограничение первой стадией и '
        'восстановление флага после исключения; исходные стоимости при отказе и точные категориальные действия; '
        'два последовательных обновления входа глаза воспроизводят прежнее состояние, центры и оценки. '
        'Первичный служебный источник: runs/z13_region_runtime_preparation_20261025/verified.json. '
        'Конфигурация требует пройденного основного критерия, хешей модели/калибровки/движка и совпадения варианта/сида/порога. '
        'Подключение не активировано; текущие научные источники и расчёт recovery2 неизменны. '
        'Это подготовка к условной следующей ступени, не новая оценка эффективности и не HOTA. '
        'Архитектура пока не пересобиралась: в исполняемом научном пути новый блок ещё не включён.')
    with (RUN/'validation.log').open('w',encoding='utf-8') as out:
        validation=subprocess.run([str(ROOT/'.venv/Scripts/python.exe'),'scripts/validate_project_history.py'],cwd=ROOT,stdout=out,stderr=subprocess.STDOUT)
    assert validation.returncode==0
    names=SOURCES+['dissertation/ЕДИНЫЙ_ПЛАН.md',(RUN/'verified.json').relative_to(ROOT).as_posix()]
    assert subprocess.check_output(['git','branch','--show-current'],cwd=ROOT).decode().strip()=='main'
    assert subprocess.check_output(['git','remote','get-url','origin'],cwd=ROOT).decode().strip()=='https://gitlab.com/aziteam-group/drbdiss.git'
    assert not (ROOT/'.git/index.lock').exists()
    staged=set(subprocess.check_output(['git','diff','--cached','--name-only','-z'],cwd=ROOT).decode().split('\0'))-{''}
    assert staged<=set(names)
    foreign_unchanged()
    subprocess.run(['git','add','-f','--',*names],cwd=ROOT,check=True)
    subprocess.run(['git','diff','--cached','--check'],cwd=ROOT,check=True)
    subprocess.run(['git','commit','--quiet','-m','Prepare gated regional readout runtime without activating it',
                    '-m','Co-authored-by: Codex <codex@openai.com>'],cwd=ROOT,check=True)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip()
    subprocess.run(['git','push','origin','main'],cwd=ROOT,check=True)
    foreign_unchanged()
    write(RUN/'publication.json',dict(at=now().isoformat(),commit=commit,push_completed=True,paths=names))
    print(commit)


if __name__=='__main__':main()
