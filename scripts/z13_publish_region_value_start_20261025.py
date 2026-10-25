"""Publish the preregistered signed-utility action without binary weights or logs."""
from pathlib import Path
import sys
import subprocess
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_candidate_reserve_20261021 import read,write,sha,now
from scripts.z13_publish_native20_final_20261024 import foreign_unchanged,graph_rows
RUN=ROOT/'runs/z13_region_value_20261025'


def main():
    assert not (RUN/'start_publication.json').exists()
    assert '59 passed' in (RUN/'prepare_tests.log').read_text(encoding='utf-8')
    assert 'OK:' in (RUN/'prepare_validation.log').read_text(encoding='utf-8')
    p=read(RUN/'protocol.json');launch=read(RUN/'launch.json')
    write(RUN/'prepare_verified.json',dict(at=now().isoformat(),tests59_passed=True,validator_exit0=True,
        tests_sha256=sha(RUN/'prepare_tests.log'),protocol_sha256=sha(RUN/'protocol.json')))
    handoff=ROOT/'runs/z13_weekend_20261024/HANDOFF.md'
    message=('Актуален z13_region_value_v1 в runs/z13_region_value_20261025, CPU12моделей, только цель ошибки '
        '+1польза/0нейтрально/−1вред. Локальный monitor сам проверяет все модели/первичные строки/интервалы, '
        'каталог/граф/scoped push. Запуск '+str(launch)+'; реальные PID+birth — controller.json. '
        'Исходный save '+p['save_at']+', hard '+p['hard_stop']+'; не продлевать. GPU не используется, '
        'признаки прежние. Смотреть monitor_failure.json/worker.stderr.log/status.json. '
        'Обе прежние региональные потери отрицательны: unweighted +0,002п.п., balanced −0,191п.п., интервалы включают0. '
        'К HOTA не продвигать их. Следующая signed версия должна пройти2п.п. и все6парныхконтролей. '
        'При её провале не перебирать дальше скалярные потери: обоснован следующий узел отсутствующего наблюдаемого '
        'контекста сопоставления, сначала дешёвый проверяемый вход/контроль без мухи. '
        'Архитектурная веха предыдущих двух новых узлов подготовлена в генераторе, ещё не опубликована; '
        'после этой серии сверить подтверждённые сравнения/обновить один лист и просмотреть. '
        'Авторизация до26.10 07:40МСК,57чужихграфправок/.gitignore сохранить.')
    handoff.write_text('# Актуально — '+now().strftime('%d.%m.%Y %H:%M')+' MSK\n\n'+message+'\n\nНиже прежние передачи сохранены.\n\n'+handoff.read_text(encoding='utf-8'),encoding='utf-8')
    foreign=foreign_unchanged();receipt=RUN/'start_owned_graph.json'
    write(receipt,{n:sha(ROOT/n) for _,n in graph_rows() if (ROOT/n).is_file() and n not in foreign})
    subprocess.run([str(ROOT/'.venv/Scripts/python.exe'),'scripts/z13_build_vault_preserving_edits.py','--report',str(RUN/'vault_start.json'),'--refresh-owned',str(receipt)],cwd=ROOT,check=True,capture_output=True)
    names=[Path(__file__).relative_to(ROOT).as_posix(),'tracking/z13_region_value_loss.py','tests/test_z13_region_value_loss.py',
        'scripts/z13_region_value_20261025.py','scripts/z13_region_value_console.ps1','experiments/project_history.json',
        'dissertation/ЕДИНЫЙ_ПЛАН.md','runs/z13_weekend_20261024/HANDOFF.md']
    names += [(RUN/n).relative_to(ROOT).as_posix() for n in ('protocol.json','training/protocol.json','launch.json','prepare_verified.json','start_owned_graph.json','vault_start.json')]
    names += [n for _,n in graph_rows() if n not in foreign and (ROOT/n).is_file()]
    names=sorted(set(names));assert not set(names)&foreign and '.gitignore' not in names
    staged=set(subprocess.check_output(['git','diff','--cached','--name-only','-z'],cwd=ROOT).decode().split('\0'))-{''};assert staged<=set(names)
    assert not (ROOT/'.git/index.lock').exists()
    assert subprocess.check_output(['git','remote','get-url','origin'],cwd=ROOT).decode().strip()=='https://gitlab.com/aziteam-group/drbdiss.git'
    for i in range(0,len(names),25):subprocess.run(['git','add','-f','--',*names[i:i+25]],cwd=ROOT,check=True,capture_output=True)
    subprocess.run(['git','diff','--cached','--check'],cwd=ROOT,check=True)
    subprocess.run(['git','commit','--quiet','-m','Register signed assignment utility as a matched FlyVis readout target','-m','Co-authored-by: Codex <codex@openai.com>'],cwd=ROOT,check=True)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip();subprocess.run(['git','push','origin','main'],cwd=ROOT,check=True)
    foreign_unchanged();write(RUN/'start_publication.json',dict(at=now().isoformat(),commit=commit,push_completed=True,paths=names))


if __name__=='__main__':main()
