"""Save real cache progress and synthetic execution verification, not new science."""
from pathlib import Path
import sys
import subprocess
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_candidate_reserve_20261021 import read,write,sha,now
from scripts.z13_publish_native20_final_20261024 import foreign_unchanged
RUN=ROOT/'runs/z13_region_context_20261025'
SMOKE=ROOT/'runs/z13_context_execution_smoke_20261025/recovery1'


def main():
    assert not (RUN/'execution_check.json').exists()
    receipt=read(SMOKE/'service_receipt.json');p=read(RUN/'protocol.json')
    assert receipt['synthetic_only'] and receipt['all24_variant_seed_paths_exercised'] and receipt['code_sha256']==p['executed_training_sha256']
    price=read(RUN/'cache/price.json');assert price['proceed']
    assert not (RUN/'pipeline_failure.json').exists()
    completed=len(list((RUN/'cache/completed').glob('*.json')));assert completed>=2
    write(RUN/'execution_check.json',dict(at=now().isoformat(),actual_completed_cache_videos=completed,initial_cache_price=price,
        synthetic_execution_receipt=SMOKE.relative_to(ROOT).as_posix()+'/service_receipt.json',
        all24_generated_execution_paths_verified=True,scientific_sources_and_budget_unchanged=True))
    from scripts import z13_localization_labels_20261023 as notes
    notes.IDENT='z13_region_context_v1'
    notes.journal('проверено фактическое исполнение контекстной очереди',
       'Нативная CPU-очередь запущена '+read(RUN/'launch.json')['at']+'; controller.json и pipeline.json содержат PID+birth. '
       'Уже сохранено '+str(completed)+' завершённых точных базовых повторов из130; цена первого2замера '+str(price['estimate'])+'с '
       'меньше оставшихся '+str(price['remaining'])+'с, первая ценовая ступень пройдена. '
       'На синтетических тензорах отдельно исполнены все24пути вариантов/сидов с тем же хешем сгенерированной функции; '
       'это служебная проверка связности, не научные результаты. Первая синтетическая проба имела слишком короткий '
       'макетный120сбюджет и остановилась на консервативной цене; сохранена. Новая отдельная синтетическая проба600с '
       'успешна за секунды. Реальные cache1200с/CPU1800с и научные источники не менялись. '
       'Источники runs/z13_region_context_20261025/execution_check.json и runs/z13_context_execution_smoke_20261025/recovery1/service_receipt.json.')
    foreign_unchanged()
    names=[Path(__file__).relative_to(ROOT).as_posix(),'scripts/z13_context_execution_smoke_20261025.py','scripts/z13_context_execution_smoke_recovery1_20261025.py',
        'dissertation/ЕДИНЫЙ_ПЛАН.md',(RUN/'execution_check.json').relative_to(ROOT).as_posix(),(SMOKE/'service_receipt.json').relative_to(ROOT).as_posix()]
    staged=set(subprocess.check_output(['git','diff','--cached','--name-only','-z'],cwd=ROOT).decode().split('\0'))-{''};assert staged<=set(names)
    assert not (ROOT/'.git/index.lock').exists()
    assert subprocess.check_output(['git','remote','get-url','origin'],cwd=ROOT).decode().strip()=='https://gitlab.com/aziteam-group/drbdiss.git'
    subprocess.run(['git','add','-f','--',*names],cwd=ROOT,check=True)
    subprocess.run(['git','diff','--cached','--check'],cwd=ROOT,check=True)
    subprocess.run(['git','commit','--quiet','-m','Verify all generated context-readout execution paths before training','-m','Co-authored-by: Codex <codex@openai.com>'],cwd=ROOT,check=True)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip();subprocess.run(['git','push','origin','main'],cwd=ROOT,check=True)
    foreign_unchanged();write(RUN/'execution_publication.json',dict(at=now().isoformat(),commit=commit,push_completed=True,paths=names))


if __name__=='__main__':main()
