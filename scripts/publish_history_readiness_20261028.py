"""Опубликовать только CPU-подготовку после текущей Б1 и её аудита."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.recovery_research_service_20261027 import RUN,read,write,sha,now,matching

def main():
    proof=read(RUN/'history_representations_program/preparation_checks.json')
    assert proof['catalog_ok'] and proof['catalog_tests_passed'] and proof['graph_tests_passed']
    q=read(RUN/'queue_recovery1/launch.json');assert matching(q['pid'],q['created'])
    script='scripts/history_representations_readiness_20261028.py'
    folder=RUN/'history_representations_readiness'
    name='380_history_representations_readiness'
    assert not (RUN/'ready_jobs'/f'{name}.json').exists()
    assert not (RUN/'jobs_done'/f'{name}.json').exists()
    sources=[script,'dissertation/Проверка трёх представлений истории — отдельно и совместно.md',
        'runs/recovery_research_20261027/history_representations_program/program.json']
    write(folder/'protocol.json',dict(budget_seconds=600,parent_experiment='history_representations_factorial_v1',
        cpu_only=True,created=now(),after='360_b1_cached_eye_audit',training_ready=False,
        hard_stop=read(RUN/'authorization.json')['hard_stop'],sha256={p:sha(ROOT/p) for p in sources}))
    write(RUN/'ready_jobs'/f'{name}.json',dict(name=name,folder=folder.relative_to(ROOT).as_posix(),
        script=script,script_sha256=sha(ROOT/script),protocol_sha256=sha(folder/'protocol.json')))
    print(name,'опубликована после проверки текущей Б1; только инвентаризация')

if __name__=='__main__':main()
