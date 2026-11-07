"""Сохранить отказ схемы карточки и повторить только обязательные проверки."""
from pathlib import Path
from datetime import datetime
import time, sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.continuation_service_20261028 import RUN,service


def main():
    original=RUN/'b2_causal960_checks'
    deadline=service.read(original/'execution_deadline.json')['deadline']
    budget=int(datetime.fromisoformat(deadline).timestamp()-time.time()-20)
    assert budget>250
    history=service.read(ROOT/'experiments/project_history.json')
    card=next(c for c in history['experiments'] if c['id']=='b2_causal960_path_v1')
    card['evidence']=[dict(kind='file',ref='runs/continuation_20261028/b2_causal960_profile/protocol.json',
                          claim='ДоGPU:960/реальныевремена/crowd/сильныеконтроли/безобучения')]
    service.write(ROOT/'experiments/project_history.json',history)
    from scripts.b123_safe_journal_20261027 import journal
    journal(card['id'],'Исправлена схема ссылки карточки до запускаGPUБ2',
            '112CPU отказала навалидаторе: evidence была строкой,а требуется объект kind/ref/claim. '
            'Старые script/controllerexit1/check0FAIL сохранены. Исправлено только поле evidence, '
            'повтор112a проверок в остатке исходного абсолютного600с срока '+deadline+'. GPUещёне запускалась.')
    folder=RUN/'b2_causal960_checks_recovery1'
    script='scripts/check_b2_causal960_recovery1_20261028.py'
    service.write(folder/'protocol.json',dict(budget_seconds=budget,original_deadline=deadline,
                  original_script_sha256=service.sha(ROOT/'scripts/check_b2_causal960_20261028.py'),
                  original_controller_sha256=service.sha(original/'controller_result.json'),created=service.now()))
    service.write(RUN/'ready_jobs/112a_b2_causal960_checks_recovery1.json',dict(name='b2_causal960_checks_recovery1',
                  folder=folder.relative_to(ROOT).as_posix(),script=script,
                  script_sha256=service.sha(ROOT/script),protocol_sha256=service.sha(folder/'protocol.json')))


if __name__=='__main__':main()
