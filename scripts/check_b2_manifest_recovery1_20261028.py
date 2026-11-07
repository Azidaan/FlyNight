"""Повторить проверки после конкурентной записи журнала, не менять тест."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
from datetime import datetime
import sys,subprocess,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.continuation_service_20261028 import RUN,service


def main():
    folder=RUN/'b2_manifest_checks_recovery1';source=RUN/'b2_causal_manifest'
    p=service.read(folder/'protocol.json');end=datetime.fromisoformat(p['original_absolute_deadline']).timestamp()
    c=service.read(source/'controller_result.json');assert c['exit_code']==1 and c['all_owned_ended'] and not any(service.matching(k,v) for k,v in c['owned'].items())
    report=service.read(source/'report.json');assert report['passed'] and report['complete']
    assert all(service.sha(ROOT/e['manifest'])==e['sha256'] for e in report['entries'])
    for i,args in enumerate([['scripts/validate_project_history.py'],['-m','pytest','tests/test_project_history.py','-q']]):
        with (folder/f'check_{i}.log').open('w',encoding='utf-8') as out:
            code=subprocess.run([sys.executable,*args],cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,timeout=max(1,end-time.time()-15)).returncode
        assert code==0
    service.write(folder/'report.json',dict(passed=True,data_not_repeated=True,manifests=104,clips=report['clips'],source_controller_sha256=service.sha(source/'controller_result.json'),cause='Concurrent journal update changed source fingerprint between two index loads; original test correctly invalidated cache; no test weakened',updated=service.now()))
    from scripts.b123_safe_journal_20261027 import journal
    journal('b2_manifest_checks_recovery1','Повторпровероккаталогапослепараллельнойзаписижурнала',
        '110манифест20441клип/104видеоуспешен,ноодинтесткэшаиндексаотказал:междудвумячтениямиrootагентзаписалновуюстрокувЕдиныйплан. Этообоснованнаяинвалидация,нетестовая/научнаяошибка. '
        'Оригинальный110controllerexit1сохранён. Толькоchecksповторенывостаткепрежнего110600сокна;данныенеэкспортировалисьзаново,тестнеизменён. Источник b2_manifest_checks_recovery1/report.json/check_*.log.')


if __name__=='__main__':main()
