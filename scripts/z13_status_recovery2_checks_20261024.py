"""Repeat checks on fixed index inputs; retain the prior cache-test failure."""
from pathlib import Path
import sys,subprocess,os
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_candidate_reserve_20261021 import write,now
W=ROOT/'runs/z13_weekend_20261024'


def main():
    from scripts.z13_val20_native_masa_20261024 import journal
    journal('повтор проверки индекса на неизменных входах','Первыйнаборпровероквосстановления:59прошло,одинсбойtest_r16_14_cache_invalidated_by_fingerprint. Во времятестамноюпараллельноизменялсякорневойAGENTS.md,которыйвходитвотпечатоккэшаиндекса;проверкапотребовалакэшмеждудвумячтениями,ноотпечатокизменился. Старыйвыводвstatus_recovery2_tests.logсохранён. Тесты/индексаторнеправятся. Передповторомвсе входыиндексазафиксированы,впроцессепроверокихнередактируем;отпечаткидо/послесверяются. Это непроверка научногоэффекта,расчётGPUпродолжается.')
    from evaluation.project_history.index import fingerprint
    before=fingerprint(ROOT)
    for name,args in [('catalog',['scripts/validate_project_history.py']),('tests',['-m','pytest','tests/test_project_history.py','tests/test_z13_status_read_retry.py','-q'])]:
        p=subprocess.run([str(ROOT/'.venv/Scripts/python.exe'),*args],cwd=ROOT,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONIOENCODING':'utf-8'},capture_output=True);(W/('status_recovery2_retry1_'+name+'.log')).write_bytes(p.stdout);(W/('status_recovery2_retry1_'+name+'.stderr.log')).write_bytes(p.stderr);assert p.returncode==0,(name,p.returncode,p.stderr[-1500:]);print(name,0,flush=True)
    after=fingerprint(ROOT);assert before==after,'Index inputs changed during checks'
    write(W/'status_recovery2_retry1_checks.json',dict(at=now().isoformat(),catalog_exit=0,tests_exit=0,input_fingerprint_before=before,input_fingerprint_after=after,previous_failure_retained='status_recovery2_tests.log',tests_or_indexer_changed=False))


if __name__=='__main__':main()
