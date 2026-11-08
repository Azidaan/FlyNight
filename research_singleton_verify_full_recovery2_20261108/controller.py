import sys,time,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_singleton_verify_full_recovery2_20261108.common import *


def main():
    began=time.monotonic()
    while not (RECOVERY/'deadline_registered.json').exists() and time.monotonic()-began<10:time.sleep(.1)
    assert (RECOVERY/'deadline_registered.json').exists();second_phase()
    import research_singleton_verify_full_recovery1_20261108.controller as previous
    original=subprocess.Popen
    def redirected(command,*args,**kwargs):
        # Mutate the caller's command too, so its started receipt records the
        # actual entrypoint rather than the pre-redirection path.
        for i,value in enumerate(command):
            if str(value).replace('\\','/').endswith('research_singleton_verify_full_recovery1_20261108/stage.py'):
                command[i]='research_singleton_verify_full_recovery2_20261108/stage.py'
        return original(command,*args,**kwargs)
    subprocess.Popen=redirected
    try:
        previous.launch('full_audit_recovery2','own_audit','audit');assert read(FULL/'independent_audit.json')['passed']
        write(RUN/'full_audit_queue_result.json',dict(passed=True,audit_sha256=sha(FULL/'independent_audit.json'),recovered_dispatcher='full_recovery2',
            recovery_phase_sha256=sha(RECOVERY/'phase.json'),original_schema_failure='full_recovery1/children/full_audit.log'),new=True)
        previous.subsequent();second_phase()
    finally:subprocess.Popen=original
    write(RECOVERY/'queue_result.json',dict(passed=True,full_audit_sha256=sha(FULL/'independent_audit.json'),age_result_sha256=sha(ROOT/'runs/singleton_age_20261108/queue_result.json'),
        diagnosis_result_sha256=sha(ROOT/'runs/singleton_error_audit_20261108/queue_result.json')),new=True)


if __name__=='__main__':main()
