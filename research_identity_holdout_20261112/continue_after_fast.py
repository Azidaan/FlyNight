from common import *
import time,subprocess,psutil
psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
registration_sha=sha(OUT/'register_matrix.py');runner_sha=sha(OUT/'run_queue.py')
deadline=datetime.fromisoformat('2026-11-13T07:35:00+03:00')
started=time.monotonic()
while True:
    assert time.monotonic()-started<1900 and datetime.now(ZoneInfo('Europe/Moscow'))<deadline
    q=read(OUT/'queue3_execution.json')
    if q.get('complete'):
        assert q['passed'],'Independent audit failed; no downstream execution'
        break
    time.sleep(.5)
assert sha(OUT/'register_matrix.py')==registration_sha and sha(OUT/'run_queue.py')==runner_sha
time.sleep(.5)
for command in [[sys.executable,str(OUT/'register_matrix.py')],[sys.executable,str(OUT/'run_queue.py'),str(OUT/'queue4.json')]]:
    subprocess.run(command,cwd=ROOT,check=True,creationflags=subprocess.CREATE_NO_WINDOW)
save('continuation_receipt.json',dict(passed=True,ended=now(),registration_sha256=registration_sha,runner_sha256=runner_sha,queue4_passed=read(OUT/'queue4_execution.json')['passed']))
print('REGISTERED_A_MATRIX_CHAIN_COMPLETE')
