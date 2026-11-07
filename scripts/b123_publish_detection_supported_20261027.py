"""Публикация новой CPU-ступени и независимой сверки в прежней очереди."""
from pathlib import Path
import hashlib,json
import psutil
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'runs/b123_20261027';RUN=BASE/'detection_supported_recovery'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
    assert 'OK: каталог согласован с источниками' in (RUN/'catalog_validation.log').read_text(encoding='utf-8')
    assert '58 passed' in (RUN/'catalog_tests.log').read_text(encoding='utf-8')
    p=read(RUN/'protocol.json')
    for path,h in p['sha256'].items():assert sha(ROOT/path)==h,path
    for folder in ['sam2_query_output_pilot','sam2_query_audit']:
        end=read(BASE/folder/'controller_result.json');assert end['exit_code']==0 and end['all_owned_ended']
        for pid,birth in end['owned'].items():
            if psutil.pid_exists(int(pid)):assert abs(psutil.Process(int(pid)).create_time()-birth)>.01
    source='scripts/b123_detection_supported_recovery_20261027.py';job=BASE/'ready_jobs/160_detection_supported_recovery.json';assert not job.exists()
    write(job,dict(title='Восстановление с поддержкой текущим обнаружением',folder=RUN.relative_to(ROOT).as_posix(),script=source,script_sha256=sha(ROOT/source),protocol_sha256=sha(RUN/'protocol.json')))
    audit=BASE/'detection_supported_audit';audit.mkdir();source='scripts/b123_audit_detection_supported_20261027.py'
    write(audit/'protocol.json',dict(budget_seconds=180,author_stop='2026-10-27T07:20:00+03:00',device='cpu',action='Saved output audit after stage160; no inference or repeated TrackEval',pilot_protocol_sha256=sha(RUN/'protocol.json'),source_sha256=sha(ROOT/source)))
    write(BASE/'ready_jobs/170_detection_supported_audit.json',dict(title='Сверка сохранённых выходов с поддержкой детектора',folder=audit.relative_to(ROOT).as_posix(),script=source,script_sha256=sha(ROOT/source),protocol_sha256=sha(audit/'protocol.json')))
    print('Published160 and170 in existing queue; no new GPU calculation')
if __name__=='__main__':main()
