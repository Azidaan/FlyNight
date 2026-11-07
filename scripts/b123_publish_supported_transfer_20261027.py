"""Поставить новый замер и готовую сверку, не запускать дубли."""
from pathlib import Path
import hashlib,json
import psutil
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'runs/b123_20261027';RUN=BASE/'supported_transfer'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
    assert '58 passed' in (RUN/'catalog_tests.log').read_text(encoding='utf-8') and 'OK: каталог согласован с источниками' in (RUN/'catalog_validation.log').read_text(encoding='utf-8')
    p=read(RUN/'protocol.json');assert p['search_radius']==24
    for name,h in p['sha256'].items():assert sha(ROOT/name)==h,name
    for folder in ['online_supported_verification','detection_supported_recovery','detection_supported_audit']:
        e=read(BASE/folder/'controller_result.json');assert e['exit_code']==0 and e['all_owned_ended']
        for pid,birth in e['owned'].items():
            if psutil.pid_exists(int(pid)):assert abs(psutil.Process(int(pid)).create_time()-birth)>.01
    source='scripts/b123_supported_transfer_20261027.py';job=BASE/'ready_jobs/200_supported_transfer.json';assert not job.exists()
    write(RUN/'status.json',dict(state='ready',stage='Серия подготовлена, первый полный видеопроход определит цену'))
    write(job,dict(title='Перенос фиксированного правила на другую часть val',folder=RUN.relative_to(ROOT).as_posix(),script=source,script_sha256=sha(ROOT/source),protocol_sha256=sha(RUN/'protocol.json')))
    audit=BASE/'supported_transfer_audit';audit.mkdir();source='scripts/b123_audit_supported_transfer_20261027.py'
    write(audit/'protocol.json',dict(budget_seconds=300,author_stop='2026-10-27T07:20:00+03:00',device='cpu',action='Independent saved-output and raw-array audit, no inference or repeated TrackEval',source_sha256=sha(ROOT/source),pilot_protocol_sha256=sha(RUN/'protocol.json')))
    write(BASE/'ready_jobs/210_supported_transfer_audit.json',dict(title='Сверка переноса на отдельной части val',folder=audit.relative_to(ROOT).as_posix(),script=source,script_sha256=sha(ROOT/source),protocol_sha256=sha(audit/'protocol.json')))
    print('Published200/210; one GPU, original author deadline, disjoint development split')
if __name__=='__main__':main()
