"""Публиковать техническую ступень после обязательных проверок каталога."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    run=ROOT/'runs/b123_20261027/sam2_runtime_profile'
    assert 'OK: каталог согласован с источниками' in (run/'catalog_validation.log').read_text(encoding='utf-8')
    assert '58 passed' in (run/'catalog_tests.log').read_text(encoding='utf-8')
    old=json.loads((ROOT/'runs/b123_20261027/busca_output_pilot/controller_result.json').read_text())
    assert old['exit_code']==0 and old['all_owned_ended']
    p=json.loads((run/'protocol.json').read_text(encoding='utf-8'))
    for name,h in p['sha256'].items():assert sha(ROOT/name)==h,name
    source='scripts/b123_sam2_runtime_profile_20261027.py'
    job=ROOT/'runs/b123_20261027/ready_jobs/110_sam2_runtime_profile.json';assert not job.exists()
    job.write_text(json.dumps(dict(title='Собственная масочная ветвь — причинность и стоимость SAM2',
        folder=run.relative_to(ROOT).as_posix(),script=source,script_sha256=sha(ROOT/source),protocol_sha256=sha(run/'protocol.json')),
        ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Published SAM2 technical profile')
if __name__=='__main__':main()
