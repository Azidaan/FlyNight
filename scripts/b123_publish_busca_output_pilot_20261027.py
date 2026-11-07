"""Публикация полного пилота только после зелёных проверок и цены runtime."""
from pathlib import Path
import hashlib,json,sys
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    run=ROOT/'runs/b123_20261027/busca_output_pilot'
    assert 'OK: каталог согласован с источниками' in (run/'catalog_validation.log').read_text(encoding='utf-8')
    assert '58 passed' in (run/'catalog_tests.log').read_text(encoding='utf-8')
    p=json.loads((run/'protocol.json').read_text(encoding='utf-8'))
    for name,h in p['sha256'].items():assert sha(ROOT/name)==h,name
    profile=ROOT/'runs/b123_20261027/busca_runtime_profile/recovery1'
    c=json.loads((profile/'controller_result.json').read_text());assert c['exit_code']==0 and c['all_owned_ended']
    job=ROOT/'runs/b123_20261027/ready_jobs/100_busca_output_pilot.json';assert not job.exists()
    source='scripts/b123_busca_output_pilot_20261027.py'
    job.write_text(json.dumps(dict(title='Проверка продолжения потерянных объектов сетью BUSCA — полный пилот20val',
        folder=run.relative_to(ROOT).as_posix(),script=source,script_sha256=sha(ROOT/source),protocol_sha256=sha(run/'protocol.json')),
        ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Published full pilot; first complete video controls remaining cost')
if __name__=='__main__':main()
