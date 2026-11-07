"""Публикация полного пилота масочной адаптации после зелёных проверок."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    run=ROOT/'runs/b123_20261027/sam2_output_pilot'
    assert 'OK: каталог согласован с источниками' in (run/'catalog_validation.log').read_text(encoding='utf-8')
    assert '58 passed' in (run/'catalog_tests.log').read_text(encoding='utf-8')
    end=json.loads((ROOT/'runs/b123_20261027/sam2_runtime_profile/controller_result.json').read_text())
    assert end['exit_code']==0 and end['all_owned_ended']
    p=json.loads((run/'protocol.json').read_text(encoding='utf-8'))
    for name,h in p['sha256'].items():assert sha(ROOT/name)==h,name
    source='scripts/b123_sam2_output_pilot_20261027.py'
    job=ROOT/'runs/b123_20261027/ready_jobs/120_sam2_output_pilot.json';assert not job.exists()
    job.write_text(json.dumps(dict(title='Собственная масочная ветвь SAM2 — полный пилот20val',folder=run.relative_to(ROOT).as_posix(),
        script=source,script_sha256=sha(ROOT/source),protocol_sha256=sha(run/'protocol.json')),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Published full mask pilot with measured cost gate after first video')
if __name__=='__main__':main()
