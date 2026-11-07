"""Публикация парного профиля после конца FP32 и зелёного каталога."""
from pathlib import Path
import hashlib,json
import psutil
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    run=ROOT/'runs/b123_20261027/sam2_precision_profile'
    assert 'OK: каталог согласован с источниками' in (run/'catalog_validation.log').read_text(encoding='utf-8')
    assert '58 passed' in (run/'catalog_tests.log').read_text(encoding='utf-8')
    end=json.loads((ROOT/'runs/b123_20261027/sam2_output_pilot/controller_result.json').read_text())
    assert end['all_owned_ended']
    for pid,birth in end['owned'].items():
        if psutil.pid_exists(int(pid)):assert abs(psutil.Process(int(pid)).create_time()-birth)>.01
    p=json.loads((run/'protocol.json').read_text(encoding='utf-8'))
    for name,h in p['sha256'].items():assert sha(ROOT/name)==h,name
    source='scripts/b123_sam2_precision_profile_20261027.py'
    job=ROOT/'runs/b123_20261027/ready_jobs/130_sam2_precision_profile.json';assert not job.exists()
    job.write_text(json.dumps(dict(title='SAM2 — парная проверка стоимости FP32/BF16 и изменений масок',folder=run.relative_to(ROOT).as_posix(),
        script=source,script_sha256=sha(ROOT/source),protocol_sha256=sha(run/'protocol.json')),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Published bounded paired precision profile')
if __name__=='__main__':main()
