"""Готовые проверки каталога и обзор на копии после сохранения полного итога."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import subprocess,sys,json
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'runs/b123_20261027/transfer_final_checks'
def main():
    audit=json.loads((ROOT/'runs/b123_20261027/supported_transfer/recovery2/independent_audit.json').read_text(encoding='utf-8'));assert audit['passed']
    cat=json.loads((ROOT/'experiments/project_history.json').read_text(encoding='utf-8'));card=next(c for c in cat['experiments'] if c['id']=='detection_support_transfer_v1');assert card['state'] in ['completed_positive','completed_negative','inconclusive'] and card['obtained']
    steps=[('catalog_validation.log',['scripts/validate_project_history.py']),
        ('catalog_tests.log',['-m','pytest','tests/test_project_history.py','-q']),
        ('graph_tests.log',['-m','pytest','tests/test_obsidian_transfer_outcome.py','tests/test_obsidian_fundamental_overview.py','-q']),
        ('graph_sync.log',['scripts/restore_obsidian_sections_20261027.py'])]
    for name,args in steps:
        with (RUN/name).open('w',encoding='utf-8') as f:
            done=subprocess.run([sys.executable,*args],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
        assert done.returncode==0,(name,done.returncode)
    assert 'OK: каталог согласован с источниками' in (RUN/'catalog_validation.log').read_text(encoding='utf-8')
    (RUN/'report.json').write_text(json.dumps(dict(complete=True,catalog_validated=True,catalog_tests_passed=True,graph_tests_passed=True,graph_checked_on_copy_then_applied=True,scientific_state=card['state'],scientific_result_source='runs/b123_20261027/supported_transfer/recovery2/report.json'),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Catalogue checks and fundamental overview complete; scientific results unchanged')
if __name__=='__main__':main()
