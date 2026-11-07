"""Адресная проверка публичных репозиториев автора LTR; без исполнения кода."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from b123_snapshot_recovery_sources_20261027 import fetch
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/b123_20261027/ltr_exact_source_lookup'
def main():
    OUT.mkdir(exist_ok=False)
    url='https://api.github.com/users/ManuelBGomez/repos?per_page=100'
    body=fetch(url,1_000_000);(OUT/'author_repositories.json').write_bytes(body)
    repos=json.loads(body)
    result=dict(at=datetime.now(timezone.utc).isoformat(),source=url,sha256=hashlib.sha256(body).hexdigest(),
        repositories=[dict(name=x['full_name'],description=x.get('description'),default_branch=x['default_branch'],url=x['html_url']) for x in repos],
        exact_ltr_repository_confirmed=False,models_executed=False,datasets_downloaded=False)
    (OUT/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result['repositories'],ensure_ascii=False))
if __name__=='__main__':main()
