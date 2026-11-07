"""Проверить восстановление на копии, затем вернуть файлы и настроить видимость."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import hashlib,json,shutil,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
VAULT=ROOT/'ObsidianFlyVis'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def hashes(vault):return {p.relative_to(vault).as_posix():sha(p) for p in vault.rglob('*') if p.is_file() and '.obsidian' not in p.parts}
def main():
    refresh_masa='--refresh-masa' in sys.argv
    from scripts.obsidian_masa_hypothesis import PATH as MASA_PATH
    allowed={MASA_PATH+'.md'} if refresh_masa else set()
    if refresh_masa:assert (ROOT/'runs/recovery_research_20261027/masa_hypothesis_authorization.json').is_file()
    from scripts.obsidian_fundamental_overview import build
    from scripts.obsidian_visible_sections import restore_saved_files
    from scripts.audit_fundamental_obsidian import audit
    out=ROOT/'runs/b123_20261027'/('graph_sections_restored_'+datetime.now(timezone(timedelta(hours=3))).strftime('%Y%m%d_%H%M%S'))
    out.mkdir(exist_ok=False)
    before=hashes(VAULT)
    protected={p:h for p,h in before.items() if p=='Гипотезы.md' or p.startswith('Гипотезы — подходы/') or (p.startswith('09 Запланировано/') and Path(p).name.startswith(('Б1','Б2','Б3')))}
    legacy_problems={p:h for p,h in before.items() if p.startswith('10 Проблемы мухи/')}
    from scripts.obsidian_hypothesis_tree import child_paths
    assert len(protected)==4+len(child_paths())
    # Полный снимок существующих файлов, включая настройки открытого графа.
    shutil.copytree(VAULT,out/'vault-before')
    test=out/'verified-copy';shutil.copytree(out/'vault-before',test)
    restored_test=restore_saved_files(test,ROOT)
    build(test,refresh_masa=refresh_masa);first=audit(test)
    assert all((test/p).is_file() for p in before)
    assert all(sha(test/p)==h for p,h in protected.items() if p not in allowed)
    assert all(sha(test/p)==h for p,h in legacy_problems.items())
    build(test,refresh_masa=refresh_masa);assert audit(test)['hashes']==first['hashes']
    (out/'copy-audit.json').write_text(json.dumps(first,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    # Между снимком и применением никто не должен изменить исходные заметки.
    assert hashes(VAULT)==before,'Хранилище изменилось во время проверки; применение остановлено.'
    restored=restore_saved_files(VAULT,ROOT)
    assert restored==restored_test
    build(VAULT,refresh_masa=refresh_masa);result=audit(VAULT)
    assert result['hashes']==first['hashes']
    assert all((VAULT/p).is_file() for p in before)
    assert all(sha(VAULT/p)==h for p,h in protected.items() if p not in allowed)
    assert all(sha(VAULT/p)==h for p,h in legacy_problems.items())
    for item in restored:assert sha(VAULT/item['path'])==item['sha256']
    result.update(restored_files=restored,restored_file_count=len(restored),
        original_files_preserved=True,protected_hypotheses_byte_identical=not refresh_masa,
        legacy_problem_files_byte_identical=True,legacy_problem_file_count=len(legacy_problems),
        protected_hypotheses_unchanged_except_authorized_masa=True,authorized_changed_hypotheses=sorted(allowed),
        deleted_files=0,moved_files=0,verified_copy_before_live=True)
    (out/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('passed','note_count','stored_note_count','hidden_note_count','restored_file_count','deleted_files','moved_files','protected_hypotheses_byte_identical')},ensure_ascii=False))
    print(out)
if __name__=='__main__':main()
