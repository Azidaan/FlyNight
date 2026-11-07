"""Разрешённая ветка Трекинг: проверить копию, сохранить все прежние записи."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import sys,json,shutil,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def hashes(vault):
    return {p.relative_to(vault).as_posix():sha(p) for p in vault.rglob('*') if p.is_file() and '.obsidian' not in p.parts}


def main():
    from scripts.obsidian_tracking_hypotheses import load,paths
    from scripts.obsidian_fundamental_overview import build
    from scripts.audit_fundamental_obsidian import audit
    data=load();assert data and len(paths())==10
    vault=ROOT/'ObsidianFlyVis';before=hashes(vault)
    protected={p:h for p,h in before.items() if p.startswith('Гипотезы — подходы/') or (p.startswith('09 Запланировано/') and Path(p).name.startswith(('Б1','Б2','Б3')))}
    legacy={p:h for p,h in before.items() if p.startswith('10 Проблемы мухи/')}
    assert len(protected)==10
    out=ROOT/'runs/continuation_20261028'/('tracking_hypotheses_added_'+datetime.now(timezone(timedelta(hours=3))).strftime('%Y%m%d_%H%M%S'))
    out.mkdir(exist_ok=False);shutil.copytree(vault,out/'vault-before');shutil.copytree(vault,out/'verified-copy')
    def check(target):
        assert all((target/p).is_file() for p in before)
        assert all(sha(target/p)==h for p,h in protected.items())
        assert all(sha(target/p)==h for p,h in legacy.items())
        r=audit(target)
        assert r['hypothesis_node_count']==21 and r['hypothesis_edge_count']==20 and r['root_rays']==4
        assert r['note_count']==50 and not r['external_hypothesis_edges']
        return r
    copy=out/'verified-copy';build(copy,authorize_tracking=True);first=check(copy)
    build(copy);assert check(copy)['hashes']==first['hashes']
    (out/'copy_audit.json').write_text(json.dumps(first,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    assert hashes(vault)==before,'Vault changed while checking copy'
    build(vault,authorize_tracking=True);result=check(vault)
    assert result['hashes']==first['hashes']
    result.update(old_b123_and_children_byte_identical=True,legacy_problems_byte_identical=True,
        original_files_preserved=True,deleted_files=0,moved_files=0,
        author_request=data['author_request'],authorized_new_nodes=sorted(paths()),
        authorized_modified_old_nodes=['Гипотезы'],verified_copy_before_live=True,
        registry_sha256=sha(ROOT/'experiments/tracking_hypotheses.json'))
    (out/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (ROOT/'runs/continuation_20261028/tracking_hypotheses_latest.json').write_text(json.dumps(dict(
        passed=True,audit=(out/'audit.json').relative_to(ROOT).as_posix()),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ['passed','note_count','stored_note_count','hypothesis_node_count','hypothesis_edge_count','root_rays','deleted_files','moved_files','old_b123_and_children_byte_identical']},ensure_ascii=False))
    print(out)


if __name__=='__main__':main()
