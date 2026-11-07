"""Apply explicitly authorized T10–T12 through a verified copy, preserving T1–T9."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import hashlib,json,shutil,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def hashes(vault):
    return {p.relative_to(vault).as_posix():sha(p) for p in vault.rglob('*') if p.is_file() and '.obsidian' not in p.parts}

def main():
    from scripts.obsidian_tracking_hypotheses import load,method_path,HUB
    from scripts.obsidian_fundamental_overview import build
    from scripts.audit_fundamental_obsidian import audit
    data=load();assert len(data['methods'])==12
    vault=ROOT/'ObsidianFlyVis';before=hashes(vault)
    allowed={'Гипотезы.md',HUB+'.md'}
    protected={p:h for p,h in before.items() if p not in allowed and (
        p.startswith(('Гипотезы — подходы/','Гипотезы — трекинг/','10 Проблемы мухи/')) or
        (p.startswith('09 Запланировано/') and Path(p).name.startswith(('Б1','Б2','Б3'))))}
    assert all(method_path(m)+'.md' in protected for m in data['methods'][:9])
    out=ROOT/'runs/continuation_20261028'/('tracking_reid_added_'+datetime.now(timezone(timedelta(hours=3))).strftime('%Y%m%d_%H%M%S'))
    out.mkdir(exist_ok=False)
    shutil.copytree(vault,out/'vault-before');shutil.copytree(vault,out/'verified-copy')
    def check(target):
        assert all((target/p).is_file() for p in before)
        assert all(sha(target/p)==h for p,h in protected.items())
        r=audit(target)
        assert (r['hypothesis_node_count'],r['hypothesis_edge_count'],r['root_rays'])==(24,23,4)
        assert r['note_count']==53 and not r['external_hypothesis_edges']
        for m in data['methods'][9:]:
            body=(target/(method_path(m)+'.md')).read_text(encoding='utf-8')
            assert 'проверка-гипотезы/не-завершена' in body
        return r
    build(out/'verified-copy',authorize_tracking=True);first=check(out/'verified-copy')
    build(out/'verified-copy');assert check(out/'verified-copy')['hashes']==first['hashes']
    assert hashes(vault)==before,'Vault changed concurrently; do not overwrite'
    build(vault,authorize_tracking=True);result=check(vault)
    assert result['hashes']==first['hashes']
    result.update(original_files_preserved=True,deleted_files=0,moved_files=0,
                  old_b123_t1_t9_legacy_byte_identical=True,verified_copy_before_live=True,
                  authorized_modified_old_nodes=sorted(allowed),new_candidates_untried=True)
    (out/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (ROOT/'runs/continuation_20261028/tracking_hypotheses_latest.json').write_text(json.dumps(dict(passed=True,audit=(out/'audit.json').relative_to(ROOT).as_posix()),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(out)
    print(json.dumps({k:result[k] for k in ('passed','note_count','stored_note_count','hypothesis_node_count','hypothesis_edge_count','root_rays','deleted_files','old_b123_t1_t9_legacy_byte_identical')},ensure_ascii=False))

if __name__=='__main__':main()
