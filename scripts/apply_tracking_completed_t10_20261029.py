"""Only T10 may change under author's explicit tried=green instruction."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import sys,hashlib,json,shutil
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def hashes(v):return {p.relative_to(v).as_posix():sha(p) for p in v.rglob('*') if p.is_file() and '.obsidian' not in p.parts}
def main():
    from scripts.obsidian_tracking_hypotheses import load,method_path
    from scripts.obsidian_fundamental_overview import build
    from scripts.audit_fundamental_obsidian import audit
    data=load();method=next(m for m in data['methods'] if m['id']=='T10');assert method['status']=='completed'
    allowed=method_path(method)+'.md';vault=ROOT/'ObsidianFlyVis';before=hashes(vault)
    protected={p:h for p,h in before.items() if p!=allowed and (p=='Гипотезы.md' or p.startswith(('Гипотезы — подходы/','Гипотезы — трекинг/','10 Проблемы мухи/')) or (p.startswith('09 Запланировано/') and Path(p).name.startswith(('Б1','Б2','Б3'))))}
    out=ROOT/'runs/continuation_20261029'/('t10_completed_graph_'+datetime.now(timezone(timedelta(hours=3))).strftime('%Y%m%d_%H%M%S'));out.mkdir()
    shutil.copytree(vault,out/'vault-before');shutil.copytree(vault,out/'verified-copy')
    def check(target):
        assert all((target/p).is_file() for p in before)
        assert all(sha(target/p)==h for p,h in protected.items())
        result=audit(target);assert result['note_count']==53
        assert (result['hypothesis_node_count'],result['hypothesis_edge_count'],result['root_rays'])==(24,23,4)
        assert not result['external_hypothesis_edges']
        assert 'проверка-гипотезы/завершена' in (target/allowed).read_text(encoding='utf-8')
        return result
    build(out/'verified-copy',authorize_tracking=True);first=check(out/'verified-copy')
    build(out/'verified-copy');assert check(out/'verified-copy')['hashes']==first['hashes']
    assert hashes(vault)==before,'Concurrent graph change: do not overwrite'
    build(vault,authorize_tracking=True);result=check(vault);assert result['hashes']==first['hashes']
    result.update(original_files_preserved=True,deleted_files=0,moved_files=0,only_authorized_tracking_node_changed=allowed,other_hypotheses_and_legacy_byte_identical=True)
    (out/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(out)
if __name__=='__main__':main()
