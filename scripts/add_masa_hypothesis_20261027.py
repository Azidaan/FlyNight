"""Однократное прямо разрешённое добавление MASA; сначала копия, потом основное хранилище."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import sys,json,hashlib,shutil,re
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.recovery_research_service_20261027 import RUN,read,write,sha,now

def hashes(vault):return {p.relative_to(vault).as_posix():sha(p) for p in vault.rglob('*') if p.is_file() and '.obsidian' not in p.parts}

def main():
    from scripts.obsidian_fundamental_overview import build
    from scripts.obsidian_hypothesis_tree import START,END
    from scripts.obsidian_masa_hypothesis import PATH
    from scripts.audit_fundamental_obsidian import audit
    read(RUN/'masa_hypothesis_authorization.json');vault=ROOT/'ObsidianFlyVis'
    out=RUN/('masa_hypothesis_addition_'+datetime.now(timezone(timedelta(hours=3))).strftime('%Y%m%d_%H%M%S'));out.mkdir(exist_ok=False)
    before=hashes(vault);protected={p:h for p,h in before.items() if p=='Гипотезы.md' or p.startswith('Гипотезы — подходы/') or (p.startswith('09 Запланировано/') and Path(p).name.startswith(('Б1','Б2','Б3')))}
    assert len(protected)==10 and PATH+'.md' not in before
    parent=next(p for p in protected if p.startswith('09 Запланировано/Б1 '));old=(vault/parent).read_text(encoding='utf-8')
    body=lambda s:re.sub(re.escape(START)+r'.*?'+re.escape(END)+r'\s*','',s,flags=re.S).rstrip()
    shutil.copytree(vault,out/'vault-before');shutil.copytree(vault,out/'verified-copy');copy=out/'verified-copy'
    def check(target):
        assert all((target/p).is_file() for p in before)
        assert all(sha(target/p)==h for p,h in protected.items() if p!=parent)
        assert body((target/parent).read_text(encoding='utf-8'))==body(old)
        assert (target/(PATH+'.md')).is_file()
        r=audit(target);assert r['hypothesis_node_count']==11 and r['hypothesis_edge_count']==10 and r['root_rays']==3 and not r['external_hypothesis_edges']
        return r
    build(copy,authorize_masa=True);first=check(copy);build(copy);assert check(copy)['hashes']==first['hashes']
    write(out/'copy_audit.json',first);assert hashes(vault)==before,'Исходное хранилище изменилось во время проверки'
    build(vault,authorize_masa=True);result=check(vault);assert result['hashes']==first['hashes']
    result.update(original_files_preserved=True,old_hypotheses_unchanged_except_b1_children=True,main_b1_text_unchanged=True,authorized_new_nodes=[PATH],
        protected_old_file_count=len(protected),deleted_files=0,moved_files=0,verified_copy_before_live=True,updated=now())
    write(out/'audit.json',result);write(RUN/'masa_hypothesis_latest.json',dict(audit=(out/'audit.json').relative_to(ROOT).as_posix(),node=PATH+'.md',passed=True))
    print(json.dumps({k:result[k] for k in ['passed','note_count','hypothesis_node_count','hypothesis_edge_count','root_rays','main_b1_text_unchanged','deleted_files','moved_files']},ensure_ascii=False))

if __name__=='__main__':main()
