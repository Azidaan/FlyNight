"""Проверить реальные переходы и сохранность старого содержимого гипотез."""
from pathlib import Path
from urllib.parse import urlparse,parse_qs
import json,re,hashlib,sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
VAULT=ROOT/'ObsidianFlyVis';OUT=ROOT/'runs/b123_20261027'


def targets(text):
    return [parse_qs(urlparse(u).query)['file'][0] for u in re.findall(r'obsidian://open\?[^)\s]+',text)]


def main():
    from scripts.obsidian_b123_literature import START,END,SOURCES,path
    grant=json.loads((OUT/'stealth_links_authorization.json').read_text(encoding='utf-8'))
    expected_sources={path(k) for k in SOURCES};expected_branches={p[:-3] for p in grant['before_sha256']}
    rows=[]
    for name in grant['before_sha256']:
        old=(OUT/'stealth_links_build/before'/name).read_bytes()
        current=(VAULT/name).read_bytes()
        assert hashlib.sha256(old).hexdigest()==grant['before_sha256'][name]
        assert current.startswith(old)
        addition=current[len(old):].decode('utf-8')
        assert addition.count(START)==addition.count(END)==1 and '[[' not in addition
        assert set(targets(addition))==expected_sources and len(targets(addition))==2
        rows.append(dict(source=name,targets=sorted(expected_sources),old_content_unchanged=True))
    for source in expected_sources:
        text=(VAULT/(source+'.md')).read_text(encoding='utf-8')
        actual=targets(text)
        assert set(actual)==expected_branches and len(actual)==3
        rows.append(dict(source=source+'.md',targets=sorted(expected_branches)))
    for row in rows:
        for target in row['targets']:assert (VAULT/(target+'.md')).is_file(),target
    result=dict(passed=True,forward_stealth_links=6,return_stealth_links=6,all_targets_exist=True,
        old_hypothesis_content_unchanged=True,links=rows,
        clarification='The earlier build report scope text said no hypothesis files were written; authorized_stealth_sections=3 is the actual scope: old bytes retained and three requested sections appended. Graph settings were not written.')
    (OUT/'stealth_links_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('links','clarification')},ensure_ascii=False))


if __name__=='__main__':main()
