"""Check the physical note files, graph filter, tree edges and source coverage."""
from collections import Counter
from pathlib import Path
import re

from scripts.obsidian_level_graph import ACTIVE, COLORS, FOLDER, LINK, ROOT, SECTION_NAMES, read

def audit(vault):
    manifest=read(vault/'.graph-visible.json')
    nodes=manifest['nodes']
    settings=read(vault/'.obsidian/graph.json')
    assert settings['search']==manifest['search']==f'path:"{FOLDER}/" tag:#{ACTIVE}'
    assert settings['hideUnresolved'] and not settings['showAttachments'] and not settings['showTags']
    assert settings['colorGroups']==[dict(query='tag:#граф-проекта/'+k,color=dict(a=1,rgb=v)) for k,v in COLORS.items()]
    by_path={n['path']:k for k,n in nodes.items()}
    assert len(by_path)==len(nodes) and sorted(by_path)==manifest['files']
    found=set()
    edges=set()
    for file in (vault/FOLDER).rglob('*.md'):
        text=file.read_text(encoding='utf8')
        front=text.split('---\n',2)
        if len(front)!=3 or ACTIVE not in front[1]:continue
        path=file.relative_to(vault).as_posix()
        key=by_path[path]
        found.add(path)
        n=nodes[key]
        expected_tags={ACTIVE}
        if n['color']:expected_tags.add('граф-проекта/'+n['color'])
        assert set(re.search(r'tags: \[([^\]]+)\]',front[1])[1].split(', '))==expected_tags
        for m in LINK.finditer(front[2]):
            target=m[1]+'.md'
            assert target in by_path,(key,target)
            edges.add((key,by_path[target]))
        assert not re.search(r'!?\[[^\[\]\n]*\]\((?!https?://|obsidian://|file:///)[^.\n)][^\n)]*\)',front[2]),path
    assert found==set(by_path)
    expected=set()
    roots=set()
    sections=Counter()
    for k,n in nodes.items():
        sections[n['section']]+=1
        p=n['parent']
        if p is None:
            roots.add(k)
            assert n['number']==str(SECTION_NAMES.index(k)+1)
            continue
        assert n['number'].rsplit('.',1)[0]==nodes[p]['number']
        assert n['section']==nodes[p]['section']
        expected.add((p,k))
        if n['section']=='Термины':assert p=='Термины'
    assert roots==set(SECTION_NAMES) and edges==expected and len(edges)==len(nodes)-len(roots)
    catalog=read(ROOT/'experiments/project_history.json')
    assert {k[4:] for k in nodes if k.startswith('exp:')}=={c['id'] for c in catalog['experiments']}
    glossary=read(ROOT/'experiments/metrics_glossary.json')
    assert sections['Термины']==1+len(glossary['metrics'])+len(glossary['terms'])
    assert all('fly:P'+str(i) in nodes for i in range(1,12))
    journal=(ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md').read_text(encoding='utf8').split('## Журнал изменений',1)[1]
    expected_entries=len(re.findall(r'^\*\*\d\d\.\d\d\.\d\d\d\d',journal,re.M))
    entries=sum(len(re.findall(r'^## \d\d:\d\d ',(vault/n['path']).read_text(encoding='utf8'),re.M))
                for k,n in nodes.items() if k.startswith('day:'))
    assert entries==expected_entries,(entries,expected_entries)
    return dict(passed=True,note_count=len(nodes),components=len(roots),edges=len(edges),sections=sections,
                journal_entries=entries,missing_targets=[],external_edges=[],experiments=len(catalog['experiments']))
