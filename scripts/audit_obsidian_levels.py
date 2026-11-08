"""Check the physical note files, graph filter, tree edges and source coverage."""
from collections import Counter
from pathlib import Path
from urllib.parse import quote
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
    assert len({n['number'] for n in nodes.values()})==len(nodes),'Duplicate hierarchy number'
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
        assert 'пункт: "'+n['number']+'"' in front[1],path
        assert int(re.search(r'^уровень: (\d+)$',front[1],re.M)[1])==len(n['number'].split('.'))
        assert not re.fullmatch(r'\d{4}-\d{2}(?:-\d{2})?',n['title']),n['path']
        if key.startswith('day:'):
            assert n['date']==key[4:] and 'дата: "'+key[4:]+'"' in front[1]
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
    parents={n['parent'] for n in nodes.values()}
    empty_leaves=[]
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
        if k not in parents:
            body=(vault/n['path']).read_text(encoding='utf8').split('---\n',2)[2]
            body=re.sub(r'^#+.*$', '', body, flags=re.M)
            body=re.sub(r'\[[^\]]*\]\([^)]*\)', '', body).strip()
            if not body or re.fullmatch(r'\*\*Статус:\*\*[^\n]*',body):empty_leaves.append(n['path'])
    assert not empty_leaves,empty_leaves
    assert roots==set(SECTION_NAMES) and edges==expected and len(edges)==len(nodes)-len(roots)
    catalog=read(ROOT/'experiments/project_history.json')
    visible_cards={c['id'] for c in catalog['experiments'] if 'drb' not in (c['area']+' '+c['title']).lower()}
    assert {k[4:] for k in nodes if k.startswith('exp:')}==visible_cards
    assert not any(re.search(r'\bDRB\b',n['title'],re.I) for n in nodes.values())
    # The four large FlyVis outcome groups must not be flat stars.
    topics=read(ROOT/'experiments/obsidian_experiment_topics.json')['topics']
    for state in ('completed_negative','infrastructure','inconclusive','completed_positive'):
        hub='area:flyvis:'+state
        assert hub in nodes
        assert not any(k.startswith('exp:') and n['parent']==hub for k,n in nodes.items())
    for card in catalog['experiments']:
        key='exp:'+card['id']
        if key not in nodes:continue
        path=[];parent=nodes[key]['parent']
        while parent and parent.startswith('topic:'):
            path.append(nodes[parent]['title']);parent=nodes[parent]['parent']
        if path:assert list(reversed(path))==topics[card['id']],card['id']
    glossary=read(ROOT/'experiments/metrics_glossary.json')
    assert sections['Термины']==1+sum(not re.search(r'\bDRB\b',g['title'],re.I) for group in ('metrics','terms') for g in glossary[group].values())
    assert all('fly:P'+str(i) in nodes for i in range(1,12))
    recovered=read(ROOT/'experiments/obsidian_archive_review.json')['ideas']
    for item in recovered:
        matches=[n for k,n in nodes.items() if k.startswith('plan:idea:09 Запланировано/Идеи из плана/'+item['id']+' ')]
        assert len(matches)==1 and matches[0]['color']==item['color']
        assert item['finding'] in (vault/matches[0]['path']).read_text(encoding='utf8')
    journal=(ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md').read_text(encoding='utf8').split('## Журнал изменений',1)[1]
    expected_entries=len(re.findall(r'^\*\*\d\d\.\d\d\.\d\d\d\d',journal,re.M))
    entries=sum(len(re.findall(r'^## \d\d:\d\d ',(vault/n['path']).read_text(encoding='utf8'),re.M))
                for k,n in nodes.items() if k.startswith('day:'))
    assert entries==expected_entries,(entries,expected_entries)
    literature=ROOT/'experiments/useful_literature.json'
    if literature.exists():
        for source in read(literature)['sources']:
            key='literature:'+source['id']
            assert key in nodes
            text=(vault/nodes[key]['path']).read_text(encoding='utf8')
            assert source['doi'] in text and source['url'] in text
            for target in source['related']:
                assert target in nodes
                assert quote(nodes[target]['path'][:-3],safe='') in text
                reverse=(vault/nodes[target]['path']).read_text(encoding='utf8')
                assert quote(nodes[key]['path'][:-3],safe='') in reverse
    return dict(passed=True,note_count=len(nodes),components=len(roots),edges=len(edges),sections=sections,
                journal_entries=entries,missing_targets=[],external_edges=[],empty_leaves=empty_leaves,experiments=len(visible_cards),catalog_experiments=len(catalog['experiments']))
