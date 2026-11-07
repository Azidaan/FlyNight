"""Проверить реальные узлы и рёбра, а не подписи или предполагаемое состояние UI."""
from pathlib import Path
from urllib.parse import urlparse,parse_qs,unquote
import argparse,hashlib,json,re,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.obsidian_hypothesis_tree import child_paths,TITLES,FOLDER

def audit(vault):
    from scripts.obsidian_visible_sections import visible_files,SECTIONS
    selected=visible_files(vault)
    notes={Path(name).with_suffix('').as_posix():vault/name for name in selected}
    all_notes={p.relative_to(vault).with_suffix('').as_posix():p for p in vault.rglob('*.md')}
    manifest=json.loads((vault/'.graph-visible.json').read_text(encoding='utf-8'))
    graph=json.loads((vault/'.obsidian/graph.json').read_text(encoding='utf-8'))
    assert graph['search']==manifest['search'] and graph['hideUnresolved'] and not graph['showAttachments']
    assert {p.relative_to(vault).as_posix() for p in vault.rglob('*.md') if re.search(manifest['regex'],p.relative_to(vault).as_posix())}==set(selected)
    main_targets=re.findall(r'\[\[([^|\]#]+)',notes['Гипотезы'].read_text(encoding='utf-8'))
    from scripts.obsidian_tracking_hypotheses import paths as tracking_paths,HUB as TRACKING_HUB,load as tracking_load,method_path
    tracking=tracking_load();extra=tracking_paths()
    from scripts.obsidian_core_dynamics_hypothesis import paths as core_paths
    core=core_paths()
    rays=(4 if tracking else 3)+len(core)
    assert len(main_targets)==rays and len(set(main_targets))==rays
    if tracking:assert TRACKING_HUB in main_targets
    assert core.issubset(main_targets)
    component={'Гипотезы',*main_targets,*child_paths(),*extra,*core}
    assert component.issubset(notes)
    all_edges=set();externals=[];missing=[];stealth=[];numeric=[]
    for name,path in notes.items():
        content=path.read_text(encoding='utf-8')
        for target in re.findall(r'\[\[([^|\]#]+)',content):
            target=target.removesuffix('.md')
            if target not in all_notes:missing.append([name,target])
            elif target not in notes:continue
            else:
                edge=tuple(sorted((name,target)));all_edges.add(edge)
                if (name in component)!=(target in component):externals.append(edge)
            if re.match(r'^[\d. ,\[\]-]+$',target):numeric.append([name,target])
        for url in re.findall(r'obsidian://open\?[^)\s]+',content):
            target=parse_qs(urlparse(url).query).get('file',[''])[0]
            if target not in all_notes:missing.append([name,'stealth:'+target])
            else:stealth.append([name,target])
    internal={edge for edge in all_edges if set(edge).issubset(component)}
    assert len(component)==4+len(child_paths())+len(extra)+len(core) and len(internal)==len(component)-1
    expected={tuple(sorted(('Гипотезы',target))) for target in main_targets}
    assert expected.issubset(internal)
    for main in main_targets:
        children={b if a==main else a for a,b in internal if main in (a,b)}-{'Гипотезы'}
        if main in core:
            assert not children
            continue
        if tracking and main==TRACKING_HUB:
            assert children=={method_path(m) for m in tracking['methods']}
            continue
        key=next(k for k in TITLES if Path(main).name.startswith('Б'+k[-1]+' '))
        assert children=={FOLDER+'/'+t for t in TITLES[key].values()}
    assert not externals and not missing and not numeric,(externals,missing,numeric)
    from scripts.obsidian_system_problems import load,paths,HUB,problem_path,pattern_path
    problems_data=load();problem_component=paths(problems_data)
    assert problem_component.issubset(notes)
    problem_external=[e for e in all_edges if (e[0] in problem_component)!=(e[1] in problem_component)]
    problem_internal={e for e in all_edges if set(e).issubset(problem_component)}
    expected_problem_edges={tuple(sorted((HUB,p))) for p in problem_component-{HUB}}
    by_id={p['id']:p for p in problems_data['problems']}
    expected_problem_edges.update(tuple(sorted((pattern_path(p),problem_path(by_id[i])))) for p in problems_data['patterns'] for i in p['problems'])
    assert not problem_external and problem_internal==expected_problem_edges
    assert not any(name.startswith('10 Проблемы мухи/') for name in notes)
    assert not any(name.startswith('06 Журнал/') for name in notes)
    assert not any(name.endswith('.canvas') for name in selected)
    assert len(notes)<=53+len(core)+max(0,len(problems_data['problems'])-6), 'Detailed or obsolete notes leaked into overview'
    for name in [*SECTIONS.values(),'07 Исследования/Исследования']:
        assert name in notes and any(name in edge for edge in all_edges),name
    assert not any(re.fullmatch(r'[0-9eE., +\[\]-]+',Path(name).name) for name in notes)
    return dict(passed=True,note_count=len(notes),stored_note_count=len(all_notes),hidden_note_count=len(all_notes)-len(notes),visible_edges=[list(e) for e in sorted(all_edges)],
        hypothesis_nodes=sorted(component),hypothesis_edges=[list(e) for e in sorted(internal)],
        hypothesis_node_count=len(component),hypothesis_edge_count=len(internal),root_rays=rays,
        external_hypothesis_edges=externals,missing_targets=missing,numeric_nodes=numeric,
        problem_nodes=sorted(problem_component),problem_node_count=len(problem_component),
        problem_edges=[list(e) for e in sorted(problem_internal)],external_problem_edges=problem_external,
        problem_root_rays=len(problem_component)-1,
        stealth_links=len(stealth),journal_notes=0,canvas_files=0,
        hashes={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in notes.items()},
        section_hubs=SECTIONS,scope='Saved files and graph filter; no assertion about live Obsidian renderer')

def main():
    p=argparse.ArgumentParser();p.add_argument('--vault',type=Path,default=ROOT/'ObsidianFlyVis');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=audit(a.vault);a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ['passed','note_count','hypothesis_node_count','hypothesis_edge_count','root_rays','external_hypothesis_edges','missing_targets','journal_notes','canvas_files']},ensure_ascii=False))
if __name__=='__main__':main()
