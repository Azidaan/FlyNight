from pathlib import Path
root=Path(__file__).resolve().parents[1];out=Path(__file__).resolve().parent
s=(root/'research_stage_localization_20261112/publish.py').read_text(encoding='utf8')
a=s.index("for n in ['common.py','register.py'");b=s.index('\naud.ROOT=ROOT',a)
s=s[:a]+'''# Explicit task-owned tree; immutable cache arrays stay local, with audit hashes.
skip={'before','execution','__pycache__','publication_text','graph_baseline_v1','graph_live_v1','graph_publication_v1'}
for path in sorted(D.rglob('*')):
    if not path.is_file():continue
    rel=path.relative_to(D)
    if any(part in skip for part in rel.parts):continue
    if path.match('queue8*') or path.match('publication*'):continue
    if path.suffix.lower() not in ['.py','.json','.md','.txt','.log','.csv','.png','.svg','.sha256'] and path.name!='.gitignore':continue
    selected[path.relative_to(ROOT).as_posix()]=str(path)
for path in sorted((D/'execution').glob('*')):
    if path.is_file() and path.suffix in ['.json','.log']:selected[path.relative_to(ROOT).as_posix()]=str(path)
for name in ['tracking/census_correspondence.py','tracking/reference_template_history.py','tracking/correspondence_repair_guard.py','tracking/appearance_recovery_policy.py','tracking/identity_repair_adapter.py','tests/test_identity_repair_components.py']:
    selected[name]=str(ROOT/name)
''' + s[b:]
needle="assert read(D/'audit.json')['passed'];assert not git('diff','--cached','--name-only').strip()"
s=s.replace(needle,needle+'''
for name in ['project_history.json','system_problems.json','obsidian_experiment_topics.json','useful_literature.json']:
    assert (D/'before'/name).read_bytes().replace(b'\\r\\n',b'\\n')==git('show','HEAD:experiments/'+name),('Unaccounted baseline edit',name)
''')
start=s.index("previous_manifest=json.loads(git(");end=s.index('# Give the generator',start)
s=s[:start]+'''historical=[]
for ref in ['HEAD','HEAD^']:
    historical.append((ref,json.loads(git('show',ref+':ObsidianFlyVis/.graph-visible.json'))))
for file in (vault/'Карта проекта').rglob('*.md'):
    text=file.read_text(encoding='utf8');front=text.split('---\\n',2);rel=file.relative_to(vault).as_posix()
    if len(front)!=3 or gen.ACTIVE not in front[1] or rel in old_manifest['files']:continue
    verified=[]
    for ref,manifest in historical:
        candidates=[node for node in manifest['nodes'].values() if node['path']==rel]
        if not candidates:continue
        assert len(candidates)==1
        content=git('show',ref+':ObsidianFlyVis/'+rel)
        if content==file.read_bytes().replace(b'\\r\\n',b'\\n'):verified.append(dict(node=candidates[0],sha256=sha(file),historical_commit=ref))
    assert verified,('Unverified historical note; preserve user file',rel)
    orphans[rel]=verified[0]
''' + s[end:]
old="put('publication.json',dict(passed=True,head=git('rev-parse','HEAD').decode().strip(),files={n:dict(source=v,sha256=sha(v)) for n,v in selected.items()}))"
assert old in s
s=s.replace(old,'''# Archive every original selected byte before standard publication formatting.
import zipfile,ast
archive=D/'frozen_source_bytes.zip';copies=D/'publication_text';copies.mkdir(exist_ok=False);normalization=[]
with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as frozen:
    for name,value in list(selected.items()):
        source=Path(value);raw=source.read_bytes();frozen.writestr(name,raw)
        if source.suffix.lower() not in ['.py','.json','.md','.txt','.log','.csv','.svg','.sha256'] and source.name!='.gitignore':continue
        text=raw.decode('utf8').replace('\\r\\n','\\n')
        if name in {p.relative_to(ROOT).as_posix() for p in D.glob('*')} or D in source.parents:
            clean='\\n'.join(line.rstrip(' \\t') for line in text.split('\\n')).rstrip('\\n')
            clean=clean+'\\n' if clean and text.endswith('\\n') else clean
        else:clean=text
        if source.suffix=='.py':assert ast.dump(ast.parse(text),include_attributes=False)==ast.dump(ast.parse(clean),include_attributes=False)
        if source.suffix=='.json':assert json.loads(text)==json.loads(clean)
        target=copies/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(clean.encode('utf8'))
        selected[name]=str(target)
        normalization.append(dict(file=name,frozen_sha256=sha(source),published_sha256=sha(target),changed=raw!=target.read_bytes()))
put('publication_formatting.json',dict(passed=True,archive_sha256=sha(archive),files=normalization,frozen_working_files_unchanged=True,python_AST_and_JSON_exact=True))
selected[archive.relative_to(ROOT).as_posix()]=str(archive)
selected[(D/'publication_formatting.json').relative_to(ROOT).as_posix()]=str(D/'publication_formatting.json')
put('publication.json',dict(passed=True,head=git('rev-parse','HEAD').decode().strip(),files={n:dict(source=v,sha256=sha(v)) for n,v in selected.items()}))''')
s=s.replace("path.suffix.lower() in ['.jpg','.png','.npz']","path.suffix.lower() in ['.jpg','.png','.npz','.zip']")
(out/'publish.py').write_bytes((s.rstrip()+'\n').encode())
(out/'.gitignore').write_bytes(b'*.npz\nbefore/\ngraph_*/\npublication_text/\npublication*.json\npublication_plan.md\n__pycache__/\n*.pending\n')
print('PREPARED_FROZEN_BYTE_SAFE_PUBLISHER')
