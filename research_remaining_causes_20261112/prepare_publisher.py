from pathlib import Path
root=Path(__file__).resolve().parents[1];out=Path(__file__).resolve().parent
s=(root/'research_stage_localization_20261112/publish.py').read_text(encoding='utf8')
a=s.index("for n in ['common.py','register.py'")
b=s.index('\naud.ROOT=ROOT',a)
s=s[:a]+'''# Only this task's own deliverables. Large local tensor archives are preserved locally.
for path in sorted(D.rglob('*')):
    if not path.is_file():continue
    rel=path.relative_to(D)
    if any(part=='before' or part=='__pycache__' or part.startswith('graph_') or part=='execution' for part in rel.parts):continue
    if path.name.startswith('publication') or path.suffix.lower() not in ['.py','.json','.md','.txt','.log','.png','.svg','.sha256'] and path.name!='.gitignore':continue
    selected[path.relative_to(ROOT).as_posix()]=str(path)
for path in sorted((D/'execution').glob('*')):
    if path.is_file() and path.suffix in ['.json','.log']:selected[path.relative_to(ROOT).as_posix()]=str(path)
''' + s[b:]
needle="assert read(D/'audit.json')['passed'];assert not git('diff','--cached','--name-only').strip()"
assert needle in s
s=s.replace(needle,needle+'''
for name in ['project_history.json','system_problems.json','obsidian_experiment_topics.json','useful_literature.json']:
    assert (D/'before'/name).read_bytes().replace(b'\\r\\n',b'\\n')==git('show','HEAD:experiments/'+name),('Unaccounted baseline catalog edit',name)
''')
with (out/'publish.py').open('x',encoding='utf8') as f:f.write(s)
with (out/'.gitignore').open('x',encoding='utf8') as f:f.write('*.npz\nbefore/\ngraph_*/\npublication*.json\npublication_plan.md\n__pycache__/\n*.pending\n')
print('PREPARED_PUBLISHER')
