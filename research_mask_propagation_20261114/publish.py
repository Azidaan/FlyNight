"""Prepare reviewed graph derivatives and stage only this follow-up audit."""
from common import *
import subprocess,shutil
import scripts.obsidian_level_graph as gen
import scripts.audit_obsidian_levels as aud
D=OUT
def git(*args,**kwargs):return subprocess.check_output(['git',*args],cwd=ROOT,**kwargs)
def put(n,v):
    with (D/n).open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n')
if sys.argv[1]=='stage':
    p=read(D/'publication.json');assert git('rev-parse','HEAD').decode().strip()==p['head'];assert not git('diff','--cached','--name-only').strip();entries=[]
    for name,value in p['files'].items():
        path=Path(value['source']);assert sha(path)==value['sha256'],name
        blob=path.read_bytes();blob=blob if path.suffix.lower() in ['.jpg','.png','.npz','.zip'] else blob.replace(b'\r\n',b'\n');oid=git('hash-object','-w','--stdin',input=blob).decode().strip();entries.append(('100644 '+oid+'\t'+name+'\0').encode())
    git('update-index','-z','--index-info',input=b''.join(entries));git('diff','--cached','--check');print('STAGED',len(entries));sys.exit()
assert read(D/'audit.json')['passed'] and read(D/'checks.json')['passed'];assert not git('diff','--cached','--name-only').strip()
for name in ['project_history.json','system_problems.json','obsidian_experiment_topics.json','useful_literature.json']:
    assert (D/'before'/name).read_bytes().replace(b'\r\n',b'\n')==git('show','HEAD:experiments/'+name),('Unaccounted baseline edit',name)

vault=ROOT/'ObsidianFlyVis';live=D/'graph_live_v1';pub=D/'graph_publication_v1';baseline=D/'graph_baseline_v1'
settings=(vault/'.obsidian/graph.json').read_bytes()
protected={p.relative_to(vault).as_posix():sha(p) for p in vault.rglob('*.md') if 'Гипотез' in p.as_posix()}
old_manifest=read(vault/'.graph-visible.json');orphans={}
historical=[]
for ref in ['HEAD','HEAD^']:
    historical.append((ref,json.loads(git('show',ref+':ObsidianFlyVis/.graph-visible.json'))))
for file in (vault/'Карта проекта').rglob('*.md'):
    text=file.read_text(encoding='utf8');front=text.split('---\n',2);rel=file.relative_to(vault).as_posix()
    if len(front)!=3 or gen.ACTIVE not in front[1] or rel in old_manifest['files']:continue
    verified=[]
    for ref,manifest in historical:
        candidates=[node for node in manifest['nodes'].values() if node['path']==rel]
        if not candidates:continue
        assert len(candidates)==1
        content=git('show',ref+':ObsidianFlyVis/'+rel)
        if content==file.read_bytes().replace(b'\r\n',b'\n'):verified.append(dict(node=candidates[0],sha256=sha(file),historical_commit=ref))
    assert verified,('Unverified historical note; preserve user file',rel)
    orphans[rel]=verified[0]
# Give the generator its verified historical records so it archives their tags.
# No scientific text or protected hypothesis is rewritten.
original_init=gen.Tree.__init__
def initialize_with_history(self,target):
    original_init(self,target)
    for rel,value in orphans.items():self.old['verified_archival_path:'+rel]=value['node']
gen.Tree.__init__=initialize_with_history
# Two compact text-only copies. Baseline is measured in memory, then its
# work tree is reused for the live build; no third vault snapshot on disk.
def compact_copy(dest):
    assert not dest.exists();dest.mkdir()
    for source in vault.rglob('*.md'):
        if any(part in {'.trash','plugins'} for part in source.relative_to(vault).parts):continue
        target=dest/source.relative_to(vault);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
    for rel in ['.graph-visible.json','.obsidian/graph.json']:
        target=dest/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(vault/rel,target)
for dest in [live,pub]:compact_copy(dest)

class Proxy:
    def __init__(self,overrides):self.overrides=overrides
    def __truediv__(self,n):return self.overrides.get(str(n),ROOT/n)
    def __getattr__(self,n):return getattr(ROOT,n)
    def __fspath__(self):return str(ROOT)
gen.local_file=lambda n,label:'['+label+']('+(ROOT/n).as_uri()+')'
gen.ROOT=Proxy({'experiments/project_history.json':D/'before'/'project_history.json','experiments/useful_literature.json':D/'before'/'useful_literature.json','dissertation/ЕДИНЫЙ_ПЛАН.md':D/'before'/'ЕДИНЫЙ_ПЛАН.md'})
load=gen.system_problems;gen.system_problems=lambda:read(D/'before'/'system_problems.json');gen.build(live);baseline_hashes={f.relative_to(live).as_posix():sha(f) for f in live.rglob('*.md')}
gen.ROOT=ROOT;gen.system_problems=load;gen.build(live);live_audit=aud.audit(live);assert live_audit['passed']
changed=[]
for new in (live/'Карта проекта').rglob('*.md'):
    rel=new.relative_to(live).as_posix();target=vault/rel
    if not target.exists() or target.read_bytes()!=new.read_bytes():
        if target.exists():
            head_result=subprocess.run(['git','show','HEAD:ObsidianFlyVis/'+rel],cwd=ROOT,capture_output=True)
            unchanged_head=head_result.returncode==0 and target.read_bytes().replace(b'\r\n',b'\n')==head_result.stdout
            assert sha(target)==baseline_hashes.get(rel) or unchanged_head or (rel in orphans and sha(target)==orphans[rel]['sha256']),('Dirty note differs from generated baseline and HEAD',rel)
        changed.append(rel)
for rel,h in protected.items():assert sha(live/rel)==h
for rel in changed:
    target=vault/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((live/rel).read_bytes())
(vault/'.graph-visible.json').write_bytes((live/'.graph-visible.json').read_bytes())
assert (vault/'.obsidian/graph.json').read_bytes()==settings
before=(D/'before'/'ЕДИНЫЙ_ПЛАН.md').read_text(encoding='utf8');after=(ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md').read_text(encoding='utf8');marker='## Журнал изменений\n';bp,bt=before.split(marker,1);ap,at=after.split(marker,1);assert bp==ap and at.endswith(bt);extra=at[:-len(bt)]
headplan=git('show','HEAD:dissertation/ЕДИНЫЙ_ПЛАН.md').decode().replace('\r\n','\n');(D/'publication_plan.md').write_text(headplan.replace(marker,marker+extra,1),encoding='utf8')
gen.ROOT=aud.ROOT=Proxy({'dissertation/ЕДИНЫЙ_ПЛАН.md':D/'publication_plan.md'});gen.build(pub);pub_audit=aud.audit(pub);assert pub_audit['passed']
color_manifest=read(pub/'.graph-visible.json')
color_checks={}
for identity,expected in [('identity_repair_observed_episode_20261112','выполнено'),('identity_repair_combined_assignment_20261112','выполнено'),('identity_repair_holdout_correspondence_20261112','без-положительного-итога'),('identity_repair_holdout_recovery_20261112','без-положительного-итога'),('association_weights_2179_copy_20261112','без-положительного-итога'),('history_representation_2179_20261113','без-положительного-итога'),('visible_components_2179_20261114','выполнено'),('yolo26n_seg_visible_part_20261114','без-положительного-итога'),('flyvis_visible_mask_reader_20261114','без-положительного-итога')]:
    node=color_manifest['nodes']['exp:'+identity];assert node['color']==expected
    assert 'граф-проекта/'+expected in (pub/node['path']).read_text(encoding='utf8')
    color_checks[identity]=dict(path=node['path'],color=expected)
identity='mask_propagation_20261114';expected='выполнено' if read(D/'result.json')['scientific_passed'] else 'без-положительного-итога';node=color_manifest['nodes']['exp:'+identity];assert node['color']==expected;color_checks[identity]=dict(path=node['path'],color=expected)
put('graph_science_color_verification.json',dict(passed=True,checks=color_checks,success_rgb='#39ab61',scope='Old scientific outcomes preserved. New conditional mask-reader result colored from its registered criteria; protected hypothesis tree unchanged.'))
put('publication_storage_receipt.json',dict(passed=True,calculation_artifact_bytes=sum((D/n).stat().st_size for n in ['audit.json','result.json','predictions.npz','scores.npz','memory_predictions.npz','memory_scores.npz']),task_bytes_before_frozen_archive=sum(f.stat().st_size for f in D.rglob('*') if f.is_file()),graph_copy_count=2,graph_copy_format='Textandmanifests; nobinaryvaultcopies.',own_outputs_drive='E:',temp_drive='E:',cleanup_performed=False,scope='TaskfilesandrequiredrecordkeepingonE; not wholeappC history orphysicalallocation.'))
selected={}
node_paths={node['path'] for node in read(pub/'.graph-visible.json')['nodes'].values()}|{node['path'] for node in old_manifest['nodes'].values()}|set(orphans)
nodes=[dict(path=path) for path in sorted(node_paths)]
names=['ObsidianFlyVis/'+node['path'] for node in nodes]
batch=git('cat-file','--batch',input=('\n'.join('HEAD:'+n for n in names)+'\n').encode());position=0
for node,name in zip(nodes,names):
    end=batch.index(b'\n',position);header=batch[position:end].decode();position=end+1
    if header.endswith(' missing'):previous=b''
    else:
        _,kind,size=header.split();assert kind=='blob';size=int(size);previous=batch[position:position+size];position+=size+1
    path=pub/node['path']
    if previous.replace(b'\r\n',b'\n')!=path.read_bytes().replace(b'\r\n',b'\n'):selected[name]=str(path)
for rel,h in protected.items():assert sha(pub/rel)==h
selected['ObsidianFlyVis/.graph-visible.json']=str(pub/'.graph-visible.json');selected['dissertation/ЕДИНЫЙ_ПЛАН.md']=str(D/'publication_plan.md')
for n in ['experiments/project_history.json','experiments/system_problems.json','experiments/obsidian_experiment_topics.json','experiments/useful_literature.json']:selected[n]=str(ROOT/n)
# Explicit task-owned tree; immutable cache arrays stay local, with audit hashes.
skip={'tmp','anchors','continuity','candidate_panels','candidate_frames','sequential_frames','source_renderer_repeat','pytest_temp','pytest_cache','synthetic_fixture','synthetic_fixture_v2','frames','hypothesis_before_bytes','temp','before','execution','__pycache__','publication_text','graph_baseline_v1','graph_live_v1','graph_publication_v1'}
for path in sorted(D.rglob('*')):
    if not path.is_file():continue
    rel=path.relative_to(D)
    if any(part in skip or part.startswith('hypothesis_preview') or part.startswith('self_test') or part.startswith('fixture') for part in rel.parts):continue
    if path.match('queue_publication*') or path.match('publication*') or path.name in ['trace.json','features.npz']:continue
    if path.suffix.lower() not in ['.py','.json','.md','.txt','.log','.csv','.png','.jpg','.svg','.sha256','.zip','.npz'] and path.name!='.gitignore':continue
    selected[path.relative_to(ROOT).as_posix()]=str(path)
for path in sorted((D/'execution').glob('*')):
    if path.is_file() and path.suffix in ['.json','.log']:selected[path.relative_to(ROOT).as_posix()]=str(path)

aud.ROOT=ROOT
assert aud.audit(vault)['passed']
put('graph_check.json',dict(passed=True,live=live_audit,publication=pub_audit,changed=changed,protected_hypotheses=protected,settings_unchanged=True,repaired_verified_historical_orphans=orphans,repair='Apply archived-note changes as well as current manifest nodes. Prior helper omitted these derived tag changes. Scientific source text preserved.'))
selected[(D/'graph_check.json').relative_to(ROOT).as_posix()]=str(D/'graph_check.json')
selected[(D/'publication_storage_receipt.json').relative_to(ROOT).as_posix()]=str(D/'publication_storage_receipt.json')
# Archive every original selected byte before standard publication formatting.
import zipfile,ast
archive=D/'frozen_source_bytes.zip';copies=D/'publication_text';copies.mkdir(exist_ok=False);normalization=[]
with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as frozen:
    for name,value in list(selected.items()):
        source=Path(value);raw=source.read_bytes();frozen.writestr(name,raw)
        if source.suffix.lower() not in ['.py','.json','.md','.txt','.log','.csv','.svg','.sha256'] and source.name!='.gitignore':continue
        text=raw.decode('utf8').replace('\r\n','\n')
        if name in {p.relative_to(ROOT).as_posix() for p in D.glob('*')} or D in source.parents:
            clean='\n'.join(line.rstrip(' \t') for line in text.split('\n')).rstrip('\n')
            clean=clean+'\n' if clean and text.endswith('\n') else clean
        else:clean=text
        if source.suffix=='.py':assert ast.dump(ast.parse(text),include_attributes=False)==ast.dump(ast.parse(clean),include_attributes=False)
        if source.suffix=='.json':assert json.loads(text)==json.loads(clean)
        target=copies/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(clean.encode('utf8'))
        selected[name]=str(target)
        normalization.append(dict(file=name,frozen_sha256=sha(source),published_sha256=sha(target),changed=raw!=target.read_bytes()))
put('publication_formatting.json',dict(passed=True,archive_sha256=sha(archive),files=normalization,frozen_working_files_unchanged=True,python_AST_and_JSON_exact=True))
selected[archive.relative_to(ROOT).as_posix()]=str(archive)
selected[(D/'publication_formatting.json').relative_to(ROOT).as_posix()]=str(D/'publication_formatting.json')
put('publication.json',dict(passed=True,head=git('rev-parse','HEAD').decode().strip(),files={n:dict(source=v,sha256=sha(v)) for n,v in selected.items()}))
print('PREPARED',len(selected),'GRAPH_CHANGED',len(changed))
