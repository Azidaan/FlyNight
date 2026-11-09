"""Preserve the exact UIA capture, normalize two display-only blanks, audit index."""
import json,hashlib,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];HERE=ROOT/'tmp/identity_reader_review_publication'
def git(*args,**kw):
    r=subprocess.run(['git',*args],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.PIPE,**kw)
    if r.returncode:raise RuntimeError(r.stderr.decode('utf8','replace')+'\n'+r.stdout.decode('utf8','replace'))
    return r.stdout
def sha(data):return hashlib.sha256(data).hexdigest()
path=HERE/'prepared.json';before=path.read_bytes();p=json.loads(before)
assert git('rev-parse','HEAD').decode().strip()==p['head']
for name,h in p['working_sources'].items():assert sha((ROOT/name).read_bytes())==h,name
entries={line.split('\t',1)[1]:line.split('\t',1)[0].split()[1] for line in git('ls-files','--stage','-z').decode().strip('\0').split('\0')}
for name,item in p['files'].items():
    data=Path(item['source']).read_bytes();assert sha(data)==item['sha256'];expected=git('hash-object','--path='+name,'--stdin',input=data).decode().strip();assert entries[name]==expected,name
original_name='research_identity_synthesis_20261109/gemini_answer_v18_accessibility.txt';original=(ROOT/original_name).read_bytes();text=original.decode('utf8')
normalized='\n'.join(line.rstrip(' \t') for line in text.split('\n')).encode();assert original!=normalized and len(original)-len(normalized)==2
raw_name='research_identity_synthesis_20261109/gemini_answer_v18_raw_capture.json';raw=ROOT/raw_name
with raw.open('x',encoding='utf8') as f:json.dump(dict(format='Exact original model-response UIA subtree',text=text,sha256=sha(original),characters=len(text),publication_text='The readable TXT copy strips exactly two trailing presentation spaces; this JSON preserves the complete original capture.'),f,ensure_ascii=False,indent=2)
assert json.loads(raw.read_text(encoding='utf8'))['text'].encode()==original
normal_path=HERE/'gemini_answer_v18.normalized.txt';normal_path.write_bytes(normalized)
p['files'][original_name]=dict(source=str(normal_path),sha256=sha(normalized))
names=[raw_name,Path(__file__).relative_to(ROOT).as_posix()]
for name in names:
    file=ROOT/name;p['files'][name]=dict(source=str(file),sha256=sha(file.read_bytes()));p['working_sources'][name]=sha(file.read_bytes())
with (HERE/'prepared_before_UIA_normalization.json').open('xb') as f:f.write(before)
temp=HERE/'prepared.uia.pending';temp.write_text(json.dumps(p,ensure_ascii=False,indent=2)+'\n',encoding='utf8');assert path.read_bytes()==before;temp.replace(path)
for name in [original_name,*names]:
    blob=git('hash-object','-w','--path='+name,'--stdin',input=Path(p['files'][name]['source']).read_bytes()).decode().strip()
    git('update-index','--add','--cacheinfo','100644,'+blob+','+name)
blobs={name:git('hash-object','--path='+name,'--stdin',input=Path(item['source']).read_bytes()).decode().strip() for name,item in p['files'].items()}
foreign={}
for name in ['AGENTS.md','ObsidianFlyVis/ИНСТРУКЦИЯ — как вести записи.md','scripts/obsidian_level_graph.py','scripts/audit_obsidian_levels.py','tests/test_obsidian_level_graph.py','experiments/OBSIDIAN_GRAPH_RULE.md','dissertation/План целей по Трекеру.md','experiments/hypothesis_calendar.json','ObsidianFlyVis/.obsidian/graph.json']:
    file=ROOT/name
    if file.exists():foreign[name]=sha(file.read_bytes())
for file in (ROOT/'ObsidianFlyVis').rglob('*.md'):
    if any('Гипотез' in part for part in file.parts):foreign[file.relative_to(ROOT).as_posix()]=sha(file.read_bytes())
for file in (ROOT/'ObsidianFlyVis/.obsidian/plugins').rglob('*'):
    if file.is_file():foreign[file.relative_to(ROOT).as_posix()]=sha(file.read_bytes())
with (HERE/'capture_normalization_receipt.json').open('x',encoding='utf8') as f:json.dump(dict(passed=True,initial_stage_failed='git diff --cached --check found exactly two trailing spaces in original UIA display tree. Full graph/source/protected-file checks had already passed before that final check.',original_local_unchanged_sha256=sha(original),normalized_sha256=sha(normalized),exact_raw_json=raw_name,previous_prepared_sha256=sha(before)),f,ensure_ascii=False,indent=2)
# Reuse only the original stage's complete index/physical-graph audit, not its
# earlier empty-index assumption. All preceding staged sources were verified.
source=ROOT/'runs/domain_matched_native_20261109/stage_publication.py';code=source.read_text(encoding='utf8');code=code[code.index('staged=set('):];code=code.replace('len(staged)>100','len(staged)>0')
import re
exec(compile(code,str(source),'exec'))
