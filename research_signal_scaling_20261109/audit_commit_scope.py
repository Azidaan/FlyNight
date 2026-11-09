"""Inspect the actual complete commit and reject unrelated/private material."""
import json,hashlib,re,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];HERE=ROOT/'tmp/signal_scaling_publication'
def git(*args,**kw):return subprocess.check_output(['git',*args],cwd=ROOT,**kw)
def digest(b):return hashlib.sha256(b).hexdigest()
head=git('rev-parse','HEAD').decode().strip();parents=git('rev-list','--parents','-n','1',head).decode().split()
p=json.loads((HERE/'prepared.json').read_text(encoding='utf8'));stage=json.loads((HERE/'staging_receipt.json').read_text(encoding='utf8'))
assert parents==[head,p['head']] and p['head']=='dd91b93750c7021164f384bcc54aa1af2813b82d'
changed=git('diff-tree','--no-commit-id','--name-only','-r','-z',head).decode().strip('\0').split('\0');assert len(changed)==stage['staged_files']
assert set(changed)<=set(p['files']) and not git('diff','--cached','--name-only').strip()
assert not any(n in ['AGENTS.md','ObsidianFlyVis/.obsidian/graph.json'] or '/Гипотезы/' in n or n.startswith(('scripts/','.codex/','.agents/')) for n in changed)
findings=[];files=[]
credential=re.compile(rb'(?:ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,}|sk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{30,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|AIza[A-Za-z0-9_-]{30,})')
account=re.compile(rb'(?:Google'+rb' Account:|accounts\.google\.com/SignOutOptions|From your IP'+rb' address|@gmail\.com|Cook'+rb'ie:|Authorization: Be'+rb'arer)')
for name in changed:
    data=git('show',head+':'+name)
    assert git('rev-parse',head+':'+name).decode().strip()==stage['blobs'][name],name
    if credential.search(data):findings.append(dict(file=name,category='possible_credential'))
    if account.search(data):findings.append(dict(file=name,category='possible_account_or_browser_data'))
    assert not name.lower().endswith(('.png','.jpg','.mp4','.npy','.npz','.pt','.safetensors','.db','.sqlite'))
    files.append(dict(path=name,bytes=len(data),sha256=digest(data)))
assert not findings,findings
git('diff-tree','--check',head+'^',head)
out=dict(passed=True,head=head,parent=p['head'],changed_files=len(files),total_bytes=sum(f['bytes'] for f in files),all_changes_match_explicit_prepared_whitelist=True,no_foreign_drafts_or_hypotheses=True,findings=findings,
    authorization='User AGENTS.md sections0/7 authorize own checked commits and publication to public Azidaan/FlyNight main working branch. Autonomous identity research remains authorized through 2026-11-09.',
    exclusions='No raw UI capture, account or browser inventory; only a technical Gemini draft and its non-submission status. No foreign calendar/plugin/hypothesis drafts, two foreign journal entries, maps, weights or media.',files=files)
with (HERE/'actual_commit_scope_audit.json').open('x',encoding='utf8') as f:json.dump(out,f,ensure_ascii=False,indent=2)
print('ACTUAL_SCALING_COMMIT_SCOPE_AUDITED',head,len(files),out['total_bytes'],flush=True)
