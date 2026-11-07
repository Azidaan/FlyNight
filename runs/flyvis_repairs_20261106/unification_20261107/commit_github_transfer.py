"""Commit only migration documentation and receipts; preserve both histories."""
from pathlib import Path
import subprocess,json,os
ROOT=Path(__file__).resolve().parents[3];U=Path(__file__).resolve().parent
def git(*args,data=None):
    p=subprocess.run(['git','-c','gc.auto=0',*args],cwd=ROOT,input=data,capture_output=True)
    if p.returncode:raise RuntimeError(p.stderr.decode(errors='replace'))
    return p.stdout
head=git('rev-parse','HEAD').decode().strip();assert head=='283e289d47ab6a848184eba21990db9bcbbc0dff'
paths=['README.md','AGENTS.md','dissertation/ЕДИНЫЙ_ПЛАН.md']
assert set(git('diff','--cached','--name-only','-z').decode().split('\0'))-{''}<=set(paths)
(U/'local_union_receipt.json').write_bytes((U/'final_union/installed.json').read_bytes())
paths += [p.relative_to(ROOT).as_posix() for p in [U/'local_union_receipt.json',U/'prepare_github_docs.py',Path(__file__),U/'history_after_merge.log',U/'cache_after_merge.log']]
for name in ('fa0dd6336aca4599bc9f7eb4bc6d2b5f',):
    paths += [(U/'uncapped_checks'/name/f).relative_to(ROOT).as_posix() for f in ('memory_guard.json','memory_guard.worker.log')]
last=json.loads((U/'cache_after_merge.log').read_text().split('SYSTEM_MEMORY_GUARD ')[1])
assert last['returncode']==0 and last['reason'] is None
name=last['command'][-1].split('=')[1]
paths += [(U/'uncapped_checks'/name/f).relative_to(ROOT).as_posix() for f in ('memory_guard.json','memory_guard.worker.log')]
assert set(git('diff','HEAD','--name-only','-z').decode().split('\0'))-{''}==set(paths[:3])
git('add','-f','--',*paths)
tree=git('write-tree').decode().strip()
other=git('rev-parse','refs/codex/github-flynight-initial-20261107').decode().strip()
assert git('show',other+':README.md').strip()==b''
message=('Prepare FlyNight publication and preserve its initial GitHub history\n\n'
         'Use the consolidated primary branch, document local checks, and retain the frozen GitLab remote. '
         'Public publication was explicitly authorized by the owner.\n\nCo-authored-by: Codex <noreply@openai.com>\n')
commit=git('commit-tree',tree,'-p',head,'-p',other,data=message.encode()).decode().strip()
git('update-ref','refs/heads/codex/system-integrity-20261106-e8f674ce',commit,head)
assert not git('diff','--cached','--name-only')
assert not git('diff','--name-only')
remotes=git('remote').decode().split()
if 'github' not in remotes:git('remote','add','github','https://github.com/Azidaan/FlyNight.git')
assert git('remote','get-url','github').decode().strip()=='https://github.com/Azidaan/FlyNight.git'
print(json.dumps(dict(commit=commit,parents=[head,other],remote='github')))
