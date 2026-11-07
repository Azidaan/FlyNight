"""Select only the new protocol and its two journal entries for a commit."""
import hashlib
import json
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=Path(__file__).resolve().parent


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    assert read(OUT/'validation_v2.json')['passed']
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()=='01815dc9253108c81737c6723077ab603336d7c2'
    assert not subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT).strip()
    checks=read(OUT/'preparation_checks.json')
    assert sha(ROOT/'dissertation/План целей по Трекеру.md')==checks['tracker_plan_sha256']
    for name,h in checks['protected_sha256'].items():assert sha(ROOT/name)==h,name
    plan=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';before=plan.read_bytes()
    head=subprocess.check_output(['git','show','HEAD:dissertation/ЕДИНЫЙ_ПЛАН.md'],cwd=ROOT).decode('utf-8')
    entries=read(OUT/'journal_v2_entry.json')['entry']+read(OUT/'journal_entry.json')['entry']
    anchor='## Журнал изменений\n\n';assert head.count(anchor)==1
    staged=head.replace(anchor,anchor+entries,1).encode('utf-8')
    marker='**07.11.2026, 20:43 MSK** — Решение автора: дальнейшие варианты ΣΠ не развивать;'
    assert marker in before.decode('utf-8') and marker.encode('utf-8') not in staged
    blob=subprocess.check_output(['git','hash-object','-w','--stdin'],cwd=ROOT,input=staged).decode().strip()
    subprocess.run(['git','update-index','--cacheinfo',f'100644,{blob},dissertation/ЕДИНЫЙ_ПЛАН.md'],cwd=ROOT,check=True)
    names=[p.relative_to(ROOT).as_posix() for p in OUT.iterdir() if p.is_file() and p.name!='commit_message.txt']
    data=b''.join(n.encode('utf-8')+b'\0' for n in sorted(names))
    subprocess.run(['git','add','--pathspec-from-file=-','--pathspec-file-nul'],cwd=ROOT,input=data,check=True)
    assert plan.read_bytes()==before
    subprocess.run(['git','diff','--cached','--check'],cwd=ROOT,check=True)
    files=[p.decode('utf-8') for p in subprocess.check_output(['git','diff','--cached','--name-only','-z'],cwd=ROOT).split(b'\0') if p]
    for name in files:
        if name!='dissertation/ЕДИНЫЙ_ПЛАН.md':
            (ROOT/name).resolve().relative_to(OUT.resolve())
    with (OUT/'staging_receipt.json').open('x',encoding='utf-8',newline='\n') as f:
        json.dump(dict(files=files,staged_plan_blob=blob,working_plan_sha256=hashlib.sha256(before).hexdigest(),
            preexisting_tracker_plan_preserved=True,foreign_journal_entry_excluded=True),f,ensure_ascii=False,indent=2);f.write('\n')
    subprocess.run(['git','add','--',str((OUT/'staging_receipt.json').relative_to(ROOT))],cwd=ROOT,check=True)
    print('Staged protocol files and own journal entries:',len(files)+1)


if __name__=='__main__':main()
