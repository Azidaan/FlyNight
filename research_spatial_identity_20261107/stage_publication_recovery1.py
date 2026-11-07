"""Close receipts and stage only this task, retaining the shared worktree."""
import sys
import subprocess
import hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research_spatial_identity_20261107.run_common import RUN,read,sha,write,status,verified_protocol,datetime,MSK

GUARDS={
 'b3367148eedb4c13a6a2fc37730a5e74':0,
 '4236c30a808046b9a880b4470e322856':0,
 '1fe98bbe1ebb4762af313dfc64edb609':15,
 '9d466fdbc01e4d90823e77544648ba28':0,
 '165f5c8b71eb4d2a8885ca8059271dff':1,
 'fc4b922cb1c441769ec882c16d0b6dae':0,
 '9e2a83cc8d9e4e3d95f65ee31ecfa7e2':0,
 '64eb7265e07943b78f9c239da0393a40':125,
 'ec9c4a8fbd724834bf7a535cbecc2b9c':0,
}


def finish():
    import psutil
    verified_protocol();checks=read(RUN/'publication_checks.json');assert checks['passed']
    receipts=[]
    for identifier,expected in GUARDS.items():
        path=ROOT/'runs/flyvis_repairs_20261106/unification_20261107/uncapped_checks'/identifier/'memory_guard.json'
        value=read(path);assert value['returncode']==expected,(identifier,value)
        for pid,born in value['owned'].items():
            try:p=psutil.Process(int(pid))
            except psutil.NoSuchProcess:continue
            assert abs(p.create_time()-born)>.01,(identifier,pid,'still alive')
        receipts.append(dict(path=path.relative_to(ROOT).as_posix(),sha256=sha(path),returncode=expected,reason=value['reason']))
    write(RUN/'controller_verified.json',dict(passed=True,all_owned_ended=True,scientific_complete=True,
        publication_checks_passed=True,receipts=receipts,
        preserved_failures='Original exhaustive scorer intentionally interrupted for exact lazy execution; original audit key error recovered; initial publication start refused for memory. All retained, not relabeled as successes.'),new=True)
    stamp=datetime.now(MSK)
    entry=f"**{stamp.strftime('%d.%m.%Y, %H:%M MSK')}** — Завершено сохранение `spatial_identity_maps_v1_20261107`. После освобождения памяти автором последовательно пройдены валидатор каталога и 58 тестов (`runs/spatial_identity_maps_20261107/publication_checks.json`). Два прежних предупреждения о шаблонных путях сохранены. Граф сначала проверен на копии, затем обновлён; все 25 защищённых заметок гипотез побайтно сохранены, прежние файлы хранилища не удалялись. Научные расчёты не повторялись. Завершение собственных процессов и прежние технические отказы сохранены в `controller_verified.json`. Незавершённые правки другого чата в плане целей и его запись 20:43 не включаются в этот коммит; рабочие файлы сохраняются. Научный итог и критерии опыта не менялись.\n\n"
    write(RUN/'journal_publication_entry.json',dict(entry=entry),new=True)
    path=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';text=path.read_text(encoding='utf-8');anchor='## Журнал изменений\n\n'
    assert text.count(anchor)==1
    path.write_text(text.replace(anchor,anchor+entry,1),encoding='utf-8')
    sources={p.relative_to(ROOT).as_posix():sha(p) for p in (ROOT/'research_spatial_identity_20261107').iterdir() if p.is_file()}
    files={}
    for p in RUN.rglob('*'):
        if not p.is_file():continue
        rel=p.relative_to(RUN)
        if rel.parts[0]=='vault_check' or p.suffix=='.log' or p.name in ['status.json','final_checksums.json','final_checksums.sha256']:continue
        files[p.relative_to(ROOT).as_posix()]=sha(p)
    write(RUN/'final_checksums.json',dict(created=stamp.isoformat(),sources=sources,artifacts=files,
        scope='Includes local recovered arrays and saved correspondence artifacts; large arrays remain local and are not added to Git. Final live status and later Git receipts are outside this immutable manifest.'),new=True)
    with (RUN/'final_checksums.sha256').open('x',encoding='ascii') as f:f.write(sha(RUN/'final_checksums.json')+'  final_checksums.json\n')
    status('scientific_and_record_checks_completed',scientific_complete=True,publication_checks_passed=True,
        git_publication_pending=True,scientific_proceed=False)


def stage():
    verified_protocol();assert read(RUN/'controller_verified.json')['passed']
    assert subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()=='codex/system-integrity-20261106-e8f674ce'
    assert not subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT).strip(),'Index contains earlier staging'
    boundary=read(RUN/'working_tree_boundary.json')
    assert sha(ROOT/'dissertation/План целей по Трекеру.md')==boundary['preexisting_tracker_plan_sha256']
    plan=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';work=plan.read_bytes();decoded=work.decode('utf-8').replace('\r\n','\n')
    anchor='## Журнал изменений\n\n'
    start=decoded.index('**07.11.2026, 23:24 MSK**');end=decoded.index('\n\n',start)+2
    registration=decoded[start:end]
    entries=read(RUN/'journal_publication_entry.json')['entry']+read(RUN/'journal_entry.json')['entry']+registration
    head=subprocess.check_output(['git','show','HEAD:dissertation/ЕДИНЫЙ_ПЛАН.md'],cwd=ROOT).decode('utf-8')
    assert head.count(anchor)==1
    staged=head.replace(anchor,anchor+entries,1).encode('utf-8')
    marker='**07.11.2026, 20:43 MSK** — Решение автора: дальнейшие варианты ΣΠ не развивать;'
    assert marker in decoded and marker.encode('utf-8') not in staged
    blob=subprocess.check_output(['git','hash-object','-w','--stdin'],cwd=ROOT,input=staged).decode().strip()
    subprocess.run(['git','update-index','--cacheinfo',f'100644,{blob},dissertation/ЕДИНЫЙ_ПЛАН.md'],cwd=ROOT,check=True)
    selected={'experiments/project_history.json','experiments/system_problems.json','scripts/obsidian_fundamental_overview.py'}
    for p in (ROOT/'research_spatial_identity_20261107').iterdir():
        if p.is_file():selected.add(p.relative_to(ROOT).as_posix())
    for p in RUN.rglob('*'):
        if not p.is_file() or p.relative_to(RUN).parts[0]=='vault_check':continue
        if p.suffix in ['.json','.md','.sha256'] or p.name in ['.gitignore','.gitattributes']:
            selected.add(p.relative_to(ROOT).as_posix())
    changed=subprocess.check_output(['git','diff','--name-only','-z','--','ObsidianFlyVis'],cwd=ROOT).split(b'\0')
    selected.update(p.decode('utf-8') for p in changed if p)
    # Explicit files only; never recurse with force-add over the run directory.
    data=b''.join(p.encode('utf-8')+b'\0' for p in sorted(selected))
    subprocess.run(['git','add','-f','--pathspec-from-file=-','--pathspec-file-nul'],cwd=ROOT,input=data,check=True)
    assert plan.read_bytes()==work
    subprocess.run(['git','diff','--cached','--check'],cwd=ROOT,check=True)
    staged_names=subprocess.check_output(['git','diff','--cached','--name-only','-z'],cwd=ROOT).split(b'\0')
    staged_names=[p.decode('utf-8') for p in staged_names if p]
    assert all(not p.endswith('.npz') and 'vault_check/' not in p and p!='dissertation/План целей по Трекеру.md' for p in staged_names)
    write(RUN/'staging_receipt.json',dict(files=staged_names,working_plan_sha256=hashlib.sha256(work).hexdigest(),
        staged_plan_blob=blob,foreign_journal_entry_excluded=True,preexisting_tracker_plan_preserved=True),new=True)
    subprocess.run(['git','add','-f','--','runs/spatial_identity_maps_20261107/staging_receipt.json'],cwd=ROOT,check=True)
    print('Staged files:',len(staged_names)+1)


if __name__=='__main__':
    if sys.argv[1:] == ['--stage']:stage()
    elif sys.argv[1:] == ['--finish']:finish()
    else:raise SystemExit('Use --finish after checks, then --stage with Git index permission.')
