"""Validate an isolated vault copy and all mandatory checks, then apply only verified notes."""
import os,sys,subprocess,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];os.chdir(ROOT);sys.path.insert(0,str(ROOT))
from scripts.flyvis_two_frame_input_20261107 import OUT,read,write,sha

def snapshot(vault):
    return {str(p.relative_to(vault)):sha(p) for p in vault.rglob('*') if p.is_file()}

def main():
    assert read(OUT/'controller_verified.json')['all_owned_ended']
    from scripts.obsidian_fundamental_overview import build
    from scripts.audit_fundamental_obsidian import audit
    vault=Path('ObsidianFlyVis');before=snapshot(vault);copy=OUT/'vault_check_copy'
    assert not copy.exists()
    shutil.copytree(vault,copy)
    build(copy)
    after=snapshot(copy);removed=sorted(set(before)-set(after));added=sorted(set(after)-set(before))
    changed=[k for k in before if after.get(k)!=before[k]]
    expected={'Проверенные гипотезы — результаты и ограничения.md',str(Path('Опорные задачи')/'Задача 3 — Узнавать нужный объект и выбирать полезное исправление.md')}
    assert set(changed)==expected and not removed and not added,(changed,removed,added)
    report=audit(copy);assert report['note_count']==55 and not report['missing_targets']
    write(OUT/'graph_copy_verified.json',dict(passed=True,visible=report['note_count'],stored=len(after),changed=changed,added=added,removed=removed,all_other_files_unchanged=True))
    codes=[]
    for mode in ['validator','catalog','graph']:
        with (OUT/('final_'+mode+'.log')).open('xb') as log:
            code=subprocess.call([sys.executable,'-u','scripts/check_flyvis_two_frame_input_20261107.py',mode],stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
        codes.append(dict(stage=mode,code=code))
        if code:
            write(OUT/'finish_controller.json',dict(passed=False,steps=codes));return code
    assert snapshot(vault)==before,'Live vault changed while validating; preserve foreign edits.'
    for name in changed:
        shutil.copyfile(copy/name,vault/name)
    live=snapshot(vault);assert live==after
    checked=audit(vault);assert checked['note_count']==55 and not checked['missing_targets']
    write(OUT/'graph_live_verified.json',dict(passed=True,changed=changed,all_other_files_unchanged=True,visible=55,stored=len(live)))
    write(OUT/'finish_controller.json',dict(passed=True,steps=codes,graph_live_verified_sha256=sha(OUT/'graph_live_verified.json')))
    print('All final checks and preserving graph update passed',flush=True)
    return 0

if __name__=='__main__':raise SystemExit(main())
