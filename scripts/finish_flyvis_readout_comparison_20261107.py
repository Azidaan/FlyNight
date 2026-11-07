"""Check a preserving vault copy and final catalogue sequentially, then apply notes."""
import os,sys,time,subprocess,shutil
from pathlib import Path
import psutil
ROOT=Path(__file__).resolve().parents[1];os.chdir(ROOT);sys.path.insert(0,str(ROOT))
from scripts.flyvis_readout_comparison_20261107 import OUT,read,write,sha
def snapshot(vault):return {str(p.relative_to(vault)):sha(p) for p in vault.rglob('*') if p.is_file()}

def main():
    assert read(OUT/'controller_verified.json')['all_owned_ended']
    from scripts.obsidian_fundamental_overview import build
    from scripts.audit_fundamental_obsidian import audit
    vault=Path('ObsidianFlyVis');before=snapshot(vault);copy=OUT/'vault_check_copy'
    assert not copy.exists();shutil.copytree(vault,copy);build(copy)
    after=snapshot(copy);removed=sorted(set(before)-set(after));added=sorted(set(after)-set(before));changed=[k for k in before if after.get(k)!=before[k]]
    allowed={'Проверенные гипотезы — результаты и ограничения.md',str(Path('Опорные задачи')/'Задача 3 — Узнавать нужный объект и выбирать полезное исправление.md'),str(Path('Проблемы проекта')/'С4 — Измеренный резерв пока не превращён в большую прибавку.md')}
    assert set(changed)<=allowed and len(changed)>=2 and not removed and not added,(changed,removed,added)
    report=audit(copy);assert report['note_count']==55 and not report['missing_targets']
    write(OUT/'graph_copy_verified.json',dict(passed=True,visible=55,stored=len(after),changed=changed,added=added,removed=removed,all_other_files_unchanged=True))
    me=psutil.Process();owned={str(me.pid):me.create_time()};codes=[]
    for mode in ['validator','catalog','graph']:
        with (OUT/('final_'+mode+'.log')).open('xb') as log:
            child=subprocess.Popen([sys.executable,'-u','scripts/check_flyvis_readout_comparison_20261107.py',mode],stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW|subprocess.IDLE_PRIORITY_CLASS)
            owned[str(child.pid)]=psutil.Process(child.pid).create_time()
            while child.poll() is None:
                try:
                    for p in psutil.Process(child.pid).children(recursive=True):owned.setdefault(str(p.pid),p.create_time())
                except psutil.NoSuchProcess:pass
                write(OUT/'finish_status.json',dict(stage=mode,owned=owned));time.sleep(.3)
            code=child.wait()
        codes.append(dict(stage=mode,code=code))
        if code:
            write(OUT/'finish_controller.json',dict(passed=False,steps=codes,owned=owned));return code
    assert snapshot(vault)==before,'Preserve foreign edits to live vault.'
    for name in changed:shutil.copyfile(copy/name,vault/name)
    assert snapshot(vault)==after
    checked=audit(vault);assert checked['note_count']==55 and not checked['missing_targets']
    write(OUT/'graph_live_verified.json',dict(passed=True,visible=55,stored=len(after),changed=changed,all_other_files_unchanged=True))
    write(OUT/'finish_controller.json',dict(passed=True,steps=codes,owned=owned))
    print('All mandatory checks and preserving graph update passed',flush=True)
    return 0

if __name__=='__main__':raise SystemExit(main())
