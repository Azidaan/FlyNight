"""Independent records-only completion; scientific verdict uses original fixed gates."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_center_hota_20261021 import RUN,read,write,sha,now
CONTRACT=RUN/'resume1_finalization_contract.json'
NOTE='ObsidianFlyVis/09 Запланировано/z13_center_hota_v1.md'


def seal():
    if CONTRACT.exists():raise RuntimeError('Existing finalization contract')
    files=['scripts/z13_center_hota_finalize_20261022.py','scripts/z13_center_hota_morning_save_20261021.py','scripts/z13_record_center_hota_20261021.py','scripts/z13_center_hota_progress_20261021.ps1']
    write(CONTRACT,dict(created=now().isoformat(),authorization_sha256=sha(RUN/'resume1_authorization.json'),sha256={n:sha(ROOT/n) for n in files},owned_note=NOTE,owned_note_sha256=sha(ROOT/NOTE),scope='Record/check/graph only after verified end of own resumed controller. No science jobs, no Git mutation; agent reviews and publishes concrete own files.'))


def render():
    source=(ROOT/'scripts/z13_center_hota_morning_save_20261021.py').read_text(encoding='utf-8')
    cutoff='        # Exact own paths only.';assert source.count(cutoff)==1
    source=source.split(cutoff)[0]
    source+="""        write(marker,dict(state='records_ready_for_agent_review',at=now().isoformat(),completed_jobs=result['completed_jobs'],remaining=len(result['remaining']),compute_ended=True))
    except Exception as exc:
        write(RUN/'morning_save_error.json',dict(at=now().isoformat(),error=str(exc)));raise
    finally:lock.close()

if __name__=='__main__':main()
"""
    mapping={'full_controller_result.json':'resume1_controller_result.json','full_controller.json':'resume1_controller.json','morning_save_result.json':'resume1_finalize_result.json','morning_save_launch.json':'resume1_finalize_launch.json','morning_save_attention.json':'resume1_finalize_attention.json','morning_save_error.json':'resume1_finalize_error.json','morning_save_audit.json':'resume1_finalize_audit.json','morning_save.log':'resume1_finalize.log','morning_save.lock':'resume1_finalize.lock','finish_recorded.json':'resume1_finish_recorded.json','scripts/z13_record_center_hota_20261021.py':'scripts/z13_record_center_hota_resume_20261022.py','2026-10-21T07:35:00':'2026-10-22T07:35:00','2026-10-21T07:39:55':'2026-10-22T07:39:55'}
    for old,new in mapping.items():assert old in source,old;source=source.replace(old,new)
    old="        run([PY,'scripts/z13_build_vault_preserving_edits.py','--report','runs/z13_center_hota_20261021/vault_morning_save.json'])"
    assert old in source
    new="""        own=read(RUN/'resume1_finalization_contract.json');note=ROOT/own['owned_note'];refresh={}
        if note.exists() and sha(note)==own['owned_note_sha256']:refresh[own['owned_note']]=own['owned_note_sha256']
        write(RUN/'resume1_owned_refresh.json',refresh)
        run([PY,'scripts/z13_build_vault_preserving_edits.py','--report','runs/z13_center_hota_20261021/resume1_vault_finish.json','--refresh-owned','runs/z13_center_hota_20261021/resume1_owned_refresh.json'])"""
    source=source.replace(old,new)
    assert "run(['git'" not in source,'Records service must not publish unchecked results'
    return source


def main():
    if '--seal' in sys.argv:seal();return
    contract=read(CONTRACT);assert sha(RUN/'resume1_authorization.json')==contract['authorization_sha256']
    for name,digest in contract['sha256'].items():assert sha(ROOT/name)==digest,name
    scope=dict(__name__='__main__',__file__=str(ROOT/'scripts/z13_center_hota_morning_save_20261021.py'))
    exec(compile(render(),'resumed_center_hota_finalize','exec'),scope)


if __name__=='__main__':main()
