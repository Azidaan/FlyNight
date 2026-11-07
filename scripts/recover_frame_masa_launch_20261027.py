"""Исправить только запуск через scripts; старые источники/отказы и окно165 сохранить."""
from pathlib import Path
from datetime import datetime
import sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.recovery_research_service_20261027 import RUN,read,write,sha,matching,now
from scripts.b123_safe_journal_20261027 import journal

HEADER="from pathlib import Path\nimport sys\nROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))\n"

def make(name,body):
    p=ROOT/'scripts'/name;assert not p.exists();p.write_text(HEADER+body,encoding='utf-8');return p

def main():
    for sub in ['frame_masa_dev','frame_masa_dev_audit']:
        c=read(RUN/sub/'controller_result.json');assert c['exit_code']==1 and c['all_owned_ended'] and not any(matching(k,v) for k,v in c['owned'].items())
        assert 'ModuleNotFoundError' in (RUN/sub/'stderr.log').read_text(encoding='utf-8')
        write(RUN/sub/'status_before_recovery1.json',read(RUN/sub/'status.json'))
    end=datetime.fromisoformat(read(RUN/'frame_masa_dev/execution_deadline.json')['deadline']).timestamp();remaining=end-time.time()
    assert remaining>1200;budget=int(remaining-960)
    dev=make('frame_masa_dev_recovery1_20261027.py',"from scripts import run_frame_masa_association_20261027 as worker\nfrom scripts.recovery_research_service_20261027 import RUN,read\noriginal=worker.read\ndef recover(path):\n    if Path(path)==RUN/'frame_masa_dev/execution_deadline.json':return read(RUN/'frame_masa_dev/recovery1/execution_deadline.json')\n    return original(path)\nworker.read=recover\nif __name__=='__main__':worker.run('frame_masa_dev')\n")
    val=make('frame_masa_validation_recovery1_20261027.py',"from scripts import run_frame_masa_association_20261027 as worker\nfrom scripts.recovery_research_service_20261027 import RUN,read\noriginal=worker.read\ndef recover(path):\n    if Path(path)==RUN/'frame_masa_dev_audit/controller_result.json':return read(RUN/'frame_masa_dev_audit/recovery1/controller_result.json')\n    return original(path)\nworker.read=recover\nif __name__=='__main__':worker.run('frame_masa_validation')\n")
    va=make('frame_masa_validation_audit_recovery1_20261027.py',"from scripts.audit_frame_masa_association_20261027 import audit\nif __name__=='__main__':audit('frame_masa_validation')\n")
    audit=make('frame_masa_dev_audit_recovery1_20261027.py',"from scripts import audit_frame_masa_association_20261027 as worker\nfrom scripts import prepare_frame_masa_chain_20261027 as preparation\nfrom scripts.recovery_research_service_20261027 import RUN,read,write,sha\noriginal=worker.read\ndef recover(path):\n    if Path(path)==RUN/'frame_masa_dev/controller_result.json':return read(RUN/'frame_masa_dev/recovery1/controller_result.json')\n    return original(path)\nworker.read=recover\npreparation.read=recover\n# Конец исправленногоdevстрогопроверенфункциейauditдоэтогоперехода.\nbase_publish=preparation.publish\ndef publish(name,folder,script):\n    changed={'scripts/frame_masa_validation_20261027.py':'scripts/frame_masa_validation_recovery1_20261027.py','scripts/frame_masa_validation_audit_20261027.py':'scripts/frame_masa_validation_audit_recovery1_20261027.py'}\n    return base_publish(name,folder,changed.get(script,script))\npreparation.publish=publish\nif __name__=='__main__':\n    worker.audit('frame_masa_dev')\n    preparation.prepare_validation()\n")
    # Никакой CUDA при проверке адресации модулей: функции run/audit не вызываются.
    import runpy
    for script in [dev,val,va,audit]:runpy.run_path(str(script),run_name='prelaunch_import_check')
    # runpy меняет модули только этого короткого служебного процесса, не расчёт.
    for sub,seconds in [('frame_masa_dev/recovery1',budget),('frame_masa_dev_audit/recovery1',900)]:
        write(RUN/sub/'protocol.json',dict(parent_experiment='frame_masa_association_v1',budget_seconds=seconds,author_stop=read(RUN/'authorization.json')['save_at'],
            original_absolute_end=datetime.fromtimestamp(end).astimezone().isoformat(),remaining_165_window_seconds=remaining,reserved_audit_seconds=900,reserved_seconds=60))
    jobs=[]
    for name,sub,script in [('166_frame_masa_dev_recovery1','frame_masa_dev/recovery1',dev),('171_frame_masa_dev_audit_recovery1','frame_masa_dev_audit/recovery1',audit)]:
        job=dict(name=name,folder=(RUN/sub).relative_to(ROOT).as_posix(),script=script.relative_to(ROOT).as_posix(),script_sha256=sha(script),protocol_sha256=sha(RUN/sub/'protocol.json'));jobs.append(job)
    proof=dict(created=now(),cause='У запускаемых scripts-обёрток не было корня проекта в sys.path;остановка доGPU/кадров',original_165_end=end,
        remaining_seconds=remaining,recovery_dev_budget=budget,recovery_audit_budget=900,reserve_seconds=60,old_170_not_resumed=True,
        scientific_sources_unchanged=True,all_four_imports_checked=True,sha256={s.relative_to(ROOT).as_posix():sha(s) for s in [dev,val,va,audit]},jobs=jobs)
    write(RUN/'frame_masa_launch_recovery1.json',proof)
    journal('frame_masa_association_v1','исправлена адресация запуска до первого полного видео',
        '160профильпрошёл:120кадров/1480областей/119событий,415из415ключей,13,516с/733830656байт,строгоепрошлое/finite/BNнеизменны;это неHOTA. '
        '165/170завершилисьнастоящимкодом1/all_owned_ended:вновыхтонкихобёрткахнебылокорняпроектавsys.path. Научныйrunner/признаки/детекции/выходыне менялись;полныхвидеонет. '
        'Отказы/замороженныеисточники/контроллерысохранены. Четыреотдельныеобёрткиrecovery1провереныполнымимпортомбезвызоваGPU. '
        'Времядля166devи171CPUаудитвыделеноВНУТРИпрежнегоабсолютного165окнадо'+str(end)+'UTCtimestamp:dev'+str(budget)+'с/аудит900с/резерв60с. '
        'Старый170непродлеваетсяинеповторяется;это разделение оставшегося165бюджета,не новый7200с. Источник frame_masa_launch_recovery1.json. '
        'Основнойbaseline/критерии/одинGPU/срок07:40не меняются,старыйlockнетронут.')
    for job in jobs:
        target=RUN/'ready_jobs'/(job['name']+'.json');assert not target.exists();write(target,job)
    print('Published bounded import recovery inside original165 window:',budget,'+900+60')

if __name__=='__main__':main()
