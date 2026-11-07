"""Детерминированные переходы после фиксированного обучения; решение о научной пользе остаётся агенту."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.recovery_research_service_20261027 import RUN,read,write,sha,now

def publish(name,folder,script,protocol):
    target=RUN/folder;assert not target.exists();write(target/'protocol.json',protocol)
    write(RUN/'ready_jobs'/(name+'.json'),dict(name=name,folder=target.relative_to(ROOT).as_posix(),script=script,script_sha256=sha(ROOT/script),protocol_sha256=sha(target/'protocol.json')))

def prepare(phase):
    learning=read(RUN/'pixel_box_learning/report.json');assert learning['complete']
    sources=[];bound=['scripts/evaluate_pixel_box_refinement_20261027.py','scripts/audit_pixel_box_evaluation_20261027.py','tracking/pixel_box_refinement.py',
                      'runs/recovery_research_20261027/pixel_box_dataset/protocol.json','runs/recovery_research_20261027/pixel_box_learning/report.json']
    for m in learning['models']:assert sha(RUN/m['checkpoint'])==m['sha256'];bound.append((RUN/m['checkpoint']).relative_to(ROOT).as_posix())
    if phase=='dev':
        done=read(RUN/'pixel_box_learning/controller_result.json');assert done['exit_code']==0 and done['all_owned_ended']
        for v in read(RUN/'pixel_box_dataset/protocol.json')['videos']['dev']:
            sources.append(dict(video=v,baseline=f'runs/recovery_research_20261027/admission_dataset/dev/{v}/baseline.json',alignment=f'runs/train_cache/{v}/alignment.json',video_path=f'data/bdd100k/videos/train/{v}.mov'))
        folder='pixel_box_dev';script='scripts/pixel_box_dev_20261027.py';job='100_pixel_box_dev';depends='pixel_box_learning';groups={}
    else:
        assert read(RUN/'pixel_box_dev/independent_audit.json')['passed']
        fixed=read(RUN/'pixel_box_frozen_for_validation.json');assert not fixed['val_used']
        bound.append('runs/recovery_research_20261027/pixel_box_frozen_for_validation.json')
        previous=read(RUN/'learned_validation/protocol.json')
        sources=[{k:s[k] for k in ['video','baseline','alignment','video_path','baseline_metrics']} for s in previous['sources']]
        folder='pixel_box_validation';script='scripts/pixel_box_validation_20261027.py';job='110_pixel_box_validation';depends='pixel_box_dev_audit';groups=previous['groups']
    for source in sources:bound += [source[k] for k in ['baseline','alignment','video_path']]
    bound.append(script)
    p=dict(parent_experiment='pixel_box_refinement_v1',created=now(),phase=phase,budget_seconds=3600,author_stop=read(RUN/'authorization.json')['save_at'],
        sources=sources,depends_on=depends,groups=groups,alpha_grid=[0.,.5,1.] if phase=='dev' else None,
        action='Меняются только координаты по причинным областям. Все шесть моделей фиксированы до выбора силы; сила выбирается толькоdev из[0,0.5,1],приравенствеменьшая. '
               'Идентификаторы, классы, баллы, количество рамок и исходные назначения неизменны. Альфа0точно равна базе. GTтолько уоценщика,не дляROI/прошлого/входа.',
        acceptance='Полные35devдля выбора силы,затем40val безподгона; все3сида,нулевойконтроль,положительнаянижняяграница к базе во всех3,добавкавремени отдельно к current во всех3.',
        cost_gate='ПервоеПОЛНОЕвидео:2*time*(N-1)+120<остаток3600с,иначестопбезпродления.',
        limitation='Собственная поправка,неполныеIoU-Net/Cascade/Б1–Б3. Историческийval,не новыйнезависимыйтест. Пакетныйпричинныйрасчёт не являетсязамеромреальной30Гцзадержки. '
                   'Детектор виделtrain/dev-видео;devэто выборполитики,не проверкаобобщениядетектора. Ядро/полныймодульFlyVisнеиспользуются.',
        sha256={s:sha(ROOT/s) for s in set(bound)})
    publish(job,folder,script,p)
    audit_script='scripts/pixel_box_'+('dev' if phase=='dev' else 'validation')+'_audit_20261027.py'
    audit_folder=folder+'_audit';audit_job='105_pixel_box_dev_audit' if phase=='dev' else '115_pixel_box_validation_audit'
    publish(audit_job,audit_folder,audit_script,dict(parent_experiment='pixel_box_refinement_v1',created=now(),budget_seconds=600,
        author_stop=read(RUN/'authorization.json')['save_at'],phase=phase,
        sha256={s:sha(ROOT/s) for s in [audit_script,'scripts/audit_pixel_box_evaluation_20261027.py','scripts/pixel_box_evaluation_schedule_20261027.py']}))
    from scripts.b123_safe_journal_20261027 import journal
    journal('pixel_box_refinement_v1','подготовлена полная '+phase+'-оценка геометрии с фиксированными весами',
        '*Сделано.* '+p['action']+' '+p['cost_gate']+' '+p['limitation']+' Источник '+(RUN/folder/'protocol.json').relative_to(ROOT).as_posix()+
        '. Выходы/CPU-повтор/интервалы проверит отдельный аудит. Это постановка следующего этапа,не измеренная HOTA. Готовые этапы не повторяются.')

def prepare_dev():prepare('dev')
def prepare_val():prepare('val')

if __name__=='__main__':prepare_dev()
