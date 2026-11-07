"""Готовые переходы после профиля и dev-аудита, без ожидания модели и новых бюджетов."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.recovery_research_service_20261027 import RUN,read,write,sha,matching,now
from scripts.b123_safe_journal_20261027 import journal

def ended(folder):
    c=read(RUN/folder/'controller_result.json');assert c['exit_code']==0 and c['all_owned_ended'] and not any(matching(k,v) for k,v in c['owned'].items())

def publish(name,sub,script):
    p=RUN/'ready_jobs'/(name+'.json');assert not p.exists()
    write(p,dict(name=name,folder=(RUN/sub).relative_to(ROOT).as_posix(),script=script,script_sha256=sha(ROOT/script),protocol_sha256=sha(RUN/sub/'protocol.json')))

def protocol(phase):
    dev=phase=='dev';parent='lazy_appearance_dev' if dev else 'lazy_appearance_validation';old=read(RUN/parent/'protocol.json')
    folder=RUN/('frame_masa_'+phase);assert not folder.exists()
    bound=['tracking/frame_masa_appearance.py','tracking/masa_r50.py','tracking/lazy_appearance_association.py','tracking/xywh_bytetrack.py',
        'scripts/run_frame_masa_association_20261027.py','scripts/frame_masa_'+phase+'_20261027.py','weights/masa/masa_r50.pth',
        'runs/recovery_research_20261027/frame_masa_profile/report.json','runs/recovery_research_20261027/frame_masa_profile/protocol.json']
    if not dev:bound+=['runs/recovery_research_20261027/frame_masa_frozen.json','runs/recovery_research_20261027/lazy_appearance_validation/report.json']
    for s in old['sources']:bound += [s[k] for k in ['baseline','alignment','detections','video_path']]+([s['raw_array']] if 'raw_array' in s else [])
    p=dict(parent_experiment='frame_masa_association_v1',created=now(),phase='dev' if dev else 'val',budget_seconds=7200,
        author_stop=read(RUN/'authorization.json')['save_at'],sources=old['sources'],config=old['config'],
        cost_gate='Первое ПОЛНОЕ видео:2*time*(N-1)+180<остаток7200с,иначестопбезпродления. Первое готовое не повторяется.',
        feature_regime='СуществующийMASA-R50,полныйтекущийкадр1024,FP32,BNзаморожена,ROIвекторы256,L2. Реальныевекторывсехнаблюдений сохраняются до перехода кследующему кадру;никакихбудущихизображений.',
        limitation=read(RUN/'frame_masa_profile/protocol.json')['limitation'],sha256={f:sha(ROOT/f) for f in set(bound)})
    if dev:p.update(weights_grid=[0.,.1,.3],ambiguity_margin=.15)
    else:p.update(groups=old['groups'],frozen=read(RUN/'frame_masa_frozen.json'),previous_reid='runs/recovery_research_20261027/lazy_appearance_validation/report.json')
    write(folder/'protocol.json',p)
    audit=RUN/('frame_masa_'+phase+'_audit');write(audit/'protocol.json',dict(parent_experiment=p['parent_experiment'],budget_seconds=900,author_stop=p['author_stop']))

def prepare_dev():
    ended('frame_masa_profile');r=read(RUN/'frame_masa_profile/report.json');assert r['passed'] and r['strict_past'] and r['batchnorm_unchanged'] and r['frames']==120
    protocol('dev')
    publish('165_frame_masa_dev','frame_masa_dev','scripts/frame_masa_dev_20261027.py')
    publish('170_frame_masa_dev_audit','frame_masa_dev_audit','scripts/frame_masa_dev_audit_20261027.py')
    journal('frame_masa_association_v1','после технического профиля подготовлены полные35dev','Профиль frame_masa_profile/report.json прошёл,дерево фактически завершено. Это не HOTA и не цена полной серии. '
        'В существующуюочередьподготовлены165/170:35dev,таже сетка[0,.1,.3],7200сGPUи900сCPUаудит. Первое ПОЛНОЕ видео с3реальнымитрекерамидаётфактическуюцену,2*time*34+180должнопоместитьсявостаток. '
        'Приотрицательнойценеостановкабезпродления,неподменятьнеполныйитоготрицательнойнаукой. Веса/BN/детекции/порогине меняются. Готовые35старогоReIDнеповторяются.')

def prepare_validation():
    assert read(RUN/'frame_masa_dev/independent_audit.json')['passed'];ended('frame_masa_dev')
    # Аудит публикует переход своим последним действием; очередь начнёт его
    # только после фактического конца этого аудитора.
    protocol('validation')
    publish('175_frame_masa_validation','frame_masa_validation','scripts/frame_masa_validation_20261027.py')
    publish('180_frame_masa_validation_audit','frame_masa_validation_audit','scripts/frame_masa_validation_audit_20261027.py')
    journal('frame_masa_association_v1','MASA-сила зафиксирована наdev,подготовлены40val','Источник frame_masa_frozen.json;полный35devаудитpassed. '
        '175/180:теже40историческихval,исходныйтрекер/MASA3null256;прежнийReIDизготовогоотчётабезповторногоинференса. Сравнениякбазе,прежнемуReIDивсем3null. '
        'ПервоеПОЛНОЕвидео2*time*39+180<остаток7200с;отдельные900сCPUаудит. Параметрыпоvalне менять,выбор0неуспех. Историческийvalненовыйнезависимыйтест,ядроFlyVisнеиспользуется.')

if __name__=='__main__':prepare_dev()
