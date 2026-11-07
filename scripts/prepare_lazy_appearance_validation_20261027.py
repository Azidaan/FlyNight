"""Подготовить фиксированный сорокавидеопилот после полного dev-аудита."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.recovery_research_service_20261027 import RUN,read,write,sha,now

def prepare():
    assert read(RUN/'lazy_appearance_dev/independent_audit.json')['passed'];folder=RUN/'lazy_appearance_validation';assert not folder.exists()
    old=read(RUN/'learned_validation/protocol.json');frozen=read(RUN/'lazy_appearance_frozen.json');bound=['scripts/run_lazy_appearance_association_20261027.py','scripts/lazy_appearance_validation_20261027.py',
        'tracking/lazy_appearance_association.py','tracking/xywh_bytetrack.py','tracking/busca_runtime.py','tracking/recovery_admission_features.py','weights/busca_official_20261027/model_busca.pth','runs/recovery_research_20261027/lazy_appearance_frozen.json']
    for s in old['sources']:bound += [s[k] for k in ['baseline','alignment','detections','video_path','raw_array']]
    p=dict(parent_experiment='lazy_appearance_association_v1',created=now(),phase='val',budget_seconds=7200,author_stop=read(RUN/'authorization.json')['save_at'],sources=old['sources'],groups=old['groups'],
        config=read(RUN/'admission_dataset/protocol.json')['config'],frozen=frozen,cost_gate='Первое ПОЛНОЕ видео5трекеров:2*time*39+180<остаток7200с,иначестопбезпродления.',
        limitation='40историческихval,не новыйнезависимыйтест. ФиксированныевесаReID/weight,безпереобучениясети;3nullseedне3обучениянастоящейверсии. '
            'Толькопервое назначение изменено,но меняется дальнейшее состояние. Собственнаяадаптация,неполнаяBUSCA/DeepSORT/DeepOCSORT. '
            'Ядро/модульFlyVisнеиспользуются,основнойbaselineне меняется.',sha256={f:sha(ROOT/f) for f in set(bound)})
    write(folder/'protocol.json',p)
    script='scripts/lazy_appearance_validation_20261027.py';write(RUN/'ready_jobs/140_lazy_appearance_validation.json',dict(name='140_lazy_appearance_validation',folder=folder.relative_to(ROOT).as_posix(),script=script,script_sha256=sha(ROOT/script),protocol_sha256=sha(folder/'protocol.json')))
    audit='scripts/lazy_appearance_validation_audit_20261027.py';ap=dict(parent_experiment='lazy_appearance_association_v1',created=now(),budget_seconds=900,author_stop=read(RUN/'authorization.json')['save_at'],sha256={f:sha(ROOT/f) for f in [audit,'scripts/audit_lazy_appearance_association_20261027.py']});write(RUN/'lazy_appearance_validation_audit/protocol.json',ap)
    write(RUN/'ready_jobs/145_lazy_appearance_validation_audit.json',dict(name='145_lazy_appearance_validation_audit',folder='runs/recovery_research_20261027/lazy_appearance_validation_audit',script=audit,script_sha256=sha(ROOT/audit),protocol_sha256=sha(RUN/'lazy_appearance_validation_audit/protocol.json')))
    from scripts.b123_safe_journal_20261027 import journal
    journal('lazy_appearance_association_v1','вес внешнего вида зафиксирован наdev до40val',
        '*Сделано.* Полныйdevи независимое повторение всех назначений прошли. Источник lazy_appearance_dev/independent_audit.json. '
        'Выбран вес '+str(frozen['weight'])+' толькоdevHOTA,записанlazy_appearance_frozen.json. Теперьтотже40val:исходныйтрекер,настоящиепризнаки,3бессмысленныхконтроля. '
        'Никакогоподборапоval,критерийположительнойнижнейграницыкбазеивсем3nullсохранён;нулевойвыборнеуспехHOTA. '
        'Источник lazy_appearance_validation/protocol.json,первыйполныйвидеогейтсудвоениемстоимости,7200снепродлевать. Основнойbaselineне меняется.')

if __name__=='__main__':prepare()
