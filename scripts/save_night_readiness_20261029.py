"""Preserve exact B2 limitations and truthful candidate readiness before save."""
from pathlib import Path
import sys,subprocess,shutil
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.continuation_service_20261028 import RUN,service

def main():
    folder=RUN/'night_readiness_save'
    for f,h in service.read(folder/'protocol.json')['sha256'].items():assert service.sha(ROOT/f)==h,f
    catalog=ROOT/'experiments/project_history.json'
    shutil.copy2(catalog,folder/'project_history_before.json')
    d=service.read(catalog)
    def card(cid):return next(c for c in d['experiments'] if c['id']==cid)
    prior={
        'b2_causal960_path_v1':service.read(RUN/'b2_causal960_profile/protocol.json')['limitations'],
        'b2_sparse_training_pilot_v1':'Это не обучение на всех20441клипах и не научная проверка HOTA;нетпользыядрапобезконтрольнойloss.'}
    for cid,text in prior.items():
        c=card(cid)
        if not any(v.get('field')=='cannot_claim' and v.get('old')==text for v in c.get('revisions',[])):
            c.setdefault('revisions',[]).append(dict(date='2026-10-29',commit=None,field='cannot_claim',old=text,
                reason='Восстановлена точная прежняя формулировка из зарегистрированного протокола/скрипта; ранний recorder не сохранил revisions. Научный исход не изменён.'))
    b=card('b2_sparse_training_pilot_v1')
    old=b['cannot_claim']
    suffix=' Независимый полный повтор1200проверочных потерь пока не завершён: b2_sparse_training_audit_recovery3/controller_result.json и b2_training_gpu_predictor_audit/controller_result.json сохраняют отказы; b2_training_semantics_audit/controller_result.json также exit1. В третьем dev-видео на GPUразницаD5,531311e-5превысила прежний3e-5. Это препятствие приёмке исполнения, не отрицательная HOTA и не установленный вред ядра. Готовое обучение сохранено, проверку полного детектора/трекера не запускали.'
    assert suffix not in old
    b.setdefault('revisions',[]).append(dict(date='2026-10-29',commit=None,field='cannot_claim',old=old,
        reason='Добавлен точный текущий предел приёмки после сохранённых CPU/GPUаудиторских отказов; старый результат обучения остаётся infrastructure.'))
    b['cannot_claim']=old+suffix
    for name in ['b2_sparse_training_audit_recovery3','b2_training_gpu_predictor_audit','b2_training_semantics_audit']:
        b.setdefault('evidence',[]).append(dict(kind='file',ref=f'runs/continuation_20261028/{name}/controller_result.json',claim='Неуспешная независимая проверка сохранена, полного допуска к HOTA нет'))
    m=card('reid_masa_detic_v1');old=m['cannot_claim']
    m.setdefault('revisions',[]).append(dict(date='2026-10-29',commit=None,field='cannot_claim',old=old,
        reason='Уточнена фактическая техническая готовность после загрузки весов, GPUпробы и неполного численного сравнения; научная проверка остаётся planned.'))
    m['cannot_claim']=old+' Веса и отдельный путь подготовлены; masa_detic_gpu_input_profile/report.json: один реальный train-кадр/7областей/256признаков,0,719сGPU,неHOTA. Общая численная эквивалентность ещё не принята: masa_detic_backbone_reference_recovery2/status.json превышает1e-4вFP32; float64подтвердил один кратный32вход, второй некратный вход имеет разное дополнение у двух реализаций. Техническая исправность и частное структурное совпадение не закрывают критерии научного опыта.'
    m.setdefault('evidence',[]).append(dict(kind='file',ref='runs/continuation_20261028/masa_detic_gpu_input_profile/report.json',claim='Только фактическая техническая GPUпроба; научныйвердиктnull'))
    service.write(catalog,d)
    from scripts.b123_safe_journal_20261027 import journal
    journal('night_readiness_save','Уточнена сохранённая готовность Б2 и новых признаков',
        'b2_sparse_training_pilot_v1 остаётся infrastructure: обучение завершено, но полный независимый численный аудит не принят, HOTAнеизмерена. '
        'Ранее утраченные cannot_claim восстановлены в revisions по точным источникам двух инфраструктурных карточек. '
        'reid_masa_detic_v1 остаётся planned/красной: успешная отдельная GPUпроба одного реального кадра не является научной проверкой. '
        'Первичные контроллеры/исходники/ошибки не переписаны; старые формулировки сохранены в revisions. '
        'Новых вычислительных научных запусков нет, общийsave07:30/stop07:40сохранён.')
    service.write(folder/'status.json',dict(stage='Validating saved catalogue and graph',updated=service.now()))
    commands=[['scripts/validate_project_history.py'],['-m','pytest','tests/test_project_history.py','-q'],
        ['scripts/restore_obsidian_sections_20261027.py'],['-m','pytest','tests/test_obsidian_fundamental_overview.py','-q']]
    for i,args in enumerate(commands):
        with (folder/f'check_{i}.log').open('w',encoding='utf-8') as out:
            code=subprocess.run([sys.executable,*args],cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,timeout=420).returncode
        assert code==0,(i,code)
    service.write(folder/'report.json',dict(complete=True,checks_passed=True,scientific_verdict=None,
        training_state_unchanged='infrastructure',new_candidates_state_unchanged='planned',old_limitations_preserved_in_revisions=True,
        catalog_sha256=service.sha(catalog),updated=service.now()))
    with (RUN/'HANDOFF.md').open('a',encoding='utf-8') as f:
        f.write('\n\n29.10 около06:00МСК: night_readiness_save/report.json и check_0..3.log фиксируют сохранение честной готовности Б2/Т10. Две утраченные прежние cannot_claim восстановлены точно в revisions; численные отказы не устранены и не ослаблены, HOTAБ2/Т10нет. Каталог/58tests/граф/3tests проверены только по фактическому report/controller. После save новых научных заданий нет; проверить deadline_result/все PID+birth к07:40, затем удалить ЭТУавтоматизацию1.\n')

if __name__=='__main__':main()
