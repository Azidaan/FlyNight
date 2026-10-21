"""Bounded technical completion using the unused active-compute probe allowance."""
import gc
import sys
from datetime import datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import z13_center_hota_recovery1_20261021 as recovery
science=recovery.science;service=recovery.service;RUN=science.RUN
CONTRACT=RUN/'completion_contract.json'


def prepare():
    if CONTRACT.exists():raise RuntimeError('Existing completion contract')
    old=science.read(RUN/'probe_controller_result.json');recent=science.read(RUN/'probe_recovery1_controller_result.json')
    assert old['worker_ended'] and recent['worker_ended']
    used=sum((datetime.fromisoformat(end['ended_at'])-datetime.fromisoformat(science.read(RUN/name)['started_at'])).total_seconds() for name,end in [('probe_failed_controller.json',old),('controller.json',recent)])
    remaining=int(science.read(RUN/'protocol.json')['probe_budget_seconds']-used-25)
    assert remaining>0 and not (RUN/'technical.json').exists()
    repro=science.read(RUN/'recovery1_reproduction.json');assert set(repro)=={'C','D','R1'}
    assert all(x['max_center_px']<.5 and x['max_score']<.001 and x['same_threshold_decisions'] for x in repro.values())
    files=['scripts/z13_center_hota_completion_20261021.py','runs/z13_center_hota_20261021/recovery1_contract.json','runs/z13_center_hota_20261021/recovery1_reproduction.json','runs/z13_center_hota_20261021/probe_controller_result.json','runs/z13_center_hota_20261021/probe_recovery1_controller_result.json']
    video=science.read(RUN/'protocol.json')['probe_video']
    files.extend(f'runs/z13_center_hota_20261021/probe/R1/{video}/signatures{suffix}' for suffix in ('.json','.npz'))
    science.write(CONTRACT,dict(created=science.now().isoformat(),previous_active_seconds=used,budget_seconds=remaining,total_active_allowance_seconds=1200,sha256={n:science.sha(ROOT/n) for n in files},reason='Original recovery wall deadline retained and old process ended. Separate completion uses only unused active-compute allowance within author night window; scientific settings unchanged, completed arms/reproduction reused. Earlier timeouts/failures retained.'))
    science.journal('сохранение ограниченной пробы и остаток технического бюджета','*Получено.* Прежняя ошибка и отдельное окончание восстановления сохранены в probe_controller_result.json и probe_recovery1_controller_result.json. Проверка recovery1_reproduction.json уже пройдена для всех3вариантов; полные C/D/noop сохранены. Исходное окно восстановления не продлевалось.\n\n*Решение.* Отдельная техническая завершающая стадия использует только остаток1200с фактической работы процессов: completion_contract.json → previous_active_seconds='+str(used)+', budget_seconds='+str(remaining)+'. Суммарная работа и20с завершения укладываются в1200с; общий срок автора07:40 неизменен. Завершённые видео и проверка эпизодов не повторяются; считается только незавершённый R1, общая MASA и проверка сшивки35. Это служебное завершение, научные критерии/данные/порог не менялись; HOTA ещё нет.')
    print(remaining)


def guard(p):
    for rel,digest in science.read(CONTRACT)['sha256'].items():assert science.sha(ROOT/rel)==digest,rel
    if (RUN/'technical.json').exists() and ('--full-control' in sys.argv or '--stage' in sys.argv):assert science.read(RUN/'technical.json')['execution_completion_contract_sha256']==science.sha(CONTRACT)
    return recovery.guard(p)


def probe():
    import numpy as np
    import torch
    from tracking.gpu_lock import GpuLock
    torch.set_num_threads(2);p=science.read(RUN/'protocol.json');science.verify(p);repro=science.read(RUN/'recovery1_reproduction.json');video=p['probe_video'];timing={}
    with GpuLock('flyvis_fovea_gpu'),torch.no_grad():
        with np.load(RUN/'probe/noop'/video/'signatures.npz') as a,np.load(ROOT/'runs/fly_signatures'/video/'signatures.npz') as b:assert np.array_equal(a['outputs'],b['outputs']) and int(a['frame_count'])==int(b['frame_count'])
        timing['noop_seconds']=science.read(RUN/'probe/noop'/video/'signatures.json')['seconds']
        for arm in p['arms']:
            path=RUN/'probe'/arm/video/'signatures.json';reused=path.exists()
            if not reused:science.status('Завершение незавершённого полного видео',variant=arm,done=2,total=3);science.build(video,arm,RUN/'probe'/arm)
            meta=science.read(path)
            with np.load(path.with_suffix('.npz')) as actual,np.load(ROOT/'runs/fly_signatures'/video/'signatures.npz') as baseline:assert int(actual['frame_count'])==int(baseline['frame_count']) and np.isfinite(actual['outputs']).all()
            timing[arm]=dict(seconds=meta['seconds'],events=meta['z13_events'],reused_saved=reused)
        from tracking.masa_r50 import build as masa_model
        from scripts.z13_shared_masa import extract_video
        science.status('Техническое завершение общей MASA',done=0,total=1);model=masa_model()
        timing['masa']=extract_video(model,ROOT/'data/bdd100k/videos/train'/(video+'.mov'),{a:RUN/'probe'/a/video/'signatures.npz' for a in p['arms']},{a:RUN/'probe/masa'/a/(video+'.npz') for a in p['arms']},check_first_frame=True)
        del model;gc.collect();torch.cuda.empty_cache()
    science.status('Техническое завершение сверки сшивки35',done=0,total=35);science.environment_head();science.verify(p)
    science.write(RUN/'technical.json',dict(passed=True,reproduction=repro,no_intervention_outputs_bitwise_equal=True,baseline_stitch35_equal=True,timing=timing,rough_full_seconds=35*(sum(timing[a]['seconds'] for a in p['arms'])+timing['masa']['seconds']),caveat='One complete train video; saved inference times reused, initialization excluded. No HOTA; partial queue expected.',execution_service_contract_sha256=science.sha(RUN/'service_contract.json'),recovery_execution_contract_sha256=science.sha(RUN/'recovery1_contract.json'),execution_completion_contract_sha256=science.sha(CONTRACT),initial_whole_corpus_verified=service._probe_verified))


def main():
    if '--prepare-completion' in sys.argv:prepare();return
    import tracking.z13_ridge_localizer as original
    from tracking.z13_ridge_localizer_v2 import RecoveredCenterRidgeLocalizer
    original.CenterRidgeLocalizer=RecoveredCenterRidgeLocalizer;science.verify=guard;science.probe=probe
    if '--full-control' in sys.argv:
        source=(ROOT/'scripts/z13_center_hota_controller_20261021.py').read_text(encoding='utf-8');old="str(ROOT/'scripts/z13_center_hota_20261021.py')";assert source.count(old)==1
        source=source.replace(old,"str(ROOT/'scripts/z13_center_hota_completion_20261021.py')");assert source.count("RUN/'probe_controller_result.json'")==2
        source=source.replace("RUN/'probe_controller_result.json'","RUN/'probe_completion_controller_result.json'")
        scope=dict(__name__='completed_center_hota_controller',__file__=str(ROOT/'scripts/z13_center_hota_controller_20261021.py'));exec(compile(source,'completed_center_hota_controller','exec'),scope);scope['main']();return
    science.main()


if __name__=='__main__':main()
