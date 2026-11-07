"""Register new authorization and first concrete bounded action."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.continuation_service_20261029 import RUN,service

def main():
    assert not RUN.exists(),'Do not duplicate an existing window'
    old=ROOT/'runs/continuation_20261028/final_stop_verified_20261029.json'
    assert service.read(old)['all_recorded_owned_actually_ended']
    assert not service.matching(9664,1793203064.7696502)
    assert not service.matching(20528,1793203064.802624)
    RUN.mkdir()
    for name in ['ready_jobs','jobs_done']:(RUN/name).mkdir()
    service.write(RUN/'registry.json',dict(processes=[]))
    service.write(RUN/'authorization.json',dict(user_request='Продолжаем, и когда будет ближайший результат?',
        created=service.now(),basis='Direct new human continuation after return; AGENTS9.1',
        hard_stop='2026-10-30T07:40:00+03:00',save_at='2026-10-30T07:30:00+03:00',controller_stop='2026-10-30T07:39:30+03:00',
        single_gpu=True,stage_time_budgets_cancelled=True,
        scope='Continue already authorized B2/B3 and MASA-Detic/MGOR/ScaleTrack; prioritize nearest valid full tracker result, preserve all completed work and failed gates',
        no_new_external_datasets=True,no_environment_changes=True,no_baseline_changes=True))
    folder=RUN/'masa_output_interface';folder.mkdir()
    script='scripts/check_masa_detic_output_interface_20261029.py'
    files=[script,'scripts/check_masa_detic_backbone_reference_recovery2_20261029.py','tracking/masa_detic_prepared.py','tracking/masa_detic_bdd_input.py']
    service.write(folder/'protocol.json',dict(budget_seconds=86400,frames=[0,30],normalized_vector_tolerance=1e-5,cosine_tolerance=1e-5,
        question='Does the actual normalized association-vector interface agree with an independent backbone on fixed real TRAIN frames?',
        reason='Old absolute raw-feature comparison remains failed; new downstream interface test declared before any new val tracking, no changing old acceptance',
        strongest_counterargument='Shared FPN/fusion/ROI/head is not an independent full MMDetection execution; this check isolates backbone and interface only',
        action_if_pass='Implement matched same-query tracker comparison with original detector and strong appearance control; declare HOTA protocol before val',
        action_if_fail='Preserve mismatch, localize output function; do not repeat arbitrary diagnostic variants or claim HOTA',
        sha256={f:service.sha(ROOT/f) for f in files},scientific_verdict=None))
    service.write(RUN/'ready_jobs/001_masa_output_interface.json',dict(name='001_masa_output_interface',folder=folder.relative_to(ROOT).as_posix(),script=script,
        script_sha256=service.sha(ROOT/script),protocol_sha256=service.sha(folder/'protocol.json')))
    (RUN/'HANDOFF.md').write_text('Новое прямое разрешение29.10 после возвращения автора, общий конец30.10 07:40МСК. Старую очередь не возобновлять. Первая001 проверяет реальные нормированные выходные признаки MASA-Detic, не повторяя Б1/матрицу/обучения и не объявляя старый FP32gate успешным. Протокол ДОпроверки. Послеpassed нужны реальное сопоставление и полный tracker/HOTA с согласованными контролями. Новые данные/пакеты/веса не нужны. Б2 численный аудит не принят, не запускать по инерции.\n',encoding='utf-8')
    print(RUN)

if __name__=='__main__':main()
