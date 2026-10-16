"""Record read-only code review, time estimates, and the author-shortened window."""
from datetime import datetime
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tracking.z13_checkpoint import atomic_json

FULL = ROOT / 'runs/z13_triton_hota_v1_20261014/full'
PROBE = ROOT / 'runs/z13_cache_speed_v1_20261014'


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def main():
    now = datetime.now().astimezone()
    status = read(FULL / 'status.json')
    assert status['state'] == 'running'
    guard = read(FULL / 'deadline_guard_20261015_002318/started.json')
    assert guard['state'] == 'armed' and guard['deadline'] == '2026-10-15T07:40:00+03:00'
    report = read(PROBE / 'report.json')
    protocol = read(FULL / 'protocol.json')
    completed = {p.stem for p in (FULL / 'completed_jobs').glob('*.json')}
    rows = [read(p) for p in (FULL / 'B/signatures').glob('*/signatures.json')]
    b_per_frame = sum(r['seconds'] for r in rows) / sum(r['frames'] for r in rows)
    frames = {v: read(ROOT / 'runs/fly_signatures' / v / 'signatures.json')['frames']
              for v in protocol['videos']}
    old = read(ROOT / 'runs/z13_triton_hota_v1_20261014/report.json')
    d_low = old['video_comparisons']['D']['compiled_seconds']
    d_high = report['timings']['D']['triton']['processing_seconds']
    d_rates = (d_low / 1206, d_high / 1206)
    estimates = []
    for rate in d_rates:
        seconds = 0
        for arm in protocol['arms']:
            for video in protocol['videos']:
                if f'{arm}_sig_{video}' not in completed:
                    seconds += frames[video] * (b_per_frame if arm == 'B' else rate) + 8
        estimates.append(seconds)
    masa = read(ROOT / 'runs/z13_spatial_hota_v1_20261014/price.json')['masa_hours_if_all_videos_like_probe']
    remaining_hours = (datetime.fromisoformat(guard['deadline']) - now).total_seconds() / 3600
    audit = {
        'at': now.isoformat(), 'gpu_used_for_review': False,
        'completed_signature_jobs': len([s for s in completed if '_sig_' in s]),
        'remaining_window_hours': remaining_hours,
        'signature_remaining_hours_estimate': [s / 3600 for s in estimates],
        'masa_hours_one_video_extrapolation': masa,
        'estimate_limitations': 'C not measured in this full run; D rates from two single-video Triton measurements; whole active video counted; excludes final evaluation and possible stalls.',
        'code_findings': {
            'immutable_cache_has_version_and_gradient_checks': 'tracking/z13_cached_params_backend.py:get_params',
            'checks_cost_not_measured': True,
            'runtime_already_caches_dereferences_within_param_namespace': 'tools/flyvis-runtime/flyvis/utils/tensor_utils.py:AutoDeref.__getitem__',
            'namespace_rebuilt_per_backbone_step_not_every_integration_step': 'tracking/z13_readout.py:FlyBackbone.step',
            'timing_design_limitation': 'Fixed order, one repetition, B baseline reused from prior process after compiler failure; timing noise cannot establish stable marginal speed gain.',
            'correctness_error_found': False,
            'correctness_scope': 'Only measured sequence and this complete video; weights frozen; IDs and event totals exact; coordinates within 0.001 px.',
        },
        'resume': read(PROBE / 'resume_verified.json'),
        'deadline_guard': guard,
    }
    atomic_json(PROBE / 'code_review_and_window.json', audit)
    path = ROOT / 'experiments/project_history.json'
    catalog = read(path)
    card = next(c for c in catalog['experiments'] if c['id'] == 'z13_cached_params_speed_v1')
    previous = card['decision']
    card.setdefault('revisions', []).append({'date': now.strftime('%Y-%m-%d'), 'commit': '',
        'field': 'decision', 'old': previous, 'reason': 'Автор сократил окно работы до 07:40; прежний срок сохранён.'})
    card['decision'] = 'Исходный Triton HOTA возобновлён без нового кэша. Автор изменил крайний срок на 15.10 07:40 MSK; новый независимый ограничитель вооружён. По одному замеру устойчивое ускорение кэша не заявляется.'
    card['evidence'].append({'kind': 'json', 'ref': 'runs/z13_cache_speed_v1_20261014/code_review_and_window.json',
                             'claim': 'Чтение кода без GPU, оговорки измерения времени и новый срок'})
    hota = next(c for c in catalog['experiments'] if c['id'] == 'fly_spatial_hota_policy_z13_v1')
    hota['evidence'].append({'kind': 'json', 'ref': 'runs/z13_triton_hota_v1_20261014/full/deadline_revision_0740.json',
                             'claim': 'Автор сократил срок на 15.10 07:40; прежнее разрешение сохранено отдельно'})
    path.write_text(json.dumps(catalog, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    text_path = ROOT / 'dissertation/ЕДИНЫЙ_ПЛАН.md'
    text = text_path.read_text(encoding='utf-8')
    entry = f'''**{now:%d.%m.%Y, %H:%M} MSK** — возобновление HOTA и новый предел 07:40

*Получено.* После завершения технической пробы проверены выход её процесса, отсутствие прежнего собственного дерева HOTA и свободный мьютекс (`runs/z13_cache_speed_v1_20261014/resume_verified.json`). Исходный Triton HOTA возобновлён в 00:20 с сохранёнными 9 заданиями; веса, пороги, набор и код идентичны. Новый кэш в этот прогон не внедрён. Автор затем уточнил: «Цель закончить прогон в 7:40». Срок изменён на 15.10 07:40 MSK (`runs/z13_triton_hota_v1_20261014/full/full_authorization.json`), прежнее разрешение сохранено в `authorization_before_0740.json`. Независимый ограничитель `deadline_guard_20261015_002318/started.json` вооружён на новый срок; прежний собственный ограничитель остановлен после проверки PID и времени создания.

*Вывод по коду.* `runs/z13_cache_speed_v1_20261014/code_review_and_window.json`: ошибки вычислений в проверенных выходах не найдено. В кэше остаются проверки версий весов и запрета градиентов; их цена не измерена. Исходный AutoDeref уже кэширует разыменование внутри одного набора параметров, так что новая правка убирает повторную подготовку между шагами, а не всё суммирование по связям. Замер фиксированного порядка и одной пары не отделяет устойчивый эффект от колебаний. Это ограничение методики замера, не доказательство бесполезности кэша.

*Решение.* Дополнительные GPU-пробы не ставятся. Продолжается полный утверждённый HOTA до 07:40; набор не сокращается. При нехватке времени готовые видео сохраняются для продолжения, GPU останавливается. HOTA пока не получена, процент улучшения качества не заявляется.

'''
    text_path.write_text(text.replace('## Журнал изменений\n\n', '## Журнал изменений\n\n' + entry, 1), encoding='utf-8')
    print(json.dumps(audit, ensure_ascii=False))


if __name__ == '__main__':
    main()
