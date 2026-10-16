"""Conditionally migrate fixed HOTA after strict shared-MASA parity and speed gates."""
from datetime import datetime
from pathlib import Path
import ctypes
import hashlib
import json
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.z13_hota_deadline_guard import WindowsProcesses
from scripts.z13_shared_hota_runner import scope
from tracking.z13_checkpoint import atomic_json

SOURCE = ROOT / 'runs/z13_triton_hota_v1_20261014/full'
PROBE = ROOT / 'runs/z13_shared_masa_probe_v1_20261015'
RUN = ROOT / 'runs/z13_shared_masa_hota_v1_20261015/full'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def copy_checked(source, target):
    source.resolve().relative_to(ROOT)
    target.resolve().relative_to(ROOT)
    assert not target.exists()
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    assert digest(source) == digest(target)


def main():
    now = datetime.now().astimezone()
    report = read(PROBE / 'report.json')
    assert read(PROBE / 'status.json')['state'] == 'completed'
    assert report['passed'] and report['embedding_gate_passed'] and report['fixed_stitcher_input_gate_passed']
    assert report['reduction_percent'] > 0
    assert all(x['bitwise_arrays_equal'] for x in report['embedding_equality'].values())
    assert all(x['pairs_equal'] and x['all_features_equal'] for x in report['pair_feature_equality'].values())
    processes = WindowsProcesses()
    launched = read(PROBE / 'launch_process.json')
    wait_until = time.monotonic() + 30
    while processes.creation(launched['launcher_pid']) == int(launched['launcher_start_filetime']):
        assert time.monotonic() < wait_until, 'Probe has not exited'
        time.sleep(.5)
    assert read(SOURCE / 'status.json')['state'] == 'paused'
    previous = read(max(SOURCE.glob('deadline_guard_*/result.json'), key=lambda p:p.stat().st_mtime))
    assert not any(processes.creation(int(pid)) == stamp for pid, stamp in previous['known_owned_processes'].items())
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.CreateMutexW.restype = ctypes.c_void_p
    kernel.WaitForSingleObject.argtypes = (ctypes.c_void_p, ctypes.c_uint32)
    kernel.ReleaseMutex.argtypes = (ctypes.c_void_p,)
    kernel.CloseHandle.argtypes = (ctypes.c_void_p,)
    handle = kernel.CreateMutexW(None, False, 'flyvis_fovea_gpu')
    result = kernel.WaitForSingleObject(handle, 0)
    assert result in (0, 128), 'GPU mutex is occupied'
    kernel.ReleaseMutex(handle); kernel.CloseHandle(handle)
    authorization = read(SOURCE / 'full_authorization.json')
    assert authorization['approved'] and now < datetime.fromisoformat(authorization['deadline'])
    assert authorization['deadline'] == '2026-10-15T07:40:00+03:00'
    assert not RUN.exists()
    RUN.mkdir(parents=True)
    for name in ('protocol.json', 'videos.json', 'environment.json'):
        copy_checked(SOURCE / name, RUN / name)
    for arm in ('B', 'C', 'D'):
        copy_checked(SOURCE / arm / 'model.pt', RUN / arm / 'model.pt')
    copy_checked(SOURCE / 'technical/report.json', RUN / 'technical/report.json')
    copy_checked(PROBE / 'report.json', RUN / 'technical/shared_masa_report.json')
    authorization.update(backend='Original Triton signatures; shared MASA frame features with unchanged per-arm box operations',
        conditional_human_answer='Если тест положительный и переход безопасен, прогон делаем с оптимизацией',
        conditional_gate_source='runs/z13_shared_masa_probe_v1_20261015/report.json',
        conditional_gate_passed=True, migration_at=now.isoformat())
    atomic_json(RUN / 'full_authorization.json', authorization)
    protocol = read(RUN / 'protocol.json')
    files = scope()['FILES'] + [str((RUN / 'protocol.json').relative_to(ROOT)).replace('\\', '/')]
    files += [str((RUN / arm / 'model.pt').relative_to(ROOT)).replace('\\', '/') for arm in protocol['arms']]
    files += ['weights/masa/masa_r50.pth', 'runs/flyvis/results/flow/0000/000/best_chkpt',
              'runs/train_cache/selection.json', 'tracking/retina.py', 'tracking/foveated_eyes.py']
    for video in protocol['videos']:
        files += [f'runs/train_cache/{video}/detections/raw_video.npz',
                  f'runs/train_cache/{video}/motion/flyvis_tiled.json',
                  f'data/bdd100k/videos/train/{video}.mov']
    identity = {p: digest(ROOT / p) for p in files}
    old_identity = read(SOURCE / 'inference_identity.json')
    old_prefix = SOURCE.relative_to(ROOT).as_posix()
    new_prefix = RUN.relative_to(ROOT).as_posix()
    common = {}
    for old_name, value in old_identity.items():
        new_name = old_name.replace(old_prefix + '/', new_prefix + '/')
        if new_name in identity:
            assert identity[new_name] == value, 'Changed shared input/code: ' + new_name
            common[new_name] = value
    assert len(common) > 100
    atomic_json(RUN / 'inference_identity.json', identity)
    reused = []
    for marker in sorted((SOURCE / 'completed_jobs').glob('*_sig_*.json')):
        original = read(marker)
        assert original['identity'] == old_identity
        arm = original['job']['arm']; video = original['job']['video']
        assert arm in protocol['arms'] and video in protocol['videos']
        hashes = {}
        for name in ('signatures.npz', 'signatures.json'):
            source = SOURCE / arm / 'signatures' / video / name
            target = RUN / arm / 'signatures' / video / name
            copy_checked(source, target); hashes[name] = digest(target)
        atomic_json(RUN / 'completed_jobs' / marker.name, {
            'identity': identity, 'job': original['job'], 'completed': original['completed'],
            'reused_from': marker.relative_to(ROOT).as_posix(),
            'reuse_proof': (RUN / 'reuse_proof.json').relative_to(ROOT).as_posix()})
        reused.append({'arm': arm, 'video': video, 'source_marker': marker.relative_to(ROOT).as_posix(), 'copied_sha256': hashes})
    proof = {'created': now.isoformat(), 'reused': reused, 'shared_inputs_and_code_equal': len(common),
        'signature_equivalence': 'Identical original compiled backend, worker and all scientific inputs; only downstream MASA scheduling changes.',
        'source_identity': (SOURCE / 'inference_identity.json').relative_to(ROOT).as_posix(),
        'new_execution_files': [p for p in identity if p not in common],
        'source_protocol_byte_identical': digest(SOURCE / 'protocol.json') == digest(RUN / 'protocol.json'),
        'probe_outputs_not_reused': True, 'one_gpu_worker': True, 'deadline': authorization['deadline']}
    atomic_json(RUN / 'reuse_proof.json', proof)
    atomic_json(RUN / 'status.json', {'state': 'authorized', 'completed_jobs': len(reused), 'total_jobs': 107,
                                    'reason': 'All shared-MASA gates passed; conditional author approval applies'})
    path = ROOT / 'experiments/project_history.json'; catalog = read(path)
    eid = 'z13_shared_masa_execution_v1'
    assert not any(c['id'] == eid for c in catalog['experiments'])
    area = next(c['area'] for c in catalog['experiments'] if c['id'] == 'z13_triton_execution_v1')
    catalog['experiments'].append({'id': eid, 'title': 'З13 — полное исполнение с общими признаками кадра MASA',
        'area': area, 'state': 'ready', 'started': now.strftime('%Y-%m-%d'),
        'question': 'Можно ли продолжить фиксированный HOTA после строгой проверки shared MASA без потери готовых выходов?',
        'hypothesis': 'Общие признаки кадра убирают повторные вычисления; неизменные сигнатуры и входы сшивки сохраняют научную постановку.',
        'expected': '105 заданий сигнатур, один совместный MASA и одна итоговая оценка; готовые видео переиспользуются с проверкой хэшей.',
        'acceptance': [{'criterion': 'Строгая проба скорости и совпадения массивов/входов сшивки пройдена до перехода.', 'met': True, 'evidence': 'runs/z13_shared_masa_probe_v1_20261015/report.json'},
                       {'criterion': 'Модели, пороги, научные входы и переиспользованные сигнатуры не изменились.', 'met': True, 'evidence': 'runs/z13_shared_masa_hota_v1_20261015/full/reuse_proof.json'},
                       {'criterion': 'Полный фиксированный HOTA завершён.', 'met': None}],
        'cannot_claim': 'Ускорение и побитовое равенство измерены на одном полном видео; весь набор ещё не оценён. Это инфраструктура, не улучшение HOTA или FlyVis.',
        'decision': 'Условное разрешение автора выполнено: продолжить тот же опыт в отдельном каталоге с общим MASA, один GPU-процесс, до 15.10 07:40.',
        'evidence': [{'kind': 'json', 'ref': 'runs/z13_shared_masa_hota_v1_20261015/full/reuse_proof.json', 'claim': 'Хэши, происхождение выходов и границы перехода'},
                     {'kind': 'json', 'ref': 'runs/z13_shared_masa_probe_v1_20261015/report.json', 'claim': 'Строгие входы сшивки и фактическая скорость'}]})
    parent = next(c for c in catalog['experiments'] if c['id'] == 'fly_spatial_hota_policy_z13_v1')
    parent['evidence'].append({'kind': 'json', 'ref': 'runs/z13_shared_masa_hota_v1_20261015/full/reuse_proof.json',
                               'claim': 'Автор разрешил переход после строгой проверки MASA; научный протокол не менялся'})
    path.write_text(json.dumps(catalog, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    plan = ROOT / 'dissertation/ЕДИНЫЙ_ПЛАН.md'; text = plan.read_text(encoding='utf-8')
    entry = f'''**{now:%d.%m.%Y, %H:%M} MSK** — условный переход на общий MASA подготовлен

*Получено (`z13_shared_masa_execution_v1`).* Все три критерия пробы `runs/z13_shared_masa_probe_v1_20261015/report.json` выполнены. Подготовлено отдельное исполнение `runs/z13_shared_masa_hota_v1_20261015/full`: {len(reused)} готовых сигнатур перенесены с проверкой хэшей и происхождения, общие код/модели/входы и научный протокол совпадают; `reuse_proof.json`. Сигнатуры остаются в прежнем Triton без дополнительного кэша параметров. Меняется только порядок расчёта MASA: признаки кадра общие, операции рамок каждого варианта прежние. Выходы пробы в полный опыт не перенесены. Изменения `run_full` после начала пробы касаются только сохранения/возобновления, функция `extract_video` не менялась; текущий код зафиксирован в `inference_identity.json`.

*Решение.* Подготовлено к разрешённому автором запуску после положительного теста, без нового согласования. Исходный процесс остаётся остановленным, GPU свободен. Абсолютный предел 15.10 07:40 не меняется; независимый сторож будет запущен вместе с контроллером. HOTA пока не измерена.

'''
    assert '## Журнал изменений\n\n' in text
    plan.write_text(text.replace('## Журнал изменений\n\n', '## Журнал изменений\n\n' + entry, 1), encoding='utf-8')
    print(json.dumps({'prepared': True, 'reused_signatures': len(reused), 'common_hashes': len(common), 'gpu_started': False}))


if __name__ == '__main__':
    main()
