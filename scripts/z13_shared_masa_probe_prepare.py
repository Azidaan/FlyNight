"""Prespecify the authorized switch test using only saved full B/C/D outputs."""
from datetime import datetime
from pathlib import Path
import ctypes
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.z13_hota_deadline_guard import WindowsProcesses
from tracking.z13_checkpoint import atomic_json

RUN = ROOT / 'runs/z13_shared_masa_probe_v1_20261015'
FULL = ROOT / 'runs/z13_triton_hota_v1_20261014/full'
now = datetime.now().astimezone()
status = json.loads((FULL / 'status.json').read_text(encoding='utf-8'))
assert status['state'] == 'paused'
last = max(FULL.glob('deadline_guard_*/result.json'), key=lambda p:p.stat().st_mtime)
ended = json.loads(last.read_text(encoding='utf-8'))
processes = WindowsProcesses()
assert not any(processes.creation(int(pid)) == stamp for pid, stamp in ended['known_owned_processes'].items())
k = ctypes.WinDLL('kernel32', use_last_error=True)
k.CreateMutexW.restype = ctypes.c_void_p; k.WaitForSingleObject.argtypes = (ctypes.c_void_p, ctypes.c_uint32)
k.ReleaseMutex.argtypes = (ctypes.c_void_p,); k.CloseHandle.argtypes = (ctypes.c_void_p,)
handle = k.CreateMutexW(None, False, 'flyvis_fovea_gpu')
assert k.WaitForSingleObject(handle, 0) in (0, 128)
k.ReleaseMutex(handle); k.CloseHandle(handle)
assert not RUN.exists(); RUN.mkdir()
video = '00067cfb-e535423e'
sources = {'B': 'runs/z13_triton_hota_v1_20261014/full/B/signatures',
           'C': 'runs/z13_triton_hota_v1_20261014/full/C/signatures',
           'D': 'runs/z13_cache_speed_v1_20261014/D/triton'}
files = {arm: hashlib.sha256((ROOT / p / video / 'signatures.npz').read_bytes()).hexdigest() for arm,p in sources.items()}
protocol = {'id': 'z13_shared_masa_probe_v1', 'created': now.isoformat(), 'video': video,
    'signature_roots': sources, 'signature_sha256': files, 'arms': ['B', 'C', 'D'],
    'gates': {'embedding_arrays': 'Bitwise equal first/last fp16 arrays for every B/C/D life.',
              'stitch_inputs': 'Exact same candidate pair indices and complete pair feature matrices for fixed unchanged stitcher.',
              'speed': 'Shared total wall time below separate total wall time.'},
    'precision': 'Same original fp32 model, fp16 saved embeddings; no AMP, no batching boxes across arms.',
    'budget_seconds': 900, 'includes': 'Original three full-video MASA passes, one joint pass, CPU pair-feature comparison.',
    'decision': 'If all gates pass, continue same fixed HOTA in isolated shared-MASA execution; otherwise resume original Triton.',
    'limits': 'One complete video; no claim about full-set speed or changed HOTA; no training, downloads or heldout.'}
atomic_json(RUN / 'protocol.json', protocol)
atomic_json(RUN / 'status.json', {'state': 'authorized'})
atomic_json(RUN / 'stop_verified.json', {'own_hota_tree_exited': True, 'gpu_mutex_free': True,
            'completed_jobs_preserved': status['completed_jobs'], 'at': now.isoformat()})
path = ROOT / 'experiments/project_history.json'
catalog = json.loads(path.read_text(encoding='utf-8'))
assert not any(c['id'] == protocol['id'] for c in catalog['experiments'])
area = next(c['area'] for c in catalog['experiments'] if c['id'] == 'z13_triton_execution_v1')
catalog['experiments'].append({'id': protocol['id'], 'title': 'З13 — совместные признаки MASA: скорость и сохранение входа сшивки',
    'area': area, 'state': 'running', 'started': now.strftime('%Y-%m-%d'),
    'question': 'Можно ли убрать повторный расчёт общих кадров MASA и безопасно продолжить фиксированный HOTA?',
    'hypothesis': 'Признаки всего кадра не зависят от рамок; совместный проход должен сохранить эмбеддинги и входы сшивки, сократив время.',
    'expected': 'Оригинальные отдельные проходы B/C/D и совместный проход на одном полном видео, строгая проверка массивов и кандидатов.',
    'acceptance': [{'criterion': value, 'met': None} for value in protocol['gates'].values()],
    'cannot_claim': protocol['limits'], 'decision': protocol['decision'],
    'evidence': [{'kind': 'json', 'ref': 'runs/z13_shared_masa_probe_v1_20261015/protocol.json', 'claim': 'Источники, допуски, бюджет и условия перехода до прогона'}]})
path.write_text(json.dumps(catalog, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
plan = ROOT / 'dissertation/ЕДИНЫЙ_ПЛАН.md'; text = plan.read_text(encoding='utf-8')
entry = f'''**{now:%d.%m.%Y, %H:%M} MSK** — автор разрешил проверку совместного MASA и условное продолжение

*Постановка (`z13_shared_masa_probe_v1`).* Автор: «поставь на паузу, сделай тест. Если он положительный — и переход безопасен и не повлияет негативно на результаты — прогон делаем с оптимизацией». Исходный Triton HOTA приостановлен; 36 готовых заданий (B 35 и C 1) сохранены. Выход собственного дерева и свободный GPU проверены, `runs/z13_shared_masa_probe_v1_20261015/stop_verified.json`. До результата фиксируются: оригинальные B/C/D проходы MASA по полному видео, совместный проход; побитовое совпадение всех fp16 эмбеддингов, точное совпадение пар и матриц признаков фиксированной сшивки, положительная экономия времени. Источники и хэши — `protocol.json`. До 15 минут на GPU и CPU проверки вместе, один GPU-процесс. Модель, точность, рамки каждого варианта и порядок обработки рамок не меняются. GPU окно по-прежнему до 07:40. Если все условия выполнены — отдельное полное исполнение с общим MASA; иначе исходный Triton. Нового обучения, данных или HOTA в технической пробе нет.

'''
plan.write_text(text.replace('## Журнал изменений\n\n', '## Журнал изменений\n\n' + entry, 1), encoding='utf-8')
print(json.dumps(protocol, ensure_ascii=False))
