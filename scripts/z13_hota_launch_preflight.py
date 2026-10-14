"""Verify and record the human-authorized ten-hour inference window."""
from datetime import datetime, timedelta
import ctypes
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RUN = ROOT / 'runs/z13_spatial_hota_v1_20261014'


def main():
    import msvcrt
    from tracking.z13_checkpoint import atomic_json
    assert not (RUN / 'full_authorization.json').exists(), 'Existing authorization requires explicit review'
    assert not (RUN / 'stop.request').exists(), 'Stop request remains'
    lock = (RUN / 'controller.lock').open('a+b')
    if lock.tell() == 0:
        lock.write(b'0'); lock.flush()
    lock.seek(0); msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.CreateMutexW.restype = ctypes.c_void_p
    kernel.WaitForSingleObject.argtypes = (ctypes.c_void_p, ctypes.c_uint32)
    kernel.ReleaseMutex.argtypes = (ctypes.c_void_p,)
    kernel.CloseHandle.argtypes = (ctypes.c_void_p,)
    handle = kernel.CreateMutexW(None, False, 'flyvis_fovea_gpu')
    if not handle:
        raise OSError(ctypes.get_last_error())
    acquired = False
    try:
        acquired = kernel.WaitForSingleObject(handle, 0) in (0, 0x80)
        assert acquired, 'Another inference worker owns GPU mutex'
        assert json.loads((RUN / 'technical/report.json').read_text(encoding='utf-8'))['passed']
        assert json.loads((RUN / 'technical/status.json').read_text(encoding='utf-8'))['state'] == 'completed'
        protocol = json.loads((RUN / 'protocol.json').read_text(encoding='utf-8'))
        expected = []
        for arm in protocol['arms']:
            expected += [RUN / arm / 'model.pt', RUN / arm / 'policy.json']
        for video in protocol['videos']:
            expected += [ROOT / f'runs/train_cache/{video}/detections/raw_video.npz',
                         ROOT / f'runs/train_cache/{video}/motion/flyvis_tiled.json',
                         ROOT / f'data/bdd100k/videos/train/{video}.mov',
                         ROOT / f'runs/fly_signatures/{video}/signatures.npz',
                         ROOT / f'runs/chain_filter_rank/repro/dev/{video}.json',
                         ROOT / f'runs/stitch_v2/gap150_dist5/stitch/geometry_2/dev/{video}.json']
        expected += [ROOT / 'weights/masa/masa_r50.pth']
        assert all(p.is_file() for p in expected), [str(p) for p in expected if not p.is_file()]
        assert len(protocol['videos']) == 35
        now = datetime.now().astimezone()
        authorization = {'approved': True, 'human_answer': 'Да, запускай до 10 часов',
                         'authorized_at': now.isoformat(), 'maximum_seconds': 36000,
                         'deadline': (now + timedelta(hours=10)).isoformat(),
                         'scope': 'One fixed policy: B/C/D on full dev35 with original MASA and HOTA. No training, no downloads, no heldout videos.',
                         'one_gpu_worker': True, 'readiness_paths_checked': len(expected)}
        atomic_json(RUN / 'full_authorization.json', authorization)
        atomic_json(RUN / 'status.json', {'state': 'authorized', 'completed_jobs': 0,
                                         'total_jobs': 109, 'updated': now.isoformat()})
        catalog_path = ROOT / 'experiments/project_history.json'
        catalog = json.loads(catalog_path.read_text(encoding='utf-8'))
        card = next(c for c in catalog['experiments'] if c['id'] == protocol['id'])
        assert card['state'] == 'ready'
        card['state'] = 'running'
        card['evidence'].append({'kind': 'json', 'ref': 'runs/z13_spatial_hota_v1_20261014/full_authorization.json',
                                'claim': 'Прямое разрешение автора на окно до 10 часов; готовность файлов и свободные блокировки проверены'})
        catalog_path.write_text(json.dumps(catalog, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
        plan = ROOT / 'dissertation/ЕДИНЫЙ_ПЛАН.md'
        entry = f'''**{now:%d.%m.%Y, %H:%M} MSK** — автор разрешил полный прогон HOTA З13 v2 до 10 часов

Автор ответил «Да, запускай до 10 часов». Разрешено одно сравнение стенда и B/C/D на dev35 с прежней MASA, готовыми весами и зафиксированными порогами верхнего 1%; обучения, загрузки и отложенных видео нет. Готовность всех {len(expected)} необходимых файлов и свободные блокировки контроллера/GPU проверены. `runs/z13_spatial_hota_v1_20261014/full_authorization.json`: окно до {(now + timedelta(hours=10)):%d.%m.%Y %H:%M:%S} MSK. Карточка `fly_spatial_hota_policy_z13_v1` переведена в running до прогона. Завершённые видео сохраняются; по сроку останавливается только собственное дерево. HOTA пока не измерена.

'''
        content = plan.read_text(encoding='utf-8')
        plan.write_text(content.replace('## Журнал изменений\n\n', '## Журнал изменений\n\n' + entry, 1), encoding='utf-8')
        print(json.dumps(authorization, ensure_ascii=False))
    finally:
        if acquired:
            kernel.ReleaseMutex(handle)
        kernel.CloseHandle(handle); lock.close()


if __name__ == '__main__':
    main()
