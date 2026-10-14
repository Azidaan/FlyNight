"""Audit existing state before resuming the author's paused Z13 training."""
from datetime import datetime
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import ctypes
    import msvcrt
    import torch
    from scripts.z13_spatial_train import RUN, identity
    from tracking.z13_checkpoint import atomic_json
    run = RUN.resolve()
    run.relative_to(ROOT)
    with (run / 'controller.lock').open('r+b') as lock:
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
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
            code = kernel.WaitForSingleObject(handle, 0)
            acquired = code in (0, 0x80)
            if not acquired:
                raise RuntimeError('Another GPU worker owns the mutex; no duplicate launch')
            status = json.loads((run / 'status.json').read_text(encoding='utf-8'))
            assert status['state'] == 'paused'
            protocol = json.loads((run / 'protocol.json').read_text(encoding='utf-8'))
            jobs = []
            for job in protocol['jobs']:
                folder = run / job['arm'] / job['name']
                saved_identity = json.loads((folder / 'identity.json').read_text(encoding='utf-8'))
                assert saved_identity == identity(protocol, job), f'Changed data/code: {job}'
                progress = json.loads((folder / 'status.json').read_text(encoding='utf-8'))
                jobs.append({'arm': job['arm'], 'name': job['name'], 'status': progress})
            checkpoint = torch.load(run / 'D/full/checkpoint.pt', map_location='cpu', weights_only=False)
            assert checkpoint['arm'] == 'D' and checkpoint['step'] == 1105
            assert len(checkpoint['history']) == checkpoint['step']
            assert checkpoint['identity'] == json.loads((run / 'D/full/identity.json').read_text(encoding='utf-8'))
            assert checkpoint['torch_version'] == str(torch.__version__)
            assert checkpoint['optimizer']['state'] and checkpoint['rng_cuda'] is not None
            assert all(k in checkpoint for k in ('generator', 'rng_python', 'rng_numpy', 'rng_torch'))
            stop = (run / 'stop.request').resolve()
            stop.relative_to(ROOT)
            assert stop.is_file()
            stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            archive = run / ('stop.request.author_pause_' + stamp)
            assert not archive.exists()
            report = {'created': datetime.now().astimezone().isoformat(),
                      'authorization': 'Автор 14.10.2026: Продолжи', 'resume_step': checkpoint['step'],
                      'all_job_code_data_identities_match': True,
                      'optimizer_and_all_rng_saved': True, 'controller_and_gpu_locks_available': True,
                      'jobs': jobs, 'scope': protocol['scope'], 'archived_stop_request': str(archive.relative_to(ROOT))}
            atomic_json(run / ('resume_preflight_' + stamp + '.json'), report)
            stop.rename(archive)
            print(json.dumps({'passed': True, 'resume_step': checkpoint['step'],
                              'completed_jobs': sum(j['status']['stage'] == 'predictions_saved' for j in jobs),
                              'report': str((run / ('resume_preflight_' + stamp + '.json')).relative_to(ROOT))}))
        finally:
            if acquired:
                kernel.ReleaseMutex(handle)
            kernel.CloseHandle(handle)


if __name__ == '__main__':
    main()
