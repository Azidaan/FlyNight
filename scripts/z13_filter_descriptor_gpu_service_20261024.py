"""Device-only extraction continuation in the original time budget."""
from pathlib import Path
from datetime import datetime
import sys,time,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_filter_descriptor_cache_20261023 import RUN,read,write,sha,prior,journal,now
from scripts.z13_filter_descriptor_service_20261024 import verify_ended
REC=RUN/'recovery_gpu_2'


def transformed():
    source=(ROOT/'scripts/z13_filter_descriptor_cache_20261023.py').read_text(encoding='utf-8')
    changes=[("os.environ['CUDA_VISIBLE_DEVICES']=''","os.environ['CUDA_VISIBLE_DEVICES']='0'"),
      ("torch.set_num_threads(2);torch.set_default_device('cpu');assert not torch.cuda.is_available()","torch.set_num_threads(2);torch.set_default_device('cpu');assert torch.cuda.is_available()"),
      ("assert bb is None or str(bb.dev)=='cpu'","assert bb is None or str(bb.dev).startswith('cuda')"),
      ("hx=torch.as_tensor(np.stack([data[e][0] for e in ch]),device='cpu')","hx=torch.as_tensor(np.stack([data[e][0] for e in ch]),device=(bb.dev if bb is not None else 'cpu'))"),
      ("record['seconds']=time.monotonic()-start;write(path,record)","record['seconds']=time.monotonic()-start;record['execution_device']='cuda';write(path,record)")]
    for old,new in changes:assert source.count(old)==1,old;source=source.replace(old,new)
    source=source.replace("RUN/'timing.json'","RUN/'gpu_timing.json'")
    old="if not (RUN/'gpu_timing.json').exists():";assert source.count(old)==1;source=source.replace(old,"if not (RUN/'gpu_timing.json').exists() and record.get('execution_device')=='cuda':")
    return source


def prepare():
    assert not REC.exists();end=verify_ended(RUN/'recovery_cpu_1');assert end['exit_code']==1
    old=read(RUN/'controller.json');remaining=(datetime.fromisoformat(old['save_at'])-now()).total_seconds();assert remaining>60
    timing=read(RUN/'timing.json');assert not timing['proceed'];prior.verify_sources(read(RUN/'protocol.json'))
    retained={p.relative_to(ROOT).as_posix():sha(p) for p in RUN.glob('*/*/chunk_*.json')};assert len(retained)==1
    files=['scripts/z13_filter_descriptor_gpu_service_20261024.py','scripts/z13_filter_descriptor_service_20261024.py','scripts/z13_bounded_recovery_controller_20261022.py','tracking/gpu_lock.py']
    p=dict(id=read(RUN/'protocol.json')['id'],created=now().isoformat(),device='cuda',budget_seconds=int(remaining),save_at=old['save_at'],hard_stop=old['hard_stop'],stop_request_path=read(RUN/'protocol.json')['stop_request_path'],original_started=old['started'],original_protocol_sha256=sha(RUN/'protocol.json'),executed_source_sha256=hashlib.sha256(transformed().encode()).hexdigest(),retained=retained,old_attempt=end,cpu_timing_sha256=sha(RUN/'timing.json'),
      only_change='Frozen feature extraction on one CUDA process instead of CPU; tensor/backbone devices explicitly checked. Same original8-episode groups, weights, statistics, ridge math, observed4frames, tolerances, data and scientific criteria. Keep completed CPU chunk unchanged; fit/descriptor still numpy CPU. Separate GPU timing gate; original5400seconds not extended. Author allows necessary runs and SaturdayGPU; no other research CUDA process found.',sha256={n:sha(ROOT/n) for n in files})
    write(REC/'protocol.json',p);journal('продолжение извлечения на одной GPU','CPU-воспроизведение первых8эпизодов прошло: chunk_00000.json. CPU2,625с/8эпизодов;3720групп, консервативно19710с против5170,71с остатка (runs/z13_filter_descriptor_cache_20261023/timing.json). Это штатная остановка по стоимости, не отрицательная эффективность. Замена только устройства извлечения: один CUDA-процесс с GpuLock, те же входы0..3,веса,нормы,lambda и допуски. ПервыйCPUблок сохраняется и не повторяется. Новый GPU-замер должен вписаться в остаток исходных5400с, сохранение01:51:46MSK/конец01:52:06MSK от первого контроллера; продления нет. Протокол recovery_gpu_2/protocol.json. Обучение позже только отдельным CPU-процессом.')


def worker():
    p=read(REC/'protocol.json');prior.verify_sources(p);assert sha(RUN/'protocol.json')==p['original_protocol_sha256'] and sha(RUN/'timing.json')==p['cpu_timing_sha256']
    assert all(sha(ROOT/n)==h for n,h in p['retained'].items());source=transformed();assert hashlib.sha256(source.encode()).hexdigest()==p['executed_source_sha256']
    ns={'__name__':'gpu_cache_continuation','__file__':str(ROOT/'scripts/z13_filter_descriptor_cache_20261023.py')};exec(compile(source,ns['__file__'],'exec'),ns)
    ns['T0']=time.monotonic()-(now()-datetime.fromisoformat(p['original_started'])).total_seconds()
    from tracking.gpu_lock import GpuLock
    try:
        with GpuLock('flyvis_fovea_gpu'):ns['worker']()
        assert all(sha(ROOT/n)==h for n,h in p['retained'].items())
    except Exception as exc:write(RUN/'status.json',dict(state='failed',stage=str(exc),updated=now().isoformat()));raise


def prepare_training():
    from scripts import z13_filter_descriptor_20261023 as train
    assert not train.RUN.exists();end=verify_ended(REC);assert end['exit_code']==0 and read(RUN/'report.json')['parity_pass']
    oldread=train.read
    def verified_read(path):return oldread(REC/'controller_result.json') if Path(path)==RUN/'controller_result.json' else oldread(path)
    train.read=verified_read
    try:train.prepare()
    finally:train.read=oldread
    p=read(train.RUN/'protocol.json');p['cache_execution_controller']='runs/z13_filter_descriptor_cache_20261023/recovery_gpu_2';p['service_sha256']=read(REC/'protocol.json')['sha256'];write(train.RUN/'protocol.json',p)


if __name__=='__main__':
    if '--prepare' in sys.argv:prepare()
    elif '--worker' in sys.argv:worker()
    elif '--prepare-training' in sys.argv:prepare_training()
    else:raise RuntimeError('Explicit mode required')
