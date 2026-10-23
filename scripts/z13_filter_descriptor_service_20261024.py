"""Recover CPU capability selection without changing frozen scientific sources."""
from pathlib import Path
from datetime import datetime
import sys,time,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_filter_descriptor_cache_20261023 import RUN,read,write,sha,prior,journal,now
from scripts.z13_hota_deadline_guard import WindowsProcesses
REC=RUN/'recovery_cpu_1'
OLD="torch.set_num_threads(2);torch.set_default_device('cpu');assert not torch.cuda.is_available()"
NEW="torch.cuda.is_available=lambda:False;torch.cuda.device_count=lambda:0;torch.set_num_threads(2);torch.set_default_device('cpu');assert str(torch.ones(1,device='cpu').device)=='cpu'"


def transformed():
    source=(ROOT/'scripts/z13_filter_descriptor_cache_20261023.py').read_text(encoding='utf-8');assert source.count(OLD)==1
    return source.replace(OLD,NEW)


def verify_ended(folder):
    end=read(folder/'controller_result.json');control=read(folder/'controller.json');ps=WindowsProcesses()
    assert end['all_owned_ended'] and not end['remaining'] and ps.creation(control['pid'])!=control['birth']
    assert all(ps.creation(int(pid))!=birth for pid,birth in end['owned'].items())
    return end


def prepare_recovery():
    assert not REC.exists();end=verify_ended(RUN);assert end['exit_code']==1 and not list(RUN.glob('*/*/chunk_*.json'))
    p=read(RUN/'protocol.json');prior.verify_sources(p);original=read(RUN/'controller.json');remaining=(datetime.fromisoformat(original['save_at'])-now()).total_seconds();assert remaining>60
    files=['scripts/z13_filter_descriptor_service_20261024.py','scripts/z13_bounded_recovery_controller_20261022.py']
    recovery=dict(id=p['id'],device='cpu',budget_seconds=int(remaining),save_at=original['save_at'],hard_stop=original['hard_stop'],stop_request_path=p['stop_request_path'],created=now().isoformat(),original_protocol_sha256=sha(RUN/'protocol.json'),executed_source_sha256=hashlib.sha256(transformed().encode()).hexdigest(),original_started=original['started'],old_attempt=end,
        only_change='Force CPU capability selection before FlyVis import; check real CPU tensor and backbone. Original frozen scientific source unchanged; no completed extraction or training exists. CPU failure/logs retained. Budget still counted from original controller start; no tolerance change.',sha256={n:sha(ROOT/n) for n in files})
    write(REC/'protocol.json',recovery)
    journal('служебное восстановление выбора CPU','Первая попытка завершилась00:21:50MSK с exit1 до извлечения признаков: пустой CUDA_VISIBLE_DEVICES не скрыл CUDA. Это известная особенность Windows, уже исправленная ранее для balanced_ridge; её следовало учесть при подготовке. Используется прежний явный выбор CPU до импорта FlyVis; настоящий CPU-тензор и bb.dev проверяются. Исходный код,7671хеш и допуски неизменны; старые stderr/controller_result сохранены. Протокол runs/z13_filter_descriptor_cache_20261023/recovery_cpu_1/protocol.json; сохранение/конец остаются от первого контроллера, без продления5400с.')


def run_cache():
    p=read(REC/'protocol.json');prior.verify_sources(p);assert sha(RUN/'protocol.json')==p['original_protocol_sha256']
    source=transformed();assert hashlib.sha256(source.encode()).hexdigest()==p['executed_source_sha256']
    ns={'__name__':'recovered_cache','__file__':str(ROOT/'scripts/z13_filter_descriptor_cache_20261023.py')};exec(compile(source,ns['__file__'],'exec'),ns)
    elapsed=(now()-datetime.fromisoformat(p['original_started'])).total_seconds();ns['T0']=time.monotonic()-elapsed
    try:ns['worker']()
    except Exception as exc:write(RUN/'status.json',dict(state='failed',stage=str(exc),updated=now().isoformat()));raise


def prepare_training():
    from scripts import z13_filter_descriptor_20261023 as train
    assert not train.RUN.exists();end=verify_ended(REC);assert end['exit_code']==0 and read(RUN/'report.json')['parity_pass']
    # Resolve the successful controller explicitly; retain the original failed result.
    oldread=train.read
    def verified_read(path):
        return oldread(REC/'controller_result.json') if Path(path)==RUN/'controller_result.json' else oldread(path)
    train.read=verified_read
    try:train.prepare()
    finally:train.read=oldread
    p=read(train.RUN/'protocol.json');p['cache_execution_controller']='runs/z13_filter_descriptor_cache_20261023/recovery_cpu_1';p['service_sha256']={str(Path(__file__).relative_to(ROOT).as_posix()):sha(Path(__file__))};write(train.RUN/'protocol.json',p)


if __name__=='__main__':
    if '--prepare-recovery' in sys.argv:prepare_recovery()
    elif '--worker' in sys.argv:run_cache()
    elif '--prepare-training' in sys.argv:prepare_training()
    else:raise RuntimeError('Explicit service mode required')
