"""Validate resumed sources, immutable outputs, dependency chain and live identity."""
from pathlib import Path
import sys,inspect,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_matched_trackers_20261024 import RUN,read,write,sha,now
from scripts.z13_localization_labels_20261023 import verify_sources
from scripts.z13_hota_deadline_guard import WindowsProcesses


def main():
    from scripts.z13_matched_fps_recovery_20261024 import OVERLAY,install
    from scripts import build_fly_signatures as B
    source=inspect.getsource(B.build_video)
    assert source.count('args.flow_root')==1 and '"fps"' in next(line for line in source.splitlines() if 'args.flow_root' in line)
    for folder in (RUN,RUN/'gpu_stage',RUN/'gpu_recovery1',RUN/'gpu_recovery2',RUN/'stock_controls'):verify_sources(read(folder/'protocol.json'))
    install();metadata=read(OVERLAY/'manifest.json');assert len(metadata['rows'])==7
    old=read(RUN/'gpu_recovery1/controller_result.json');assert old['exit_code']==1 and old['all_owned_ended'] and not old['remaining']
    ps=WindowsProcesses();assert all(ps.creation(int(pid))!=birth for pid,birth in old['owned'].items())
    copied=list((RUN/'gpu_recovery1/jobs').glob('*.json'));assert len(copied)==28
    for marker in copied:
        assert read(marker)==read(RUN/'gpu_recovery2/jobs'/marker.name)
        for n,h in read(marker)['outputs'].items():assert sha(ROOT/n)==h
    c=read(RUN/'gpu_recovery2/controller.json');assert ps.creation(c['pid'])==c['birth']
    assert c['save_at']==read(RUN/'gpu_recovery1/controller.json')['save_at'] and c['hard_stop']==read(RUN/'gpu_recovery1/controller.json')['hard_stop']
    q=read(RUN/'stock_controls/queue3.json');assert ps.creation(q['pid'])==q['birth'] and not (RUN/'stock_controls/controller.json').exists()
    from scripts.z13_matched_stock_execution3_20261024 import verify
    verify()
    cache=ROOT/'runs/z13_filter_descriptor_fixed_20261024/cache';cpu=read(cache/'controller.json');assert ps.creation(cpu['pid'])==cpu['birth']
    handoff=ROOT/'runs/z13_weekend_20261024/HANDOFF.md';before=handoff.read_bytes()
    entry=f'''# Актуальное восстановление — {now():%d.%m.%Y %H:%M} МСК

Ниже сохранена предыдущая передача состояния. Основной GPU теперь gpu_recovery2, scripts/z13_matched_pipeline_recovery2_20261024.py; контроллер{c['pid']}/birth{c['birth']}. Предыдущий gpu_recovery1 завершился exit1 в02:02:25: отсутствующий motion/flyvis_tiled.json пятого ролика. Сборщик читал там только fps. Отдельный fps_metadata_recovery сверяет все7частот с контейнерами,summary и всеми существующими старыми файлами точно. Только путь метаданных изменён,научные источники неизменны. Все28готовыхзаданий проверены и пропущены;старыеошибки сохранены. Сохранение{c['save_at']},конец{c['hard_stop']};бюджет не продлён. Новое видео уже продвигается,смотреть status и журнал.

Очередь stock теперь scripts/z13_matched_stock_queue3_20261024.py,realPID{q['pid']}/birth{q['birth']}. Queue2 завершилась после ошибки родителя,её failure сохранён. Queue3 ждёт recovery2,проверяет всю цепь прошлыхошибок,записывает и проверяет основной итог,потом запускает scripts/z13_matched_stock_execution3_20261024.py. Та же поправка fps применяется к stock до запуска; execution_v2 и scientific protocol неизменны,execution_v3 фиксирует новые служебныехеши. Старый matched_failure.json диспетчера остаётся как история; новый успешный итог будет matched_recovery2.json и matched_recovery2_verified.json. Старый локальный диспетчер продолжает stock-финализацию и CPU-зависимости,дубль не запускать.

CPU контроллер{cpu['pid']}/birth{cpu['birth']} остаётся активным. Прежние57чужихграфправок сохранять. Проверка восстановления: fps_recovery_verification.json.

'''.encode('utf-8')
    if not before.startswith(b'# '+ 'Актуальное восстановление'.encode()):handoff.write_bytes(entry+before)
    report=dict(at=now().isoformat(),all_five_protocol_source_hashes_valid=True,completed28_jobs_and_hashes_unchanged=True,fps_exact_all7=True,original_deadline_unchanged=True,controller=c,queue=q,cpu_controller=cpu,signature_status=read(RUN/'gpu_recovery2/status.json'),cpu_status=read(cache/'status.json'))
    write(ROOT/'runs/z13_weekend_20261024/fps_recovery_verification.json',report);print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':main()
