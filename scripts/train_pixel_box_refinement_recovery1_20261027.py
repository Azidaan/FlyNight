"""Неизменное обучение после исправленного аудитора; прежняя граница090 сохраняется."""
from pathlib import Path
from datetime import datetime
import sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.recovery_research_service_20261027 import RUN as BASE,read,write

if __name__=='__main__':
    folder=BASE/'pixel_box_learning/recovery1';p=read(folder/'protocol.json');assert time.time()<datetime.fromisoformat(p['parent_absolute_deadline']).timestamp()-600
    audit=read(BASE/'pixel_box_audit/recovery1/controller_result.json');assert audit['exit_code']==0 and audit['all_owned_ended']
    assert read(BASE/'pixel_box_dataset/audit.json')['sequential_decode']
    import scripts.train_pixel_box_refinement_20261027 as worker
    worker.RUN=folder;original_read=worker.read
    def corrected_read(path):
        path=Path(path)
        if path==BASE/'pixel_box_audit/controller_result.json':return original_read(BASE/'pixel_box_audit/recovery1/controller_result.json')
        return original_read(path)
    worker.read=corrected_read;worker.run()
    # Сводный адрес впервые получает отчёт; прежние failed status/controller не затираются.
    target=BASE/'pixel_box_learning/report.json';assert not target.exists();write(target,read(folder/'report.json'))
