"""Record failed CPU isolation attempt and authorize only service-level repair."""
from pathlib import Path
from datetime import datetime
import sys
import shutil
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_center_hota_20261021 import read,write,sha,now
from scripts.z13_hota_deadline_guard import WindowsProcesses
RUN=ROOT/'runs/z13_balanced_ridge_20261022';OUT=RUN/'recovery_cpu'


def main():
    assert not OUT.exists();p=read(RUN/'protocol.json');end=read(RUN/'controller_result.json');old=read(RUN/'controller.json');ps=WindowsProcesses()
    assert end['exit_code']==1 and end['all_owned_ended'] and not end['remaining']
    assert not any(ps.creation(int(pid))==birth for pid,birth in end['owned'].items())
    assert not (RUN/'technical.json').exists() and not (RUN/'D').exists()
    for name,digest in p['sha256'].items():assert sha(ROOT/name)==digest,name
    original=ROOT/'scripts/z13_balanced_ridge_20261022.py';source=original.read_text(encoding='utf-8')
    old_import='import torch\n    torch.set_num_threads(2);torch.set_default_device(\'cpu\')';assert source.count(old_import)==1
    import hashlib
    transformed=source.replace(old_import,"import torch\n    torch.cuda.is_available=lambda:False\n    torch.cuda.device_count=lambda:0\n    torch.set_num_threads(2);torch.set_default_device('cpu')\n    assert str(torch.ones(1,device='cpu').device)=='cpu'")
    remaining=(datetime.fromisoformat(old['save_at'])-now()).total_seconds();assert remaining>60
    files=['scripts/z13_cpu_isolated_recovery_20261022.py','scripts/z13_bounded_recovery_controller_20261022.py','scripts/z13_prepare_balanced_recovery_20261022.py']
    OUT.mkdir();shutil.copy2(RUN/'status.json',OUT/'previous_failed_status.json')
    recovery=dict(id=p['id'],device='cpu',budget_seconds=int(remaining),save_at=old['save_at'],hard_stop=old['hard_stop'],stop_request_path='runs/z13_balanced_ridge_20261022/stop.request',
        original_protocol_sha256=sha(RUN/'protocol.json'),original_script_sha256=sha(original),executed_source_sha256=hashlib.sha256(transformed.encode('utf-8')).hexdigest(),
        only_change='Force CPU capability selection before any CUDA query, as in the prior validated CPU recovery. Explicit actual CPU tensor and backbone-device checks remain. CUDA device enumeration with -1 crashed in a separate isolated probe; never call it in this CPU run. Scientific algorithm/inputs/lambda/weights/criteria unchanged. New service controller writes original stop path. No completed stages exist to repeat.',
        old_attempt=read(RUN/'controller_result.json'),sha256={n:sha(ROOT/n) for n in files},created=now().isoformat())
    write(OUT/'protocol.json',recovery)
    plan=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';content=plan.read_text(encoding='utf-8');anchor='## Журнал изменений\n';assert content.count(anchor)==1
    event=f'\n**{now():%d.%m.%Y, %H:%M} MSK** — служебное восстановление CPU `z13_balanced_ridge_v1`\n\nПервая попытка остановилась до модели/извлечения признаков: assert not torch.cuda.is_available(). Пустая строка CUDA_VISIBLE_DEVICES не обеспечила CPU-изоляцию. Отдельная проба с -1 завершилась access violation внутри torch.cuda.is_available, до научной работы. Используется явный CPU-выбор без перечисления CUDA, как в прежнем CPU-восстановлении; отдельно проверен настоящий тензор CPU, остаётся проверка bb.dev. Старые журналы и controller_result exit1 сохранены. Исходный научный файл/протокол неизменны. Бюджет отсчитывается от первого контроллера: сохранение '+old['save_at']+', окончание '+old['hard_stop']+'. Источник: runs/z13_balanced_ridge_20261022/recovery_cpu/protocol.json. Завершённых научных этапов нет; повтор обучения не выполняется.\n'
    plan.write_text(content.replace(anchor,anchor+event,1),encoding='utf-8')
    print('Remaining original seconds',int(remaining))


if __name__=='__main__':main()
