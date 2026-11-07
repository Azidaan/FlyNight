"""Отдельное исполнение той же первой проверки, вся симуляция в FP64."""
from pathlib import Path
import hashlib
import json
import os
import shutil
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'runs/flyvis_y1_latency_20261104';OUT=ROOT/'runs/flyvis_y1_latency_fp64_20261104'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    OUT.mkdir(exist_ok=False)
    old=(ROOT/'scripts/flyvis_y1_latency_20261104.py').read_text(encoding='utf-8')
    source=old.replace("OUT = ROOT / 'runs/flyvis_y1_latency_20261104'", "OUT = ROOT / 'runs/flyvis_y1_latency_fp64_20261104'")
    source=source.replace('    import numpy as np\n    import cv2\n    import torch\n    from tracking.flyvis_motion',
        "    requested_device = os.environ.get('Y1_EXECUTION_DEVICE', 'cpu')\n    if requested_device == 'cpu':os.environ['CUDA_VISIBLE_DEVICES'] = ''\n    import numpy as np\n    import cv2\n    import torch\n    import importlib\n    from contextlib import nullcontext\n    from tracking.flyvis_adapter import activate_flyvis\n    fv=activate_flyvis()\n    if requested_device == 'cpu':\n        for m in [fv,importlib.import_module('flyvis.network.initialization'),importlib.import_module('flyvis.task.decoder')]:m.device=torch.device('cpu')\n    torch.set_default_device(requested_device)\n    torch.set_default_dtype(torch.float64)\n    from tracking.flyvis_motion",1)
    source=source.replace("with GpuLock('Local\\\\DissertationSingleGpu'):","with (GpuLock('Local\\\\DissertationSingleGpu') if requested_device == 'cuda' else nullcontext()):",1)
    source=source.replace('        net, dec, eye, dev = model.network, model.decoder, model.eye, model.flyvis.device',
        '        model.network.double();model.decoder.double();model.eye.conv.double()\n        net, dec, eye, dev = model.network, model.decoder, model.eye, model.flyvis.device',1)
    source=source.replace("assert str(dev).startswith('cuda') and model.integration_steps == 2", "assert str(dev) == requested_device and model.integration_steps == 2",1)
    source=source.replace('        torch.cuda.reset_peak_memory_stats()',"        if requested_device == 'cuda':torch.cuda.reset_peak_memory_stats()",1)
    source=source.replace('                free, _ = torch.cuda.mem_get_info()',"                free = torch.cuda.mem_get_info()[0] if requested_device == 'cuda' else __import__('psutil').virtual_memory().available",1)
    source=source.replace('                torch.cuda.synchronize()',"                if requested_device == 'cuda':torch.cuda.synchronize()",1)
    source=source.replace('torch.cuda.max_memory_allocated()',"torch.cuda.max_memory_allocated() if requested_device == 'cuda' else 0")
    # Только численная точность; те же события, уравнения, dt, веса и порядок операций.
    begin=source.index('def infer():');end=source.index('\ndef reference():')
    body=source[begin:end].replace('np.float32','np.float64')
    source=source[:begin]+body+source[end:]
    assert "else nullcontext()" in source
    assert "if requested_device == 'cuda':torch.cuda.synchronize()" in source
    worker=ROOT/'scripts/flyvis_y1_latency_fp64_20261104.py';worker.write_text(source,encoding='utf-8')
    audit=(ROOT/'scripts/audit_flyvis_y1_latency_20261104.py').read_text(encoding='utf-8').replace('from scripts.flyvis_y1_latency_20261104 import', 'from scripts.flyvis_y1_latency_fp64_20261104 import')
    (ROOT/'scripts/audit_flyvis_y1_latency_fp64_20261104.py').write_text(audit,encoding='utf-8')
    old_protocol=json.loads((OLD/'protocol.json').read_text(encoding='utf-8'))
    p=dict(old_protocol)
    p.update(id='flyvis_y1_latency_fp64_v1',registered_msk=datetime.now(ZoneInfo('Europe/Moscow')).isoformat(),
             execution_precision='All network/decoder/image averaging arithmetic float64; promoted original checkpoint values, no learning',
             primary_execution_device='cpu',gpu_scope='One fixed full double-precision case only for numerical agreement',
             previous_unaccepted='runs/flyvis_y1_latency_20261104/recovery3/chain_result.json',
             old_precision_failure_preserved=True,
             numerical_tolerance_unchanged=1e-5)
    for name in ['scripts/flyvis_y1_latency_20261104.py','scripts/audit_flyvis_y1_latency_20261104.py','scripts/control_flyvis_y1_latency_20261104.py']:
        p['frozen_sources'].pop(name,None)
    for name in ['scripts/flyvis_y1_latency_fp64_20261104.py','scripts/audit_flyvis_y1_latency_fp64_20261104.py',
                 'scripts/flyvis_y1_fp64_gpu_check_20261104.py','scripts/control_flyvis_y1_fp64_20261104.py']:
        p['frozen_sources'][name]=sha(ROOT/name)
    (OUT/'protocol.json').write_text(json.dumps(p,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (OUT/'inputs').mkdir()
    for file in (OLD/'inputs').iterdir():os.link(file,OUT/'inputs'/file.name)
    for name in ['textures.npz','textures.json','geometry.npz','authorization.json','unit_checks.json']:
        shutil.copy2(OLD/name,OUT/name)
    (OUT/'inputs_completed.json').write_text(json.dumps(dict(complete=True,cases=64,protocol_sha256=sha(OUT/'protocol.json'),
         reused_inputs_by_sha=True,source='runs/flyvis_y1_latency_20261104/inputs_completed.json'),indent=2)+'\n',encoding='utf-8')
    before=OUT/'before';before.mkdir()
    for name in ['project_history.json','flyvis_core_dynamics_hypothesis.json','system_problems.json']:
        shutil.copy2(ROOT/'experiments'/name,before/name)
    history_path=ROOT/'experiments/project_history.json';history=json.loads(history_path.read_text(encoding='utf-8'))
    first=next(c for c in history['experiments'] if c['id']=='flyvis_y1_latency_gate_v1')
    fresh=json.loads(json.dumps(first))
    first.update(state='inconclusive',done='2026-11-04',
       obtained='Подготовлены64входа, сохранены8полныхGPU-последовательностей. GPUожидалкопированиявCPU; исходныйконтроллерcode15сохранён. CPU-FP32максимум1,0460615158e-5 и CPU-FP64противGPU-FP32максимум1,2713371799e-5 выше зарегистрированного1e-5. Источники: runs/flyvis_y1_latency_20261104/infer_external_stop.json,cpu_reference_fp32_failed/cpu_reference.json,cpu_reference_fp64_failed/cpu_reference.json.',
       interpretation='Техническая численная приёмка не пройдена; полная серия не выполнена. Научный вывод о задержке или временных константах не получен.',
       decision='Не продвигать эту частичную серию как научный итог. Отдельное полное исполнение в двойной точности с теми же критериями, а исходныеошибки/выходы сохранить.',
       cannot_claim='Нельзя утверждать отсутствие задержки, провал ядраFlyVis или завершениеЯ1. Допускнеослаблен, ниоднаготоваяGPU-FP32последовательностьнеповторялась.')
    first['acceptance'][0].update(met=False,evidence='runs/flyvis_y1_latency_20261104/cpu_reference_fp64_failed/cpu_reference.json')
    fresh.update(id=p['id'],title='Я1 — Переходный отклик штатной FlyVis в двойной точности',state='planned',done='',obtained='',interpretation='',decision='',cannot_claim='')
    fresh['evidence']=[dict(kind='json',ref='runs/flyvis_y1_latency_fp64_20261104/protocol.json',claim='Отдельное исполнение до новых выходов, с теми же научными условиями и допуском'),
                       dict(kind='json',ref='runs/flyvis_y1_latency_fp64_20261104/authorization.json',claim='Тот же первый разрешённый диагностический этап без обучения')]
    history['experiments'].append(fresh)
    temp=history_path.with_suffix('.fp64.next');temp.write_text(json.dumps(history,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');temp.replace(history_path)
    (OUT/'status.json').write_text(json.dumps(dict(stage='registered',done=0,total=64),indent=2)+'\n',encoding='utf-8')
    note='\n\n**04.11.2026 — отдельное численное исполнение первой проверки Я1.** Исходная серия `flyvis_y1_latency_gate_v1` сохранила8готовыхGPU-FP32последовательностей, но численнаяприёмка1e-5непройдена; это техническаянеполнота, не научный отрицательныйрезультат. Всеисточники/контроллеры/отказысохраненыв `runs/flyvis_y1_latency_20261104/`. До новых результатов зарегистрирована `flyvis_y1_latency_fp64_v1`: весь расчётсети/декодера/глаза вdouble(float64,двойнаяточность), основнойрасчётCPU иоднаполнаяконтрольнаяпоследовательностьGPUвтойжеточности. Тежевеса/структура/уравнения/времена/64точныхвхода/DIS/метрики/научныйкритерий/допуск1e-5; исходныеполувыходыFP32неподменяютсяновыми. ФиксированныесохранённыеCPU-FP64ответыпервойпоследовательностииспользуютсякакотдельныйсверочныйисточникпоSHA, а невыдаютсязауспешнуюсверкусо старымGPU-FP32. Новогообученияилиизменениядинамикинет. Протокол `runs/flyvis_y1_latency_fp64_20261104/protocol.json`.\n'
    with (ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md').open('a',encoding='utf-8') as f:f.write(note)
    print('Separate FP64 execution registered; scientific criteria/tolerance unchanged; previous run unaccepted.')


if __name__=='__main__':main()
