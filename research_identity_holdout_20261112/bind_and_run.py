from common import *
import shutil
assert read(OUT/'queue1_execution.json')['passed']
assert read(OUT/'shared/extraction_audit.json')['passed']
assert read(OUT/'correspondence/core_result.json')['passed']
inputs={}
for n in ['shared/descriptors_A.npz','shared/extraction_protocol.json','shared/extraction_result.json','shared/extraction_audit.json','queue1_execution.json']:
    inputs[(OUT/n).relative_to(ROOT).as_posix()]=sha(OUT/n)
save('recovery/descriptor_binding.json',dict(registered=now(),descriptor_file='research_identity_holdout_20261112/shared/descriptors_A.npz',inputs=inputs))
shutil.copyfile(OUT/'recovery/proposed_protocol.json',OUT/'recovery/protocol.json')
for n in ['recovery/GO.json','correspondence/replay_GO.json']:
    save(n,dict(approved=True,registered=now(),binding_sha256=sha(OUT/('recovery/descriptor_binding.json' if n.startswith('recovery') else 'correspondence/replay_binding.json'))))
for n in ['recovery/protocol.json','recovery/descriptor_binding.json','correspondence/protocol.json','correspondence/replay_binding.json']:
    p=read(OUT/n)
    for f,h in p.get('inputs',{}).items():assert sha(ROOT/f)==h,f;inputs[f]=h
    for f,h in p.get('sources',{}).items():
        target=OUT/'recovery'/f;assert sha(target)==h,f;inputs[target.relative_to(ROOT).as_posix()]=h
    inputs[(OUT/n).relative_to(ROOT).as_posix()]=sha(OUT/n)
jobs=[dict(entry=(OUT/n).relative_to(ROOT).as_posix(),timeout=1800) for n in ['recovery/worker.py','recovery/audit.py','correspondence/replay_worker.py','correspondence/audit.py']]
save('queue2.json',dict(registered=now(),inputs=inputs,runner_sha256=sha(OUT/'run_queue.py'),jobs=jobs,deadline='2026-11-13T07:35:00+03:00'))
journal('Получена вычислительная готовность новых54эпизодов без оценки исходов: банк5901признаков на388кадрах и94прохода ядра на47историях прошли проверки. Источники research_identity_holdout_20261112/shared/extraction_audit.json, correspondence/core_result.json. Разрешён последовательный повтор фиксированной политики восстановления и считывания на всех54событиях; эталонные метки используются только отдельным оценщиком. Порогов, весов и производственных исходников не меняли. РезервB остаётся без новых исходов.')
print('CPU_QUEUE_REGISTERED')
