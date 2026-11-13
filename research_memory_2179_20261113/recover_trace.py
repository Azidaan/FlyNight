from common import *
source=(OUT/'trace.py').read_text(encoding='utf8')
source=source.replace("def overlap(a,b):\n", "def overlap(a,b):\n    a=list(map(float,a)); b=list(map(float,b))\n")
source=source.replace("score=overlap(box,b)","box=list(map(float,box)); score=overlap(box,b)")
source=source.replace("save('trace.json'", "save('trace_v2.json'")
with (OUT/'trace_v2.py').open('x',encoding='utf8',newline='\n') as f: f.write(source)
failure=read(OUT/'science_queue_execution.json'); assert not failure['passed'] and len(failure['jobs'])==1
save('trace_recovery_protocol.json',dict(registered=now(),reason='Evaluation trace contained numpy.float32 overlap fractions; JSON writer rejected type. No scientific results or GPU run occurred. Cast scalar coordinates to Python float before geometry; preserve failed source, empty trace.json and original queue.',old_source_sha256=sha(OUT/'trace.py'),new_source_sha256=sha(OUT/'trace_v2.py'),old_failure=failure,new_criteria=False))
q=read(OUT/'science_queue.json');q['jobs'][0]['entry']=(OUT/'trace_v2.py').relative_to(ROOT).as_posix();q['inputs'][(OUT/'trace_v2.py').relative_to(ROOT).as_posix()]=sha(OUT/'trace_v2.py')
save('science2_queue.json',q)
journal('Техническое исправление до расчёта признаков2179: запись диагностической геометрии остановилась на типе numpy.float32, несовместимом с JSON. Исходный сбой и пустой trace.json сохранены; отдельная trace_v2.py переводит координаты в обычные числа. Критерии и научные условия не менялись. Источник research_memory_2179_20261113/trace_recovery_protocol.json.')
