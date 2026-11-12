from common import *
import psutil
sys.path.insert(0,str(ROOT/'runs/flyvis_repairs_20261106/unification_20261107'))
import system_memory_guard as guard
q=read(OUT/'queue2_execution.json');job=q['jobs'][-1]
if q.get('passed'):
    print('ORIGINAL_AUDIT_ALREADY_PASSED');raise SystemExit()
assert job['entry']=='research_identity_holdout_20261112/correspondence/audit.py'
assert not (OUT/'correspondence/audit.json').exists()
p=read(OUT/'correspondence/audit_fast_protocol_v2.json')
assert p['auditor_sha256']==sha(OUT/'correspondence/audit_fast_v2.py')
for n,h in p['inputs'].items():assert sha(ROOT/n)==h,n
marker=guard.MARK+job['token'];report=read(job['report']);owned={};usage=[]
for pid,birth in report['owned'].items():
    parent=psutil.Process(int(pid));assert parent.create_time()==birth and marker in parent.cmdline()
    for process in [parent,*parent.children(recursive=True)]:
        if marker in process.cmdline():
            owned[process.pid]=process.create_time();usage.append(dict(pid=process.pid,birth=process.create_time(),cpu_user_seconds=process.cpu_times().user,command=process.cmdline()))
save('audit_v1_interruption.json',dict(registered=now(),reason='Explicit replacement of slow independent audit by a preregistered mathematically exact implementation; original scientific A result unchanged. Original audit is INTERRUPTED, never marked passed.',job=job,owned=owned,usage=usage,fast_protocol_sha256=sha(OUT/'correspondence/audit_fast_protocol_v2.json'),scientific_output_sha256=sha(OUT/'correspondence/result.json')))
errors=guard.stop_owned(owned,marker)
save('audit_v1_stop_receipt.json',dict(stopped=now(),errors=errors,owned=owned));assert not errors
inputs=dict(p['inputs']);inputs[(OUT/'correspondence/audit_fast_protocol_v2.json').relative_to(ROOT).as_posix()]=sha(OUT/'correspondence/audit_fast_protocol_v2.json')
save('queue3.json',dict(registered=now(),inputs=inputs,runner_sha256=sha(OUT/'run_queue.py'),jobs=[dict(entry=p['entrypoint'],timeout=1800)],deadline='2026-11-13T07:35:00+03:00'))
journal('Независимый исходный аудит полногоCensus прерван только как слишком дорогая реализация пересчёта; он НЕ объявлен пройденным. Все исходные432оценки и54области поиска остаются неизменными. До нового запуска зарегистрирован audit_fast_protocol_v2.json: те же24логических сравнения, вся допустимая область, точная медиана через целочисленную гистограмму вместо сортировки; каждый сохранённый набор стоимости и победитель проверяются без прореживания. Причина затрат:6 673 813 946сравниваемых пар пикселей, два крупных случая дают63,52процента; источник correspondence/runtime_actual_counts.json в research_identity_holdout_20261112. Остановка ограничена собственным процессом с совпавшими номером, временем создания и уникальной меткой запуска; audit_v1_interruption.json и audit_v1_stop_receipt.json сохраняют факт.')
print('OWNED_SLOW_AUDIT_STOPPED_FAST_REGISTERED')
