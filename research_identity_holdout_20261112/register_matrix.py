from common import *
assert read(OUT/'queue3_execution.json')['passed']
actual=read(OUT/'correspondence/audit_fast_v2.json');assert actual['passed']
binding=dict(actual)
binding.update(actual_result='research_identity_holdout_20261112/correspondence/audit_fast_v2.json',actual_result_sha256=sha(OUT/'correspondence/audit_fast_v2.json'),actual_source='research_identity_holdout_20261112/correspondence/audit_fast_v2.py',actual_source_sha256=sha(OUT/'correspondence/audit_fast_v2.py'),original_audit='interrupted_not_passed',interruption_receipt='research_identity_holdout_20261112/audit_v1_interruption.json',binding_only=True)
save('correspondence/audit.json',binding)
protocol=OUT/'review/proposed_compose_protocol.json';p=read(protocol);inputs=dict(p['inputs'])
for n,h in inputs.items():assert sha(ROOT/n)==h,n
pending={n:sha(ROOT/n) for n in p['pending_parent_binding']};inputs.update(pending)
save('review/compose_GO_A.json',dict(approved=True,registered=now(),protocol_sha256=sha(protocol),authorize_registered_matrix_audit=True,pending_parent_binding=pending))
for n in ['review/proposed_compose_protocol.json','review/compose_GO_A.json','review/compose_worker_A.py','review/compose_adapter_A.py','review/matrix_audit.py','review/audit_compose_A.py','correspondence/compact_after_audit.py','correspondence/summarize_A.py']:
    inputs[(OUT/n).relative_to(ROOT).as_posix()]=sha(OUT/n)
jobs=[dict(entry=(OUT/'correspondence/compact_after_audit.py').relative_to(ROOT).as_posix(),timeout=180),dict(entry=(OUT/'correspondence/summarize_A.py').relative_to(ROOT).as_posix(),timeout=180),dict(entry=(OUT/'review/compose_worker_A.py').relative_to(ROOT).as_posix(),args=['--go',str(OUT/'review/compose_GO_A.json')],timeout=300),dict(entry=(OUT/'review/matrix_audit.py').relative_to(ROOT).as_posix(),args=['--go',str(OUT/'review/matrix_GO_A.json'),'--decisions',str(OUT/'review/decisions_A.json'),'--output',str(OUT/'review/matrix_result_A.json')],timeout=600),dict(entry=(OUT/'review/audit_compose_A.py').relative_to(ROOT).as_posix(),args=['--matrix-result',str(OUT/'review/matrix_result_A.json')],timeout=300)]
save('queue4.json',dict(registered=now(),inputs=inputs,runner_sha256=sha(OUT/'run_queue.py'),jobs=jobs,deadline='2026-11-13T07:35:00+03:00'))
journal('Независимая точная ускоренная проверкаCensus завершена; прерванная первоначальная реализация не подменяется успешным исходом. Источник research_identity_holdout_20261112/correspondence/audit_fast_v2.json; audit.json является явной ссылочной привязкой к этому результату и фактическому исходнику. Зарегистрировано полное назначение4вариантов×54события; пороги и компоненты заморожены. Дополнительная оценка сохранения именно целевого номера явно отделена от прежней метрики присутствия физического объекта под любым номером; она добавлена после результатов отдельной ветви восстановления и не используется для выбора правила.')
print('MATRIX_QUEUE_REGISTERED')
