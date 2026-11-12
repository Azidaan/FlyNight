from common import *
assert read(OUT/'review/bank_audit.json')['passed']
q=read(OUT/'queue2_execution.json');first=q['jobs'][:2]
assert len(first)==2 and all(j['result']['returncode']==0 and j['result']['reason'] is None for j in first)
save('bank_execution.json',dict(completed_jobs=first,recorded=now(),note='Closed bank jobs only; the still-running queue log is not bound as immutable input.'))
names=['review/reference_bank.npz','review/bank_protocol.json','review/bank_result.json','review/bank_audit.json','bank_execution.json']
binding=dict(registered=now(),descriptor_file='research_identity_repairs_20261112/review/reference_bank.npz',inputs={(OUT/n).relative_to(ROOT).as_posix():sha(OUT/n) for n in names},rule_unchanged=True,policy_outcomes_computed=False)
(OUT/'recovery574/descriptor_binding.json').write_bytes((json.dumps(binding,ensure_ascii=False,indent=2)+'\n').encode())
(OUT/'recovery574/GO.json').write_bytes((json.dumps(dict(approved=True,registered=now(),protocol_sha256=sha(OUT/'recovery574/proposal_protocol.json'),binding_sha256=sha(OUT/'recovery574/descriptor_binding.json')),indent=2)+'\n').encode())
journal('Подготовлены признаки всех конкурирующих историй: контроль банка пройден, исходные пять образцов переиспользованы точно. До расчёта правила восстановления зарегистрированы суммы банка, его протокола, результата и независимой проверки; продолжающаяся очередь не выдаётся за неизменный файл. Источники research_identity_repairs_20261112/recovery574/descriptor_binding.json и bank_execution.json. Правило подтверждения, пороги и отрицательные контроли не изменены.')
print('BANK_BOUND_BEFORE_POLICY')
