from common import *
rp=OUT/'independent_review/reid/low_protocol_v3.json';p=read(rp);assert sha(rp)=='e4fcd9774101844adcb44f5293a7f0413f0b4d9c15b9acc575afc4ab156f1c49';inputs={}
for n,h in p['inputs'].items():assert sha(ROOT/n)==h,n;inputs[n]=h
for n in ['independent_review/reid/low_protocol_v3.json','run_queue.py','register_low_v3.py']:
 f=OUT/n;inputs[f.relative_to(ROOT).as_posix()]=sha(f)
cat=read(ROOT/'experiments/project_history.json');c=next(c for c in cat['experiments'] if c['id']=='remaining_causes_raw_candidates_20261112');c['evidence'].append(dict(kind='file',ref=rp.relative_to(ROOT).as_posix()));(ROOT/'experiments/project_history.json').write_text(json.dumps(cat,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
jobs=[dict(entry=(OUT/'independent_review/reid/low_worker_v3.py').relative_to(ROOT).as_posix(),args=['--go-protocol-sha256',sha(rp)],timeout=1200),dict(entry=(OUT/'independent_review/reid/low_audit_v3.py').relative_to(ROOT).as_posix(),timeout=180)]
save('queue6.json',dict(registered=now(),inputs=inputs,runner_sha256=sha(OUT/'run_queue.py'),jobs=jobs,deadline='2026-11-13T07:35:00+03:00',failure_rule='Second failure stops this line; no additional forward retries or tolerance relaxation.'))
journal('Техническое возобновление низкоуверенногоReIDвv3: до новых исходов зарегистрирован точный исторический порядок/составпакетовсэталоннойрамкой и отдельныепакетыновыхобластей наобщихпризнакахкадра. Прежние77баллов ивсепрежниетекущиедескрипторытребуютсясдопуском1e-6, допускнеослаблен. Изменениесоставапакетовпокарабочаяпричиначисленногорасхождения,неподтверждённая;проверкойбудетстрогийповторсостава. Старыеисходники/протокол/ошибкасохранены. Источники independent_review/reid/low_failure_v2.json,low_protocol_v3.json,queue6.json. ПриповторномнеуспехеновыйGPUповторнебудетзапускаться.')
print('REGISTERED_LOW_V3')
