"""Register the narrow solver recovery; preserve all failed artifacts."""
import json,hashlib,psutil,time
from pathlib import Path
from datetime import datetime,timezone,timedelta
ROOT=Path(__file__).resolve().parents[1];OUT=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert read(OUT/'status.json')['stage']=='failed' and read(OUT/'preflight.json')['passed']
for item in [read(OUT/'executor.json'),read(OUT/'study.py.started.json')]:
    try:assert psutil.Process(item['pid']).create_time()!=item['birth'],'Old executor still alive'
    except psutil.NoSuchProcess:pass
for n,h in read(OUT/'protocol.json')['sources'].items():assert sha(ROOT/n)==h,n
assert not (OUT/'predictions.json').exists() and not (OUT/'summary.json').exists()
guard=ROOT/'runs/flyvis_repairs_20261106/unification_20261107/uncapped_checks/39f12cdf1ac9459c844b5d3b42bfe3e0'
logs=list(guard.glob('*.worker.log'));assert len(logs)==1
log=logs[0].read_text();assert 'SCALE_FOLD 4 /20' in log and "Fixed Newton iteration limit reached" in log
now=datetime.now(timezone(timedelta(hours=3)))
record=dict(registered=now.isoformat(),failed_guard='39f12cdf1ac9459c844b5d3b42bfe3e0',old_state=read(OUT/'status.json'),old_preflight_reused=read(OUT/'preflight.json'),
    failure='At least24 new fits completed but were held only in worker memory; their weights were not persisted before the numerical failure. No complete predictions/summary exists. They cannot be recovered and must be regenerated; no saved old model, map, activity or result is recomputed.',
    correction='Keep the same objective and lambda, data, folds, input scales, threshold and independent maximum gradient tolerance1e-7. Existing solver already accepts residual<=1e-7 after a precision-limited line search; the recovery accepts the exact final100-iteration state only if the SAME tolerance is met, otherwise fails. No production module edit, extra iterations or hyperparameter search. Every completed model is now atomically checkpointed and input-hashed.',
    mathematical_scope='The correction changes when a numerically stationary result is returned, not the function minimized. It does not guarantee reaching tolerance. The original independent gradient/loss/deployment audit runs unchanged and must pass.',
    sources={p.relative_to(ROOT).as_posix():sha(p) for p in [OUT/'recover_numerics.py',OUT/'recovery_queue.py',OUT/'register_recovery.py',logs[0],OUT/'preflight.json',OUT/'protocol.json']},goal_achieved=False)
with (OUT/'recovery_protocol.json').open('x',encoding='utf8') as f:json.dump(record,f,ensure_ascii=False,indent=2)
path=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';before=path.read_bytes();nl=b'\r\n' if b'\r\n' in before else b'\n';marker='## Журнал изменений'.encode()+nl
entry='**'+now.strftime('%d.%m.%Y, %H:%M MSK')+'** — Численная остановка сравнения шкал и восстановление сохранения\n\n*Факт.* Предварительные58тестов и граф прошли. Новый решатель остановился на пределе100итераций после четырёх полных обучающих исключений; завершённые минимум24новыхмодели не были сохранены из памяти. Итоговых оценок нет. Старые готовые модели и карты не затронуты. Источник research_signal_scaling_20261109/recovery_protocol.json и сохранённый guard39f12cdf1ac9459c844b5d3b42bfe3e0.\n\n*Исправление исполнения.* Не менять научную задачу, порог, штраф, данные и допуск независимой невязки1e−7. На пределе итераций принимать только состояние в томже допуске, который прежний решатель уже разрешает при ограничении точности поиска шага. Иначе снова остановка. Добавлено сохранение каждой модели; несохранённые новые веса приходится восстановить, готовые расчёты не повторяются. Исходная версия и журналы ошибки сохраняются, productionкод не меняется. Все120моделей подлежат прежнему независимому аудиту.\n\n'
assert before.count(marker)==1;data=before.replace(marker,marker+nl+entry.replace('\n',nl.decode()).encode(),1);temp=path.with_suffix('.scale_recovery.pending');temp.write_bytes(data);assert path.read_bytes()==before;temp.replace(path)
print('SCALING_NUMERICAL_RECOVERY_REGISTERED',flush=True)
