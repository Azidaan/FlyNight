"""Recover an incomplete whitelist, retain the failed assertion and all outputs."""
from common import *
start('cpu',timeout=600)
import psutil
phase=read(OUT/'execution_protocol.json');verify(phase)
old=ROOT/'runs/flyvis_repairs_20261106/unification_20261107/uncapped_checks/549bc477492b4836abd81f7e147a18b2'
guard=read(old/'memory_guard.json');assert guard['returncode']==1 and guard['reason'] is None
for pid,birth in guard['owned'].items():
    try:assert psutil.Process(int(pid)).create_time()!=birth
    except psutil.NoSuchProcess:pass
name='runs/identity_assignment_20261108/full_precision/features/b251064f-5f6b663e/62ed1c6c0da2446c7f7a401390bf9b0e.C.npy'
receipt=LEGACY/'rollouts/G_PERM/b251064f-5f6b663e.receipt.json'
assert read(receipt)['passed'] and read(receipt)['feature_files'][name]==sha(ROOT/name)
ready={};artifacts={}
for arm in ['G7','R134_spatial','R134_temporal','C34']:
    for path in (OUT/'rollouts'/arm).glob('*.receipt.json'):
        record=read(path);assert record['passed'] and record['execution_sha256']==sha(OUT/'execution_protocol.json')
        ready[path.relative_to(ROOT).as_posix()]=sha(path)
        for source,digest in record['artifacts'].items():assert sha(ROOT/source)==digest;artifacts[source]=digest
        for source,digest in record['feature_files'].items():assert sha(ROOT/source)==digest;artifacts[source]=digest
files=[OUT/n for n in ['register_cache_registry.py','cache_registry_recovery.py','cache_registry_wrapper.py',
    'cache_registry_test.py','recovery2_queue.py','status_write_recovery.py','status_recovery_protocol.json',
    'native_run.py','native_runtime.py','common.py','execution_protocol.json','deployment.json','zero_complete.json',
    'register_evaluation.py','deployment_audit.py','evaluate_native.py','native_statistics.py','id_fate.py',
    'register_boundary.py','boundary_audit.py']]
files += [old/'memory_guard.json',old/'memory_guard.worker.log',ROOT/name,
    ROOT/'research_identity_assignment_run_20261108/full_planner.py',ROOT/'research_identity_assignment_run_20261108/full_run.py']
files += [LEGACY/'rollouts/G_PERM'/(v+'.receipt.json') for v in read(OUT/'protocol.json')['videos']]
write('cache_registry_protocol.json',dict(registered=datetime.now(MSK).isoformat(),
    failure='AtC34/b251064f-5f6b663e, optional oldC8map existed but was absent from the two registeredproducerreceipts. A CLOSED G_PERM receipt binds its exact actual hash. No mismatch of a declared checksum occurred.',
    evidence='Old full_run uses the same C backbone/map recipe forG_FLYVIS andG_PERM; full_planner applies spatial permutation only AFTER loading reference map, never when savingC.npy. ThusG_PERM is a matching closed cache producer. Hash checks remain mandatory at every load.',
    correction='Extend only FeatureBank.__init__ legacy_sources with same-video CLOSED G_PERM feature_files. Conflicting declared hashes stay fatal. Keep old status-only retry and unchanged scientific source/execution passport. No core/reader/decision/threshold/dtype change.',
    ready_receipts=ready,verified_artifacts=artifacts,failed_optional_C8=name,
    sources={x.relative_to(ROOT).as_posix():sha(x) for x in files},
    resume='Skip all20 G7/all20R134spatial/all20R134temporal and10closedC34 videos. Resume unfinished C34 video from start with saved maps, then remaining videos, unchanged metric and fate stages, and boundary audit.',save_stop=STOP.isoformat()))
journal('Дополнен учёт паспортов готовых карт контрольнымG_PERM',
    'НаC34/b251064f-5f6b663e после10готовыхвидео защита остановилачтениестаройC8карты62ed1c6c0da2446c7f7a401390bf9b0e: еёпаспорта не было всобранныхG_R1/G_FLYVIS. Найден закрытыйG_PERMс точнойконтрольнойсуммой47742663e333573f127e501443c45e5321dfa2efee4ec273e41d1c0afaa5b59f; в старомкоде перестановкаидётпослечтения,а нейроннаякарта сохраняетсятемжерецептом. Ошибочнаяочередь/guard549bc477492b4836abd81f7e147a18b2сохранены. Исправлен толькоучётзакрытыхисточников, защита отнесовпадениясуммынеослабляется. Все3контроля20видео и10C34проверены/переиспользуются; научныеисточникине меняются. Паспорт `runs/full_template_native_20261109/cache_registry_protocol.json`.')
print('Cache registry correction registered',len(ready),flush=True)
