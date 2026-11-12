from common import *
import shutil
assert not (OUT/'replay.json').exists() and not (OUT/'status.json').exists()
shutil.copy2(OUT/'protocol.json',OUT/'protocol_registered_v1.json')
p=read(OUT/'protocol.json');p['implementation_correction_before_replay']=dict(date=now(),reason='Review found inherited secondary gate required either own232 or own832, while the registered secondary criterion requires both. Corrected only secondary implementation in separate evaluator; criteria unchanged and no new outputs evaluated.',evaluator='audit_secondary_v2.py')
for n in ['audit_secondary_v2.py','preflight_correction.py','protocol_registered_v1.json']:p['inputs'][(OUT/n).relative_to(ROOT).as_posix()]=sha(OUT/n)
(OUT/'protocol.json').write_text(json.dumps(p,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
journal('До запуска `'+ID+'` исправлена реализация вторичного критерия в отдельном audit_secondary_v2.py: требуются обе свои232/832, как уже записано в протоколе. Унаследованное выражение требовало хотя бы одну. Исходный протокол и первый оценщик сохранены; критерии и данные не изменены, новых выходов ещё нет.')
print('SECONDARY_IMPLEMENTATION_FIXED_BEFORE_REPLAY')
