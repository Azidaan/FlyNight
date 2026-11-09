"""Complete only the missing journal append after metadata was saved."""
from geometry import *
from record import journal_atomic
assert (ROOT/'research_identity_synthesis_20261109/СВОДКА_v8.md').exists()
data=read(ROOT/'experiments/system_problems.json');p=next(r for r in data['problems'] if r['id']=='С2')
assert 'СВОДКА_v8.md' in p['fact']
journal_atomic('Накопленное заключение собрано в одну текущую версию', 'Сформирована `research_identity_synthesis_20261109/СВОДКА_v8.md`: текущая проблема, доказанный механизм опоры, последний сопоставимый результат, отдельные ограничения общего назначения и обязательный этап реальных ID. Прежние версии1–7 сохранены. Исправлены пробелы в текущем поле карточки; старый текст перенесён в revisions. Научные данные, критерии, параметры и статусы не менялись; расчёты не повторялись. Завершающая запись журнала отдельно восстановлена после моей ошибки импорта journal_atomic: остальные файлы уже были сохранены, поэтому повторной записи каталога не было. Исходник и выход1 сохранены.')
print('Only the missing journal entry completed.',flush=True)
