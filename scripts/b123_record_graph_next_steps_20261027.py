"""Записать порученные автором содержательные пункты обзора после проверки копии."""
from pathlib import Path
from datetime import datetime
import json
from b123_safe_journal_20261027 import journal

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT/'runs/b123_20261027'
VAULT = ROOT/'ObsidianFlyVis'
manifest = json.loads((VAULT/'.graph-visible.json').read_text(encoding='utf-8'))
titles = ['Обучить отбор полезных продолжений потерянных объектов',
          'Определить, какие ошибки ограничивают HOTA']
for title in titles:
    name = 'Обзор проекта/Запланировано/'+title+'.md'
    assert name in manifest['files'] and (VAULT/name).is_file()
audits = sorted(RUN.glob('graph_sections_restored_*/audit.json'))
audit_path = audits[-1]
audit = json.loads(audit_path.read_text(encoding='utf-8'))
assert audit['passed'] and audit['note_count'] == 42
assert audit['original_files_preserved'] and audit['protected_hypotheses_byte_identical']
assert audit['deleted_files'] == audit['moved_files'] == 0
assert json.loads((RUN/'final_stop/final_verification.json').read_text())['passed']
text = ('По прямому вопросу автора проверено содержание графа: обучаемый отказ уже описывался в задаче восстановления, '
        'но отдельного пункта поиска резерва разных ошибок HOTA не было. Через генератор добавлены два фундаментальных '
        'пункта раздела «Запланировано»: «Обучить отбор полезных продолжений потерянных объектов» и '
        '«Определить, какие ошибки ограничивают HOTA». Первый отделяет полезный диагностический отбор от ещё не проверенной '
        'обучаемости; второй описывает будущие сопоставимые вмешательства с разметкой в ошибки детектора, сопоставления '
        'и восстановления после перекрытий. Диагностика не выдаётся за достижимую прибавку, взаимодействующие вклады не складываются автоматически. '
        'В задаче4 явно записано: непреодолимый предел проекта не установлен, выходные адаптации сохраняли назначения '
        'и состояние исходного трекера, следующий крупный шаг выбирается по измеренному резерву ошибок. '
        'Новых научных результатов, запусков или протоколов эта запись не создаёт; оба пункта пока непроверенные планы. '
        'Проверки генератора:5passed. Сначала проверена копия, затем основное хранилище; источник '+audit_path.relative_to(ROOT).as_posix()+'. '
        'Видимых заметок42, удалений и перемещений0, прежние файлы сохранены,10заметок дерева гипотез побайтно неизменны. '
        'Каталог экспериментов не менялся; основной базис не менялся. Коммит и push не выполнены: прежний .git/index.lock оставлен нетронутым. '
        'Ночная работа остановлена по прежнему сроку: deadline_result.json/all_owned_ended=true; '
        'final_stop/final_verification.json независимо проверил отсутствие собственных процессов и guard, статус автоматизации PAUSED. '
        'После07:30 выполнялось только прямо порученное оформление графа, без продолжения научных расчётов.')
journal('b123_graph_next_steps_20261027', 'В графе записаны отбор продолжений и поиск резерва HOTA', text)
with (RUN/'HANDOFF.md').open('a', encoding='utf-8') as stream:
    stream.write('\n\n## '+datetime.now().strftime('%H:%M')+' МСК — остановка подтверждена; по поручению автора дополнен граф\n\n'+text+'\n')
print(json.dumps(dict(saved=True, graph_audit=audit_path.relative_to(ROOT).as_posix(), visible=42, scientific_runs_started=False)))
