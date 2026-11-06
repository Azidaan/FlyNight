"""Create a narrow artifact allowlist; never stage files or copy private sources."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'runs/flyvis_repairs_20261106/temporal_heads_bundle_v2'
manifest = json.loads((BASE / 'manifest.json').read_text(encoding='utf-8'))

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

assert len(manifest['changes']) == 5
for row in manifest['changes']:
    assert sha(ROOT / row['path']) == row['after_sha256'], row['path']
for row in manifest['dependencies']:
    assert sha(ROOT / row['path']) == row['sha256'], row['path']
assert all(manifest['verification'][key] for key in (
    'no_git_repository_discovered', 'forward_equals_after_exact',
    'reverse_equals_baseline_exact', 'production_files_unchanged',
    'index_and_git_history_unchanged'))

readme = '''1. Состав и статус.

Этот пакет заменяет отозванный temporal_heads_bundle. Старый пакет сохранён локально: после его создания независимая проверка обнаружила дефект новой временной ветви Б1. Отказ вычислителя мог публиковать NaN-состояние и продвигать часы, препятствуя повтору кадра. Исправление, отрицательные проверки до него и независимое повторение после него включены в доказательства V2.

manifest.json описывает пять изменений: поправку существующего незарегистрированного пользовательского tracking/learned_recovery_admission.py и четыре новых файла агента — tracking/b1_timestamped.py, tests/test_b1_timestamped.py, scripts/b1_timestamped_smoke_20261106.py, tests/test_learned_head_contracts.py. Полные снимки пользовательских исходников и зависимостей не входят в список публикации. tracking/reliability_head.py и отдельный tests/test_reliability_head_contracts.py передаются владельцу аудита для отдельного исходного коммита и не включены в эти пять поправок.

2. Применение и происхождение.

Локальные baseline/ и after/ сохраняют точные байты; отдельные patches/ и combined.patch передают изменения. Проверка выполнена в изолированной копии без обнаруженного Git-репозитория: git apply --check, прямое и обратное применение. Для процесса заданы core.autocrlf=false и core.eol=lf. Прямое применение побайтно равно after; обратное восстанавливает исходный пользовательский файл и удаляет только четыре созданных в проверочной копии новых файла. Production-файлы, индекс и история Git этим скриптом не изменялись.

Статическая цепочка локальных импортов с SHA256 сохранена в manifest.dependencies и в локальном dependencies/. Это происхождение среды проверки, не заявление, что весь прежний пользовательский код принадлежит агенту. Опциональные импорты учитываются консервативно; динамические импорты, установленные пакеты, внешние веса и данные автоматически не закрыты. Публикуемая поправка требует точного baseline SHA256 из manifest; чистый клон без этих зависимостей не заявлен самостоятельной средой воспроизведения.

3. Доказательства для нового пути Б1.

TimestampedBackbone проверяет полное состояние ядра FlyVis, состояние контроля B или R1 и выходные каналы до возврата. Испорченные состояния не публикуются вместе с часами. Отдельная общая проверка обучения и исполнения отвергает нечисловую поправку головы и вероятность реального кандидата. Штатные вычислители создают новые состояния; произвольное изменение переданного объекта на месте не откатывается. Атомарность advance_at не означает откат всей пары advance_at → correction или всего Tracker.update при последующем отказе головы.

Две группы дополнительных проверок до ремонта дали соответственно 12 и 2 отказа. Совместный набор после ремонта и разделения тестов дал 253 passed; самостоятельный набор reliability дал 17 passed. Перенос всех тестовых функций проверен по AST (структурному представлению исходного кода), включая декораторы и условия проверок. Адреса журналов приведены в manifest.evidence.

Независимый повтор на настоящем CPU ReceptorHistory подтвердил: NaN, неправильная форма и исключение вычислителя оставляют ссылки, значения eye и часы неизменными; повтор кадра и следующий кадр точно совпадают с исправным историческим контролем. Проверены отказ начального состояния и повтор инициализации. Это узкая проверка границы исполнения, а не доказательство всех возможных отказов модели.

real_D_v5 выполнил два технических CPU-шага с настоящим ядром FlyVis, восстановил намеренно изменённые веса и состояние оптимизатора и подтвердил равенство пакетного и последовательного исполнения. outputs.npz побайтно совпадает с real_D_v3 и real_D_v4: SHA256 ca2eeb9357e838e1ad4cfd452e1087ddcb91995c3e5a01a5a3290ea96c741d22. Независимый пересчёт сохранённых массивов и времён находится в independent_validation_v5.json. Протокол V5 закрепляет фактические исходники того запуска; manifest этого пакета отдельно закрепляет финальные зависимости рабочего дерева.

После V5 другой участник исправил только start, advance и проверку состояния исторического ResidualEngine в b1_tracker_runtime.py. Новый TimestampedResidualEngine его не вызывает; импортируемый B1Tracker и прочие части файла сохранены по AST. Финальный совместный набор после этой правки снова дал 253 passed. Отдельная независимая CPU-проверка исторического исполнителя находится в historical_residual_independent/report.json. Полная поправка исторического файла передаётся владельцем отдельно и не скрыта среди пяти изменений этого пакета.

4. Обученные головы и научная граница.

Все шесть реальных сохранённых моделей допуска восстановления и их нормировки валидны. На 64 реальных строках для каждой модели исправленный CPU-выход точно равен прежнему CPU-выходу. Оба настоящих JSON считывателя надёжности и их обёртки также дали точное равенство. Нарушенные веса, нулевые масштабы и иные ошибочные параметры воспроизводились на синтетических копиях; исторические файлы не переписывались. Первичные таблицы и массивы находятся в learned_head_review.

Новый временной путь Б1 требует marker b1_observed_frame_time_v1 и отдельного обучения с корректным временем. Добавление marker к старым весам не делает их совместимыми. Технический checkpoint real_D_v5 имеет diagnostic_only и обычным исполнителем отвергается. Полное научное обучение, GPU и HOTA здесь не запускались. Пакет не доказывает улучшение трекинга или гипотезу о пользе ядра FlyVis. Обычный режим геометрии остаётся legacy_v1; pixel_centers_v2 и resize_v2 доступны только как явно выбранные исследовательские режимы.
'''
assert not (BASE / 'README.txt').exists()
(BASE / 'README.txt').write_text(readme, encoding='utf-8')

paths = [
    BASE / 'README.txt', BASE / 'manifest.json', BASE / 'combined.patch',
    *sorted((BASE / 'patches').glob('*.patch')),
    ROOT / 'runs/flyvis_repairs_20261106/build_temporal_heads_bundle_v2.py',
    Path(__file__).resolve(),
    ROOT / 'runs/flyvis_repairs_20261106/temporal_heads_bundle_v2_build.log',
]
relative = [
    'runs/flyvis_repairs_20261106/temporal_heads_bundle/PUBLICATION_WITHDRAWN.txt',
    'runs/flyvis_repairs_20261106/b1_finite_transition/before_manifest.json',
    'runs/flyvis_repairs_20261106/b1_finite_transition/before_tests.log',
    'runs/flyvis_repairs_20261106/b1_finite_transition/head_boundary_before.log',
    'runs/flyvis_repairs_20261106/b1_finite_transition/after_tests_final.log',
    'runs/flyvis_repairs_20261106/b1_finite_transition/tests_joint_final.log',
    'runs/flyvis_repairs_20261106/b1_finite_transition/REPORT.txt',
    'runs/flyvis_repairs_20261106/head_test_split/report.json',
    'runs/flyvis_repairs_20261106/head_test_split/tests_joint_after_split.log',
    'runs/flyvis_repairs_20261106/head_test_split/reliability_standalone.log',
    'runs/flyvis_repairs_20261106/head_test_split/tests_final_runtime_dependency.log',
    'runs/flyvis_repairs_20261106/historical_residual_independent/review.py',
    'runs/flyvis_repairs_20261106/historical_residual_independent/report.json',
    'runs/flyvis_repairs_20261106/historical_residual_independent/review.log',
    'runs/flyvis_repairs_20261106/learned_head_review/REPORT.txt',
    'runs/flyvis_repairs_20261106/learned_head_review/probe.py',
    'runs/flyvis_repairs_20261106/learned_head_review/report.json',
    'runs/flyvis_repairs_20261106/learned_head_review/probe.log',
    'runs/flyvis_repairs_20261106/learned_head_review/before_tests.log',
    'runs/flyvis_repairs_20261106/learned_head_review/after_tests_final_v2.log',
    'runs/flyvis_repairs_20261106/learned_head_review/replay_after.py',
    'runs/flyvis_repairs_20261106/learned_head_review/after_report.json',
    'runs/flyvis_repairs_20261106/learned_head_review/replay_after_final.log',
    'runs/flyvis_repairs_20261106/learned_head_review/replayed_outputs.npz',
    'runs/flyvis_system_fixes_20261106/b1_engine_review/before_report.json',
    'runs/flyvis_system_fixes_20261106/b1_engine_review/before_manifest.json',
    'runs/flyvis_system_fixes_20261106/b1_engine_review/probe_cpu.py',
    'runs/flyvis_system_fixes_20261106/b1_engine_review/before.log',
    'runs/flyvis_system_fixes_20261106/b1_engine_review/probe_timestamped_after.py',
    'runs/flyvis_system_fixes_20261106/b1_engine_review/timestamped_after_report.json',
    'runs/flyvis_system_fixes_20261106/b1_engine_review/timestamped_after.log',
    'runs/flyvis_system_fixes_20261106/b1_engine_review/TIMESTAMPED_INDEPENDENT_REVIEW.txt',
    'runs/flyvis_system_fixes_20261106/b1_engine_review/timestamped_review_manifest.json',
    'runs/flyvis_timestamp_repair_20261106/README.txt',
    'runs/flyvis_timestamp_repair_20261106/real_D_v5/protocol.json',
    'runs/flyvis_timestamp_repair_20261106/real_D_v5/report.json',
    'runs/flyvis_timestamp_repair_20261106/real_D_v5/runtime_binding.json',
    'runs/flyvis_timestamp_repair_20261106/real_D_v5/outputs.npz',
    'runs/flyvis_timestamp_repair_20261106/real_D_v5.log',
    'runs/flyvis_timestamp_repair_20261106/validate_saved_v5.py',
    'runs/flyvis_timestamp_repair_20261106/independent_validation_v5.json',
    'runs/flyvis_timestamp_repair_20261106/validate_saved_v5.log',
]
paths.extend(ROOT / name for name in relative)
names = [path.relative_to(ROOT).as_posix() for path in paths]
assert len(names) == len(set(names))
for path in paths:
    assert path.is_file(), path
    assert not {'original', 'baseline', 'after', 'dependencies', 'isolated_apply_check'}.intersection(path.parts), path
    assert path.suffix not in {'.pt', '.pth', '.ckpt'}, path
rows = [dict(path=name, size=path.stat().st_size, sha256=sha(path)) for name, path in zip(names, paths)]
publication = dict(
    scope='Narrow publication allowlist. Prior-untracked user source changes are patches, not first-time source commits.',
    artifacts=rows, total_bytes=sum(row['size'] for row in rows),
    excluded=['full dependency snapshots', 'full after/baseline/original source snapshots',
              'isolated apply workspace', 'diagnostic and historical checkpoints',
              'synthetic malformed checkpoint files', 'source data archives'],
    local_reproduction_note='Exact local baseline and dependency hashes are required. A clean checkout without those dependencies is not claimed to reproduce every probe.',
)
(BASE / 'publication_manifest.json').write_text(json.dumps(publication, ensure_ascii=False, indent=2), encoding='utf-8')
names.extend((BASE / name).relative_to(ROOT).as_posix() for name in ('publication_manifest.json', 'artifact_paths.txt'))
(BASE / 'artifact_paths.txt').write_text('\n'.join(names) + '\n', encoding='utf-8')
print(json.dumps(dict(artifacts=len(names), total_bytes=publication['total_bytes'],
    combined_patch_sha256=manifest['combined_patch_sha256'], manifest_sha256=sha(BASE / 'manifest.json'),
    publication_manifest_sha256=sha(BASE / 'publication_manifest.json')), indent=2))
