"""Hash an explicit final evidence allowlist; never stage or change production."""
from pathlib import Path
from collections import defaultdict
import hashlib
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
BASE = ROOT/'runs/flyvis_repairs_20261106'
assert not (OUT/'manifest.json').exists() or '--refresh-draft' in sys.argv, 'Preserve existing final package'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def info(path):
    return dict(path=path.relative_to(ROOT).as_posix(), size=path.stat().st_size, sha256=sha(path))


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.write_bytes((json.dumps(value, ensure_ascii=False, indent=2)+'\n').encode('utf-8'))


def git(*args, input=None):
    return subprocess.check_output(['git',*args], cwd=ROOT, input=input)


head = git('rev-parse','HEAD').decode().strip()
tracked_status_before = git('diff','--name-only','-z')
index_before = git('ls-files','--stage','-z')
root_attributes_before = (ROOT/'.gitattributes').read_bytes()


members = defaultdict(set)
bundles = []
for name, count in [('lifecycle_patch_bundle',13),('b2_patch_bundle',7),
                    ('temporal_heads_bundle_v2',55),('historical_residual_patch_bundle_v2',22)]:
    listing = BASE/name/'artifact_paths.txt'
    paths = listing.read_text(encoding='utf-8').splitlines()
    assert len(paths) == len(set(paths)) == count
    for relative in paths:
        path = (ROOT/relative).resolve()
        path.relative_to(ROOT.resolve())
        assert path.is_file()
        members[path.relative_to(ROOT).as_posix()].add(name)
    bundles.append(dict(name=name, count=count, listing=info(listing), manifest=info(BASE/name/'manifest.json')))

targets = []
lifecycle = read(BASE/'lifecycle_patch_bundle/manifest.json')
for item in lifecycle['files']:
    targets.append(dict(bundle='lifecycle_patch_bundle', path=item['path'], expected=item['current_sha256']))
    assert sha(ROOT/item['patch']) == item['patch_sha256']
b2 = read(BASE/'b2_patch_bundle/manifest.json')
for key in ('source','new_test'):
    item = b2[key]
    targets.append(dict(bundle='b2_patch_bundle', path=item['path'], expected=item['sha256']))
for item in b2['patches']:
    assert sha(ROOT/item['path']) == item['sha256']
temporal = read(BASE/'temporal_heads_bundle_v2/manifest.json')
for item in temporal['changes']:
    targets.append(dict(bundle='temporal_heads_bundle_v2', path=item['path'], expected=item['after_sha256']))
    assert sha(BASE/'temporal_heads_bundle_v2'/item['patch']) == item['patch_sha256']
assert sha(BASE/'temporal_heads_bundle_v2/combined.patch') == temporal['combined_patch_sha256']
for item in read(BASE/'temporal_heads_bundle_v2/publication_manifest.json')['artifacts']:
    assert sha(ROOT/item['path']) == item['sha256']
historical = read(BASE/'historical_residual_patch_bundle_v2/manifest.json')
for item in historical['changes']:
    targets.append(dict(bundle='historical_residual_patch_bundle_v2', path=item['target'], expected=item['after_sha256']))
for item in historical['package_files']:
    assert sha(ROOT/item['path']) == item['sha256']
assert len(targets) == 20
for item in targets:
    item['current'] = sha(ROOT/item['path'])
    item['exact'] = item['current'] == item['expected']
    assert item['exact'], item

extras = {
    'integrated_cpu_suite': [f'runs/flyvis_repairs_20261106/integrated_1793996598125382200/{name}'
                            for name in ('command.json','result.json','runtime.json','tests.log')],
    'lifecycle_parity_and_failures': [f'runs/flyvis_repairs_20261106/{name}' for name in
        ('lifecycle_before.log','lifecycle_after.log','lifecycle_after_v2.log','lifecycle_parity.json','lifecycle_parity_outputs.npz')],
    'b2_lifecycle_evidence': [f'runs/flyvis_system_fixes_20261106/{name}' for name in
        ('b2_lifecycle_before.log','b2_lifecycle_after.log','loading_lifecycle_review_tests.log','loading_prior_untracked_b2_manifest.json')],
    'final_report_and_branch_decision': [f'runs/flyvis_repairs_20261106/{name}' for name in
        ('FINAL_AUDIT_REPORT.txt','gitlab_branch_decision.json','gitlab_branch_protection_20261106.png')],
}
for group, paths in extras.items():
    for relative in paths:
        assert (ROOT/relative).is_file(), relative
        members[relative].add(group)

ci_manifest_path = BASE/'gitlab_ci_review/manifest_v2.json'
ci = read(ci_manifest_path)
ci_excluded = []
for item in ci['artifacts']:
    assert sha(ROOT/item['path']) == item['sha256']
    if Path(item['path']).name == 'original_ci.yml':
        ci_excluded.append(dict(**item, reason='Full preexisting CI source snapshot; original configuration remains in tracked .gitlab-ci.yml'))
        continue
    members[item['path']].add('gitlab_ci_review_v2')
members[ci_manifest_path.relative_to(ROOT).as_posix()].add('gitlab_ci_review_v2')

audit = ROOT/'runs/flyvis_system_audit_20261106'
tracked_audit = subprocess.run(['git','ls-files','--','runs/flyvis_system_audit_20261106'],
    cwd=ROOT, capture_output=True, text=True, check=True).stdout.splitlines()
tracked_set = set(tracked_audit)
audit_paths = sorted(p for p in audit.iterdir() if p.is_file())
assert len(audit_paths) == 19
audit_added = []
for path in audit_paths:
    assert path.suffix in ('.json','.txt','.log','.py')
    relative = path.relative_to(ROOT).as_posix()
    if relative not in tracked_set:
        members[relative].add('initial_weights_connections_audit')
        audit_added.append(relative)
for name, expected in read(audit/'manifest.json')['files'].items():
    assert sha(audit/name) == expected, 'Initial audit evidence drift: '+name

integrated = BASE/'integrated_1793996598125382200'
command = read(integrated/'command.json')
result = read(integrated/'result.json')
runtime = read(integrated/'runtime.json')
assert result['passed'] and result['concurrently_changed_sources'] == []
assert not runtime['cuda_initialized_before'] and not runtime['cuda_initialized_after']
integrated_changes = [name for name, expected in command['source_sha256'].items()
                      if not (ROOT/name).is_file() or sha(ROOT/name) != expected]
assert integrated_changes == [], integrated_changes
script = ROOT/'scripts/run_system_fix_checks_20261106.py'
assert sha(script) == command['source_sha256'][script.relative_to(ROOT).as_posix()]
(OUT/'source_paths.txt').write_bytes(b'scripts/run_system_fix_checks_20261106.py\n')

geometry_references = [
    'runs/flyvis_system_fixes_20261106/calibration_guard/FINAL_DECISION.txt',
    'runs/flyvis_repairs_20261106/geometry_calibration_counterevidence.json',
    'runs/flyvis_repairs_20261106/coordinate_only_roi/protocol_before_run.json',
    'runs/flyvis_repairs_20261106/coordinate_only_roi/report.json',
    'runs/flyvis_repairs_20261106/coordinate_only_roi_review/report.json',
]
tracked_geometry = subprocess.run(['git','ls-files','--',*geometry_references], cwd=ROOT,
    capture_output=True, text=True, check=True).stdout.splitlines()
assert set(tracked_geometry) == set(geometry_references)
verification = dict(patch_bundles=bundles, current_patch_target_checks=targets,
    all_twenty_patch_targets_exact=True,
    integrated_source_count=len(command['source_sha256']), integrated_current_mismatches=integrated_changes,
    integrated_result=result, integrated_runtime=runtime,
    initial_audit_already_tracked=tracked_audit, initial_audit_added=audit_added,
    initial_audit_manifest_entries_verified=len(read(audit/'manifest.json')['files']),
    geometry_correction_already_tracked=[info(ROOT/p) for p in geometry_references],
    ci_review_manifest=info(ci_manifest_path), ci_source_snapshot_excluded=ci_excluded,
    b2_evidence_boundary='Existing test_interleaved_independent_tracker_matches_native_baseline_exactly compares actual XYWHByteTrack on60synthetic frames. No separate B2 native-core report exists or is claimed.',
    no_training_or_gpu_or_test_execution=True, no_index_or_production_changes=True)
write(OUT/'verification.json', verification)

readme = '''1. Граница итогового пакета

Кандидаты artifact_paths.txt получены из четырех согласованных наборов patches (13+7+55+22 файлов), первичных результатов общей CPU-проверки, необходимых проверок жизненного цикла трекеров и первоначального аудита весов/связей. Из финального списка исключены уже tracked и неизменные относительно HEAD артефакты: их ссылки, контрольные суммы и Git blob сохранены отдельно как referenced_artifacts в manifest.json. Такое исключение не удаляет доказательство, а исключает его повторное представление как нового изменения. Новые служебные файлы этого пакета включены явно. source_paths.txt содержит ровно один новый авторский сценарий scripts/run_system_fix_checks_20261106.py. Изменения ранее неотслеживаемых пользовательских исходников передаются patches из четырех наборов.

Не включены рабочий план, каталог/словарь экспериментов, parent_baseline.zip, полные исходные пользовательские файлы, dependency snapshots, веса и контрольные точки, временные каталоги pytest, local-only actions, а также незапущенный scripts/record_system_integrity_final_20261106.py. Полная копия original_ci.yml также не включена; исходная конфигурация уже имеется как tracked .gitlab-ci.yml. Пакет только подготовлен: индекс и история Git не изменялись. Наличие файла в списке не означает, что он уже опубликован; последующий шаг записи журнала и публикации выполняет root отдельно.

2. Проверка связности доказательств

verification.json подтверждает точное совпадение текущих SHA256 всех20 файлов, охваченных четырьмя наборами patches, с их конечными хешами, а также совпадение всех1429 исходников из command.json общего прогона. Общая проверка integrated_1793996598125382200:747passed/2skipped,48файлов тестов, исходники во время прогона неизменны; CUDA до и после не инициализирована. Включены только command.json, result.json, runtime.json и tests.log; временные файлы тестов не включены.

Для жизненного цикла включены исходный отказ, успешные повторные проверки и lifecycle_parity.json с массивами выхода. Для B2 включены существующие журналы и manifest, дополняющие b2_patch_bundle. Отдельного B2 native-core report нет: test_interleaved_independent_tracker_matches_native_baseline_exactly из b2_lifecycle_test.patch сравнивает настоящий XYWHByteTrack на60кадрах синтетических детекций. Это проверка сохранения поведения трекера, не прогон ядра FlyVis и не научная метрика. Ранний loading_prior_untracked_b2_manifest.json сохранен как происхождение; актуальное применение определяют b2_patch_bundle/manifest.json и apply_report.json.

3. Первоначальный аудит весов и связей; обязательная поправка геометрического вывода

Первоначальные core_report.json, additional_report.json, boundary_report.json, saved_evidence.json, AUDIT_REPORT.txt, audit*.py, run_checks.py и журналы из runs/flyvis_system_audit_20261106 прежде не входили в Git и добавлены узко, без pytest_tmp. Их17 исходных контрольных сумм совпадают с собственным историческим manifest. Они сохраняют доказательства проверки загрузки весов, индексов связей и численной динамики, но описывают реализацию до исправлений. Старый core_report честно содержит cuda_initialized=true из-за прежнего CPU-синхронизатора; новые CPU-проверки с cuda_initialized=false находятся в последующих пакетах. Скрипты первоначального аудита не запускались при сборке пакета; повтор на измененных исходниках не заменяет сохраненный исторический результат.

Вывод AUDIT_REPORT.txt, раздел3, о подтвержденной ошибке горизонтального масштаба обученного декодера ОТОЗВАН. Формальная проверка обратного геометрического преобразования не доказала, что обученный выход обязан ему следовать. Исправление этого вывода и окончательное сохранение legacy_v1 описаны в уже опубликованном runs/flyvis_system_fixes_20261106/calibration_guard/FINAL_DECISION.txt. Контрдоказательство калибровки: runs/flyvis_repairs_20261106/geometry_calibration_counterevidence.json. Отрицательный заранее заданный гейт изменения только координат: runs/flyvis_repairs_20261106/coordinate_only_roi/protocol_before_run.json и report.json; независимое подтверждение: runs/flyvis_repairs_20261106/coordinate_only_roi_review/report.json. Эти пять источников уже tracked в проверенном пакете2; их хеши сохранены в verification.json. Историческое предложение из initialaudit нельзя читать как действующую рекомендацию или доказанное улучшение.

4. Воспроизводимость и дальнейшая публикация

В manifest.json перечислены SHA256 всех включенных payload artifacts и единственного source path. При сохранении следует использовать только artifact_paths.txt и source_paths.txt, без рекурсивного добавления каталогов. Новый runs/.gitattributes задает точные адресные правила -text только для каждого файла artifact_paths, включая себя и метаданные пакета; широких шаблонов и правил для already-tracked references нет. Это необходимо: первоначальная проверка обнаружила преобразование байтов двух patches и80артефактов настройкой Git. patch_byte_publication_check.json фиксирует последующую проверку raw==filtered для каждого публикуемого артефакта. Корневой .gitattributes и Git config не изменены; обычные правила единственного нового source script сохранены.

Инструкции точного применения и отмены находятся в собственных manifest/README четырех patch bundles; перед применением требуются исходные SHA256. В историческом B1 bundle явный -c core.autocrlf=false предотвращает подмену окончаний строк. Часть проверок требует имеющегося локального проекта и его исходных данных; полная автономная поставка зависимостей не заявлена.

FINAL_AUDIT_REPORT.txt связывает технические итоги и незакрытые ограничения. gitlab_branch_decision.json и gitlab_branch_protection_20261106.png фиксируют последнее решение автора по веткам. gitlab_ci_review/report_v2.json, manifest_v2.json и два сохраненных текста интерфейса документируют подтвержденное требование проверки личности до запуска CI; прежние ответы API и исходная неопределенность отчета сохранены. При подготовке этого пакета новые вычислительные прогоны, обучение, работа GPU и операции записи Git не выполнялись. Технические проверки не устанавливают улучшение HOTA и не доказывают отсутствие всех ошибок авторской системы.
'''
(OUT/'README.txt').write_bytes(readme.encode('utf-8'))
for name in ('build_manifest.py','source_paths.txt','verification.json','README.txt','patch_byte_publication_check.json'):
    members[(OUT/name).relative_to(ROOT).as_posix()].add('final_package_metadata')

for relative in members:
    path = ROOT/relative
    assert path.is_file()
    assert relative.startswith('runs/'), relative
    assert path.suffix.lower() not in ('.pt','.pth','.h5','.hdf5','.zip','.7z','.pkl','.pickle')
    parts = set(Path(relative).parts)
    assert not parts.intersection({'temp','pytest_tmp','original','baseline','after','snapshots','weights','localonlyactions'})
    assert 'record_system_integrity_final_20261106.py' not in relative
    assert 'parent_baseline' not in relative

candidates = sorted(members)
# Use the same Git clean-filter interpretation as verify_staged_allowlist.py.
# A CRLF worktree copy may equal the already tracked LF blob and must not become
# an unchanged path in export_pinned_commit's exact changed-path allowlist.
current_blobs = git('hash-object','--stdin-paths', input=('\n'.join(candidates)+'\n').encode()).decode().splitlines()
head_queries = git('cat-file','--batch-check', input=('\n'.join(head+':'+p for p in candidates)+'\n').encode()).decode().splitlines()
assert len(current_blobs) == len(head_queries) == len(candidates)
payload, referenced = [], []
for relative, current_blob, query in zip(candidates,current_blobs,head_queries):
    fields = query.split()
    previous_blob = fields[0] if len(fields) == 3 and fields[1] == 'blob' else None
    row = dict(**info(ROOT/relative), groups=sorted(members[relative]),
               git_worktree_blob=current_blob, head_blob=previous_blob)
    if previous_blob == current_blob:
        row['publication']='already tracked unchanged; reference only'
        referenced.append(row)
    else:
        row['publication']='new artifact' if previous_blob is None else 'modified owned artifact'
        payload.append(row)
source_blob = git('hash-object','--stdin-paths',input=(script.relative_to(ROOT).as_posix()+'\n').encode()).decode().strip()
source_head = git('cat-file','--batch-check',input=(head+':'+script.relative_to(ROOT).as_posix()+'\n').encode()).decode().strip().split()
assert not (len(source_head)==3 and source_head[1]=='blob' and source_head[0]==source_blob), 'Source is already unchanged in HEAD'

# Root-authorized, exact -text entries for new publication artifacts only.
attribute_path = ROOT/'runs/.gitattributes'
if attribute_path.exists():
    previous_manifest = read(OUT/'manifest.json')
    prior = [x for x in previous_manifest['artifacts'] if x['path']=='runs/.gitattributes']
    assert len(prior)==1 and sha(attribute_path)==prior[0]['sha256'], 'Refuse to replace an unowned attributes file'
publication_names = sorted([*[item['path'] for item in payload], 'runs/.gitattributes',
    (OUT/'manifest.json').relative_to(ROOT).as_posix(), (OUT/'artifact_paths.txt').relative_to(ROOT).as_posix()])
assert len(publication_names)==len(set(publication_names))
rules=[]
for name in publication_names:
    relative=Path(name).relative_to('runs').as_posix()
    assert not any(c in relative for c in '*?[] \\'), relative
    rules.append('/'+relative+' -text')
assert not set(publication_names).intersection(x['path'] for x in referenced)
attribute_path.write_bytes(('\n'.join(rules)+'\n').encode('utf-8'))
payload.append(dict(**info(attribute_path),groups=['byte_exact_publication_rules'],
    git_worktree_blob=git('hash-object','--path=runs/.gitattributes','runs/.gitattributes').decode().strip(),
    head_blob=None, publication='new artifact'))

patch_names=[x['path'] for x in payload if x['path'].endswith('.patch')]
patch_checks=[]
for name in patch_names:
    raw_blob=git('hash-object','--no-filters',name).decode().strip()
    filtered_blob=git('hash-object','--path='+name,name).decode().strip()
    assert raw_blob==filtered_blob, name
    patch_checks.append(dict(path=name,raw_blob=raw_blob,filtered_blob=filtered_blob,exact=True))
existing_names=[name for name in publication_names if (ROOT/name).is_file()]
raw_blobs=git('hash-object','--no-filters','--stdin-paths',input=('\n'.join(existing_names)+'\n').encode()).decode().splitlines()
filtered_blobs=git('hash-object','--stdin-paths',input=('\n'.join(existing_names)+'\n').encode()).decode().splitlines()
assert raw_blobs==filtered_blobs
assert git('diff','--name-only','-z')==tracked_status_before, 'Attributes caused tracked worktree drift'
assert git('ls-files','--stage','-z')==index_before, 'Index changed during preparation'
assert (ROOT/'.gitattributes').read_bytes()==root_attributes_before
byte_report=dict(initial_check=dict(patch_count=21,patch_mismatch_count=2,artifact_mismatch_count=80,
    mismatched_patches=['runs/flyvis_repairs_20261106/temporal_heads_bundle_v2/combined.patch',
                       'runs/flyvis_repairs_20261106/temporal_heads_bundle_v2/patches/tests__test_learned_head_contracts.py.patch']),
    final_patch_checks=patch_checks,all_final_artifact_bytes_exact=True,exact_attribute_rule_count=len(rules),
    root_gitattributes_unchanged=True,tracked_status_unchanged=True,index_unchanged=True,
    initial_findings_preserved='Initial comparison was recorded before the authorized -text rules; no artifact content was normalized to hide the discrepancy.',
    git_config_changed=False)
write(OUT/'patch_byte_publication_check.json',byte_report)
verification['byte_exact_publication']=dict(report=info(OUT/'patch_byte_publication_check.json'),
    attributes=info(attribute_path),root_attributes_sha256=hashlib.sha256(root_attributes_before).hexdigest(),
    tracked_status_digest_before_after=hashlib.sha256(tracked_status_before).hexdigest(),
    index_listing_digest_before_after=hashlib.sha256(index_before).hexdigest())
write(OUT/'verification.json',verification)
for row in payload:
    row.update(info(ROOT/row['path']))
    row['git_worktree_blob']=git('hash-object','--no-filters',row['path']).decode().strip()
manifest = dict(schema='system_integrity_final_evidence_v1', artifacts=payload, sources=[info(script)],
    baseline_head=head, referenced_artifacts=referenced,
    bundle_counts={item['name']:item['count'] for item in bundles},
    extras_counts={group:len(paths) for group,paths in extras.items()}, initial_audit_added_count=len(audit_added),
    verification=info(OUT/'verification.json'), total_payload_bytes=sum(x['size'] for x in payload)+script.stat().st_size,
    self_reference_note='manifest.json and artifact_paths.txt are listed for publication, but not self-hashed inside manifest',
    excluded=['plan/catalogue/glossary','full user source snapshots and original zip','weights/checkpoints','dependency snapshots',
              'pytest temporary files','local-only action dumps','unexecuted final-ledger script'],
    prepared_only=True, index_changed=False, production_changed=False)
write(OUT/'manifest.json', manifest)
listing = sorted([*[item['path'] for item in payload], (OUT/'manifest.json').relative_to(ROOT).as_posix(),
                  (OUT/'artifact_paths.txt').relative_to(ROOT).as_posix()])
assert len(listing) == len(set(listing))
(OUT/'artifact_paths.txt').write_bytes(('\n'.join(listing)+'\n').encode('utf-8'))
# A final read verifies that no concurrent update changed the frozen payload.
for item in payload:
    assert sha(ROOT/item['path']) == item['sha256'], item['path']
assert sha(script) == manifest['sources'][0]['sha256']
raw_final=git('hash-object','--no-filters','--stdin-paths',input=('\n'.join(listing)+'\n').encode()).decode().splitlines()
filtered_final=git('hash-object','--stdin-paths',input=('\n'.join(listing)+'\n').encode()).decode().splitlines()
assert raw_final==filtered_final and len(raw_final)==len(listing)
assert git('diff','--name-only','-z')==tracked_status_before
assert git('ls-files','--stage','-z')==index_before
assert git('rev-parse','HEAD').decode().strip() == head, 'HEAD changed during preparation'
print(json.dumps(dict(artifact_files=len(listing), source_files=1, total_bytes=manifest['total_payload_bytes'],
    already_tracked_reference_files=len(referenced), baseline_head=head,
    manifest_sha256=sha(OUT/'manifest.json'), paths_sha256=sha(OUT/'artifact_paths.txt'),
    sources_sha256=sha(OUT/'source_paths.txt')), indent=2))
