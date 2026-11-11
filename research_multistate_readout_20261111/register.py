"""Freeze provenance and acceptance before missing-state recovery."""
from common import *
start()
import subprocess

assert read(OUT / 'math_checks.json')['passed']
assert not (OUT / 'protocol.json').exists()
p = read(OLD / 'protocol_v2.json')
phase = read(OLD / 'execution_protocol.json')
assert len(p['edges']) == 26 and sum(e['kind'] == 'own' for e in p['edges']) == 20
assert len(phase['inputs']) == 52
paths = {OUT / n for n in ['common.py', 'register.py', 'math_check.py', 'math_checks.json', 'recover.py', 'compare.py', 'audit.py', 'ОСНОВАНИЕ.md']}
paths.update(OLD / n for n in ['protocol_v2.json', 'execution_protocol.json', 'features.json', 'predictions.json', 'summary.json', 'maps_complete.json'])
paths.update(ROOT / n for n in ['tracking/z13_readout.py', 'scripts/z13_target_model.py', 'tracking/foveated_eyes.py',
    'tracking/hex_linear_interpolation.py', 'tracking/observed_spatial_reader.py', 'tracking/full_template_correspondence.py',
    'tracking/channel_identity_head.py', 'tracking/channel_balanced_correspondence.py', 'tracking/flyvis_adapter.py',
    'tools/flyvis-runtime/flyvis/network/network.py', 'runs/z13_local_peak_20261021/geometry.json',
    'runs/flyvis/results/flow/0000/000/best_chkpt', 'runs/flyvis/results/flow/0000/000/_meta.yaml',
    'runs/z13_target_center_dev_20261018/C/normalization.json', 'runs/z13_target_center_dev_20261018/R1/normalization.json'])
for k, item in phase['inputs'].items():
    path = ROOT / item['path']
    assert sha(path) == item['sha256']
    paths.add(path)
    arrays(k)  # Confirm all final C34 and ordinary snapshots actually exist.
    for ready in phase['inventory_requests'].get(k, {}).get('ready_files', []): paths.add(ROOT / ready['path'])
    for suffix in ['.npz', '.json']:
        path = OLD / 'maps' / (k + suffix)
        if path.exists(): paths.add(path)
sources = {f.relative_to(ROOT).as_posix(): sha(f) for f in paths}
for name in ['tracking/z13_readout.py', 'scripts/z13_target_model.py', 'tracking/channel_identity_head.py',
             'runs/flyvis/results/flow/0000/000/best_chkpt']:
    assert sources[name] == phase['sources'][name], name
before = {}
for name in ['dissertation/ЕДИНЫЙ_ПЛАН.md', 'experiments/project_history.json', 'experiments/obsidian_experiment_topics.json', 'experiments/metrics_glossary.json']:
    before[name] = sha(ROOT / name)
    save = OUT / 'before' / name; save.parent.mkdir(parents=True, exist_ok=True); save.write_bytes((ROOT / name).read_bytes())
protected = {f.relative_to(ROOT).as_posix(): sha(f) for f in (ROOT / 'ObsidianFlyVis/Карта проекта/Гипотезы').glob('*.md')}
write('before.json', dict(files=before, protected_hypothesis_graph=protected,
    dirty=subprocess.check_output(['git', 'status', '--porcelain', '-uno', '-z'], cwd=ROOT).decode().split('\0')))
write('protocol.json', dict(registered=datetime.now(MSK).isoformat(), head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT).decode().strip(),
    authorization='Direct author request 11.11.2026: math, missing intermediate maps, joint vs final and no-FlyVis controls; no threshold tuning or completed-work repeat.',
    deadline=STOP.isoformat(), sources=sources, edges=26, own=20, foreign=6, videos=20, snapshots=52,
    changed_component='Equal first/final evidence readout, frozen weights and fixed 34 outputs; unchanged neural core and terminal recurrence.',
    controls=['duplicated_final_identity', 'R134_temporal_linear_mid', 'R134_spatial_linear_mid'],
    acceptance=['Math pass before recovery', 'All52 final-state parity max error<=1e-4, finite aligned midpoint maps, weights/state unchanged by tap',
                'All26 pairs retained, no future or unknown labels, same support/shifts/heads; old final scores reused',
                'Preliminary C34 gate: own>=ready-final, foreign<=ready-final and at least one strict improvement',
                'Unique empirical advantage needs positive lower95% video-paired interval versus both joint ordinary controls; no HOTA or public-ID claim'],
    threshold=0, fusion_weights=[.5, .5], trainable_parameters_added=0,
    ordinary_midpoint='Exact linear interpolation of previous and current archived EMA/transient maps, not two-step reintegration of EMA',
    no_new_training=True, baseline_scores_reused=True, no_automatic_native_pilot=True))
catalog_path = ROOT / 'experiments/project_history.json'
catalog = read(catalog_path); identity = 'flyvis_two_integration_state_readout_20261111'
assert not any(c['id'] == identity for c in catalog['experiments'])
catalog['experiments'].append(dict(id=identity, title='Считывание двух состояний FlyVis — равное объединение с готовой головой', area='flyvis', state='running',
    started='2026-11-11', question='Поможет ли первое состояние того же прохода отличать свои продолжения от чужих при неизменном последнем состоянии?',
    hypothesis='Равное объединение поканальных свидетельств первого и последнего шагов на одном общем сдвиге может дополнить готовое последнее считывание.',
    expected='Не меньше принятых своих и не больше чужих относительно готового последнего C34; хотя бы одно строгое улучшение. Сравнить с обоими обычными совместными контролями.',
    acceptance=[dict(criterion=s, met=None) for s in read(OUT / 'protocol.json')['acceptance']],
    cannot_claim='Разработческие26пар, не независимая оценка и не измерение сохранения публичного ID/HOTA. Два шага интегрирования не являются двумя глубинами графа. Отрицание равного смешивания не исключает обучаемое объединение.',
    evidence=[dict(kind='file', ref=OUT.name+'/protocol.json', claim='Критерии и источники зафиксированы до недостающих карт'), dict(kind='file', ref=OUT.name+'/ОСНОВАНИЕ.md', claim='Математика, контроль, прежние попытки и литературное основание')], tags=['fly-problem-P1']))
catalog_path.write_bytes((json.dumps(catalog, ensure_ascii=False, indent=2)+'\n').encode())
path = ROOT / 'experiments/obsidian_experiment_topics.json';topics = read(path)
topics['topics'][identity] = ['Сохранение личности', 'Выбор продолжения среди рамок', 'Несколько состояний обработки']
path.write_bytes((json.dumps(topics, ensure_ascii=False, indent=2)+'\n').encode())
path = ROOT / 'dissertation/ЕДИНЫЙ_ПЛАН.md';before = path.read_bytes();marker = b'## '+ 'Журнал изменений'.encode();index = before.index(marker)+len(marker)
entry = '\n\n**'+datetime.now(MSK).strftime('%d.%m.%Y, %H:%M MSK')+'** — Два состояния одного прохода FlyVis: математика и фиксированная проверка\n\n'
entry += '*Постановка (`'+identity+'`).* По прямому поручению автора: два внутренних шага существующей динамики, равные веса, один общий сдвиг и готовые головы; ядро, dt, веса, порог0 и26пар(20своих/6чужих) не меняются. Последние карты и оценки переиспользуются, восстанавливаются только недостающие промежуточные представления. Обычные временной и пространственный контроли получают то же считывание линейной промежуточной карты.\n\n'
entry += '*Получено.* До восстановления реальных карт проверены алгебра дублирования, общий максимум и линейный контроль; источник `'+OUT.name+'/math_checks.json`. Протокол: `'+OUT.name+'/protocol.json`; литературное основание и границы — `'+OUT.name+'/ОСНОВАНИЕ.md`.\n\n'
entry += '*Вывод.* Математическая различимость возможна, реальная польза пока не измерена. Это не изменение глубины графа и не повтор опыта выбора более раннего кадра.\n\n*Решение.* Выполнить только разрешённое сравнение; слоты и полный трекер не запускать. Защищённое дерево гипотез не переписывать.\n\n'
path.write_bytes(before[:index]+entry.encode()+before[index:].lstrip(b'\r\n'))
(OUT/'registration_entry.md').write_bytes(entry.encode())
status('registered', completed=0, total=26)
print('REGISTERED',len(sources))
