"""Prespecify the two conditional follow-ups authorized by the author."""
from datetime import datetime
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
now = datetime.now().astimezone()
base_run = ROOT / 'runs/z13_cache_balanced_v1_20261015'
result = json.loads((base_run / 'report.json').read_text(encoding='utf-8'))
assert result['passed'] and all(v['cache_benefit_supported_within_this_sequence'] for v in result['summary'].values())
base = json.loads((base_run / 'protocol.json').read_text(encoding='utf-8'))
source = (ROOT / 'scripts/z13_cache_balanced_probe.py').read_text(encoding='utf-8')
c_script = ROOT / 'scripts/z13_cache_followup_c_probe.py'
assert not c_script.exists()
c_script.write_text(source.replace('z13_cache_balanced_v1_20261015', 'z13_cache_followup_c_v1_20261015')
                   .replace("D/model.pt", "C/model.pt"), encoding='utf-8')
model_protocol = json.loads((ROOT / 'runs/z13_spatial_hota_v1_20261014/protocol.json').read_text(encoding='utf-8'))
specs = [
    ('z13_cache_followup_c_v1', 'runs/z13_cache_followup_c_v1_20261015',
     {**base, 'id': 'z13_cache_followup_c_v1', 'created': now.isoformat(),
      'train_videos': model_protocol['threshold_videos'][1:2],
      'probe_sha256': hashlib.sha256(c_script.read_bytes()).hexdigest(), 'model': 'C',
      'question': 'Сохраняется ли экономия при замороженной FlyVis C и другом train-видео?'}),
    ('z13_cache_clip_v1', 'runs/z13_cache_clip_v1_20261015',
     {'id': 'z13_cache_clip_v1', 'created': now.isoformat(), 'video': model_protocol['videos'][0],
      'warmup_train_videos': model_protocol['threshold_videos'][:1], 'frames': 240,
      'order': ['triton', 'cache', 'cache', 'triton'], 'cost_cap_seconds': 300,
      'gates': {'output_atol_px': 1e-3, 'same_ids_and_events': True, 'benefit': 'Saving positive in both reversed pairs.'},
      'question': 'Сохраняется ли ускорение на одинаковом фрагменте настоящего конвейера D?',
      'limits': '240 кадров одного заранее выбранного видео, две пары. Не полный набор, не HOTA, не гарантия общего времени.',
      'probe_sha256': hashlib.sha256((ROOT / 'scripts/z13_cache_clip_probe.py').read_bytes()).hexdigest()})]
path = ROOT / 'experiments/project_history.json'
catalog = json.loads(path.read_text(encoding='utf-8'))
area = next(c['area'] for c in catalog['experiments'] if c['id'] == 'z13_cache_balanced_v1')
for eid, relative, protocol in specs:
    directory = ROOT / relative; directory.mkdir(exist_ok=True)
    assert not (directory / 'protocol.json').exists()
    (directory / 'protocol.json').write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    assert not any(c['id'] == eid for c in catalog['experiments'])
    catalog['experiments'].append({'id': eid, 'title': protocol['question'], 'area': area,
        'state': 'planned', 'started': now.strftime('%Y-%m-%d'), 'question': protocol['question'],
        'hypothesis': 'Польза кэша может воспроизводиться в дополнительном заранее выбранном режиме.',
        'expected': 'Совпадение вычислений и повторный замер с изменением порядка, до 5 минут GPU на проверку.',
        'acceptance': [{'criterion': 'Выход совпадает в указанном допуске, решения одинаковы.', 'met': None},
                       {'criterion': protocol['gates']['benefit'], 'met': None}],
        'cannot_claim': protocol['limits'],
        'evidence': [{'kind': 'json', 'ref': relative + '/protocol.json', 'claim': 'Критерии, источник и порядок до дополнительных расчётов'}]})
path.write_text(json.dumps(catalog, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
plan = ROOT / 'dissertation/ЕДИНЫЙ_ПЛАН.md'; text = plan.read_text(encoding='utf-8')
entry = f'''**{now:%d.%m.%Y, %H:%M} MSK** — автор разрешил две дополнительные проверки кэша после успеха

*Постановка.* Первая проверка завершена: `runs/z13_cache_balanced_v1_20261015/report.json`. Автор: «Если тест окажется положительным, то сделай ещё пару тестов, если и они окажутся положительными, то попробуем написать с ускорением весь прогон». До следующих измерений зафиксированы `z13_cache_followup_c_v1` (C, другое train-видео, те же прогретые чередующиеся замеры) и `z13_cache_clip_v1` (D, первые 240 кадров заранее выбранного dev-видео, два обратных порядка). Источники и критерии — `runs/z13_cache_followup_c_v1_20261015/protocol.json` и `runs/z13_cache_clip_v1_20261015/protocol.json`. Каждая проба до 5 минут, один GPU-процесс, без обучения и HOTA. Предел всего окна 07:40 не продлевается. По успеху обоих — подготовка отдельного ускоренного исполнения с сохранением идентичности весов, входов, порогов и происхождения уже готового B; научный протокол не меняется.

'''
plan.write_text(text.replace('## Журнал изменений\n\n', '## Журнал изменений\n\n' + entry, 1), encoding='utf-8')
print('Prepared two conditional probes')
