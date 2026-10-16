"""Prepare isolated full execution and traceable reuse of unchanged B outputs."""
from datetime import datetime
from pathlib import Path
import hashlib
import json
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.z13_cached_hota_runner import scope
from tracking.z13_checkpoint import atomic_json

SOURCE = ROOT / 'runs/z13_triton_hota_v1_20261014/full'
RUN = ROOT / 'runs/z13_cached_hota_v1_20261015/full'


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def digest(p):
    h = hashlib.sha256()
    with p.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def copy_checked(source, target):
    source.resolve().relative_to(ROOT)
    target.resolve().relative_to(ROOT)
    assert not target.exists()
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    assert digest(source) == digest(target)


def main():
    now = datetime.now().astimezone()
    checks = [read(ROOT / p / 'report.json') for p in (
        'runs/z13_cache_balanced_v1_20261015', 'runs/z13_cache_followup_c_v1_20261015',
        'runs/z13_cache_clip_v1_20261015')]
    assert all(r['passed'] for r in checks)
    assert all(s['cache_benefit_supported_within_this_sequence'] for r in checks[:2] for s in r['summary'].values())
    assert checks[2]['benefit_in_both_pairs']
    assert not RUN.exists()
    RUN.mkdir(parents=True)
    for name in ('protocol.json', 'videos.json', 'environment.json'):
        copy_checked(SOURCE / name, RUN / name)
    for arm in ('B', 'C', 'D'):
        copy_checked(SOURCE / arm / 'model.pt', RUN / arm / 'model.pt')
    copy_checked(SOURCE / 'technical/report.json', RUN / 'technical/report.json')
    authorization = read(SOURCE / 'full_authorization.json')
    authorization.update(approved=False, backend='Triton plus guarded immutable parameter preparation',
        reason='Code preparation authorized after three positive probes. Current original HOTA continues; no new full run started.')
    atomic_json(RUN / 'full_authorization.json', authorization)
    protocol = read(RUN / 'protocol.json')
    files = scope()['FILES'] + [str((RUN / 'protocol.json').relative_to(ROOT)).replace('\\', '/')]
    files += [str((RUN / arm / 'model.pt').relative_to(ROOT)).replace('\\', '/') for arm in protocol['arms']]
    files += ['weights/masa/masa_r50.pth', 'runs/flyvis/results/flow/0000/000/best_chkpt',
              'runs/train_cache/selection.json', 'tracking/retina.py', 'tracking/foveated_eyes.py']
    for video in protocol['videos']:
        files += [f'runs/train_cache/{video}/detections/raw_video.npz',
                  f'runs/train_cache/{video}/motion/flyvis_tiled.json',
                  f'data/bdd100k/videos/train/{video}.mov']
    identity = {p: digest(ROOT / p) for p in files}
    old_identity = read(SOURCE / 'inference_identity.json')
    old_prefix = str(SOURCE.relative_to(ROOT)).replace('\\', '/')
    new_prefix = str(RUN.relative_to(ROOT)).replace('\\', '/')
    common = {}
    for old_name, value in old_identity.items():
        new_name = old_name.replace(old_prefix + '/', new_prefix + '/')
        if new_name in identity:
            assert identity[new_name] == value, 'Changed shared input/code: ' + new_name
            common[new_name] = value
    assert len(common) > 100
    atomic_json(RUN / 'inference_identity.json', identity)
    reused = []
    for marker in sorted((SOURCE / 'completed_jobs').glob('B_sig_*.json')):
        original = read(marker)
        assert original['identity'] == old_identity and original['job']['arm'] == 'B'
        video = original['job']['video']
        hashes = {}
        for name in ('signatures.npz', 'signatures.json'):
            source = SOURCE / 'B/signatures' / video / name
            target = RUN / 'B/signatures' / video / name
            copy_checked(source, target)
            hashes[name] = digest(target)
        atomic_json(RUN / 'completed_jobs' / marker.name, {
            'identity': identity, 'job': original['job'], 'completed': original['completed'],
            'reused_from': str(marker.relative_to(ROOT)),
            'reuse_proof': str((RUN / 'reuse_proof.json').relative_to(ROOT))})
        reused.append({'video': video, 'source_marker': str(marker.relative_to(ROOT)), 'copied_sha256': hashes})
    proof = {'created': now.isoformat(), 'reused': reused, 'shared_inputs_and_code_equal': len(common),
        'B_equivalence': 'Same SpatialLocalizer and compiled HexGRU. Cache enable returns before tensor operations for B; whole B video equality already verified in z13_cached_params_speed_v1.',
        'source_identity': str((SOURCE / 'inference_identity.json').relative_to(ROOT)),
        'new_execution_files': [p for p in identity if p not in common],
        'source_protocol_byte_identical': digest(SOURCE / 'protocol.json') == digest(RUN / 'protocol.json'),
        'GPU_run_started': False, 'deadline': authorization['deadline']}
    atomic_json(RUN / 'reuse_proof.json', proof)
    atomic_json(RUN / 'status.json', {'state': 'prepared', 'completed_jobs': len(reused),
                                    'total_jobs': 109, 'reason': 'Separate cached execution ready, not launched'})
    path = ROOT / 'experiments/project_history.json'
    catalog = read(path)
    eid = 'z13_cached_execution_v1'
    assert not any(c['id'] == eid for c in catalog['experiments'])
    area = next(c['area'] for c in catalog['experiments'] if c['id'] == 'z13_triton_execution_v1')
    catalog['experiments'].append({'id': eid, 'title': 'З13 — подготовленное полное исполнение Triton с кэшем',
        'area': area, 'state': 'ready', 'started': now.strftime('%Y-%m-%d'),
        'question': 'Можно ли выполнить фиксированный HOTA с проверенным кэшем без изменения моделей и потери готового B?',
        'hypothesis': 'Кэш сократит часть затрат C/D; B математически не меняется и его готовые выходы можно переиспользовать с явной проверкой происхождения.',
        'expected': 'Отдельный возобновляемый каталог полного исполнения, неизменные модели/пороги/dev35/MASA, 109 заданий, собственный дедлайн.',
        'acceptance': [{'criterion': 'Общий код/входы/протокол совпадают; скопированные выходы B байтово идентичны, происхождение сохранено.', 'met': True, 'evidence': 'runs/z13_cached_hota_v1_20261015/full/reuse_proof.json'},
                       {'criterion': 'Полный HOTA завершён в новом исполнении.', 'met': None}],
        'cannot_claim': 'GPU-прогон с новым исполнением не запущен; полное ускорение и HOTA ещё не измерены. Короткие проверки не гарантируют прежние решения на всём наборе.',
        'decision': 'Подготовлено по условному разрешению автора. Действующий исходный Triton HOTA продолжает работать; новый каталог не запускается параллельно.',
        'evidence': [{'kind': 'json', 'ref': 'runs/z13_cached_hota_v1_20261015/full/reuse_proof.json', 'claim': 'Происхождение переиспользованного B, равенство общих входов и новый контракт кода'}]})
    path.write_text(json.dumps(catalog, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    plan = ROOT / 'dissertation/ЕДИНЫЙ_ПЛАН.md'; text = plan.read_text(encoding='utf-8')
    entry = f'''**{now:%d.%m.%Y, %H:%M} MSK** — полный код с кэшем подготовлен, исходный HOTA возобновлён

*Получено (`z13_cached_execution_v1`).* Все три короткие проверки прошли критерии. Подготовлены отдельные `tracking/z13_compiled_cached_backend.py`, `scripts/z13_cached_hota_worker.py`, `scripts/z13_cached_hota_runner.py` и каталог `runs/z13_cached_hota_v1_20261015/full`. Модели, пороги, видео, научный протокол и MASA не менялись. Переиспользовано {len(reused)} завершённых B-выходов: байтовые хэши и исходные маркеры — `reuse_proof.json`, общие входы/код сверены. Кэш для B не выполняет тензорных операций. Новый код и происхождение результатов зафиксированы отдельно от действующего исполнения.

*Решение.* Исходный Triton HOTA возобновлён 15.10 в 01:01 с сохранёнными 19 заданиями (`runs/z13_cache_clip_v1_20261015/resume_verified.json`). Новый полный код готов, но GPU в нём не запущен; параллельного расчёта нет. Абсолютный предел 07:40 сохраняется. В коротком конвейере выигрыш лишь около 1–2%, на весь HOTA он не переносится.

*Поправка записи.* В двух прошлых записях нейронных замеров форматировщик заменил точки и в путях. Числа в первичных файлах и карточках не менялись. Точные источники: `runs/z13_cache_balanced_v1_20261015/report.json` и `runs/z13_cache_followup_c_v1_20261015/report.json`. Форматировщик исправлен, прежние записи сохранены.

'''
    plan.write_text(text.replace('## Журнал изменений\n\n', '## Журнал изменений\n\n' + entry, 1), encoding='utf-8')
    print(json.dumps({'prepared': True, 'reused_B': len(reused), 'common_hashes': len(common), 'gpu_started': False}))


if __name__ == '__main__':
    main()
