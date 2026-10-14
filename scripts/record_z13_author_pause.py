"""Verify and record the author's cooperative training pause."""
import json
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]


def main():
    import torch
    import msvcrt
    run = ROOT / 'runs/z13_spatial_v2_20261013'
    status = json.loads((run / 'status.json').read_text(encoding='utf-8'))
    assert status['state'] == 'paused'
    with (run / 'controller.lock').open('r+b') as lock:
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
    path = run / 'D/full/checkpoint.pt'
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    assert checkpoint['identity'] == json.loads((path.parent / 'identity.json').read_text(encoding='utf-8'))
    assert checkpoint['step'] == len(checkpoint['history']) == 1105
    assert checkpoint['optimizer']['state']
    assert all(k in checkpoint for k in ('generator', 'rng_python', 'rng_numpy', 'rng_torch', 'rng_cuda'))
    now = datetime.now().astimezone()
    report = {'created': now.isoformat(), 'request': 'Автор: Мне уходить надо, стопни обучение',
              'controller': status, 'controller_lock_released': True,
              'checkpoint': str(path.relative_to(ROOT)), 'step': checkpoint['step'],
              'identity_matches': True, 'optimizer_and_rng_saved': True, 'hota_measured': False}
    out = run / 'author_pause_20261014.json'
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    plan = ROOT / 'dissertation/ЕДИНЫЙ_ПЛАН.md'
    entry = f'''**{now:%d.%m.%Y, %H:%M} MSK** — З13 v2: обучение приостановлено по просьбе автора

Автор попросил остановить обучение перед уходом. Создан штатный `stop.request`; рабочий процесс сохранил D/full на шаге 1105 из 4000 и завершился, контроллер перешёл в `paused` с кодом 75. Сохранение загружено на процессоре: история соответствует шагу, оптимизатор и генераторы сохранены, подпись данных/кода совпадает; блокировка контроллера освобождена. Источник — `runs/z13_spatial_v2_20261013/author_pause_20261014.json`. Продолжение возможно с этого сохранения после разрешения автора; автоматического возобновления нет. Это остановка по просьбе автора, не отрицательный итог; HOTA не измерялась.

'''
    text = plan.read_text(encoding='utf-8')
    plan.write_text(text.replace('## Журнал изменений\n\n', '## Журнал изменений\n\n' + entry, 1), encoding='utf-8')
    print('Verified paused, saved step', checkpoint['step'])


if __name__ == '__main__':
    main()
