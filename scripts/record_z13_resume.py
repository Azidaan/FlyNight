"""Record a verified continuation of the author's Z13 pause."""
from datetime import datetime
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    import torch
    run = ROOT / 'runs/z13_spatial_v2_20261013'
    controller = json.loads((run / 'status.json').read_text(encoding='utf-8'))
    progress = json.loads((run / 'D/full/status.json').read_text(encoding='utf-8'))
    assert controller['state'] == 'running' and controller['stage'] == 'D_full'
    assert progress['state'] == 'running' and progress['step'] > 1105
    assert progress['checkpoint_verified']
    checkpoint = torch.load(run / 'D/full/checkpoint.pt', map_location='cpu', weights_only=False)
    assert checkpoint['step'] >= progress['step']
    assert checkpoint['identity'] == json.loads((run / 'D/full/identity.json').read_text(encoding='utf-8'))
    now = datetime.now().astimezone()
    out = run / ('author_resume_' + now.strftime('%Y%m%d_%H%M%S') + '.json')
    report = {'authorization': 'Автор 14.10.2026: Продолжи', 'created': now.isoformat(),
              'controller': controller, 'progress': progress, 'checkpoint_step': checkpoint['step'],
              'checkpoint_identity_matches': True, 'resume_from': 1105,
              'hota_measured': False, 'heldout_opened': False}
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    plan = ROOT / 'dissertation/ЕДИНЫЙ_ПЛАН.md'
    entry = f'''**{now:%d.%m.%Y, %H:%M} MSK** — З13 v2: автор разрешил продолжение после утренней остановки

*Решение автора.* «Продолжи». Сохранены исходные 4000 шагов, выборки и критерии. Предварительная проверка `scripts/z13_resume_preflight.py` сверила подписи кода/данных всех девяти заданий, сохранение D/full на шаге 1105, оптимизатор и генераторы; проверила свободные блокировки контроллера и GPU. Восемь заданий имеют готовые предсказания. Прежний `stop.request` архивирован, не удалён; источник — `runs/z13_spatial_v2_20261013/resume_preflight_20261014_185313.json`.

*Запуск.* В 18:53 скрыто возобновлён контроллер с приоритетом Idle, один рабочий процесс. Готовые B/C переиспользуют обученные сохранения; повторно формируется их завершённая оценка качества без новых шагов обучения. D/full восстановлен с 1105: подтверждено новое сохранение на шаге {checkpoint['step']}, подпись совпадает. Источник — `{out.relative_to(ROOT).as_posix()}`. Дальше — завершение D/full, его предсказания и прежний выбор порога на train. Dev/MASA и отложенная проверка не запускаются автоматически; HOTA не измерялась, процент улучшения неизвестен.

'''
    content = plan.read_text(encoding='utf-8')
    plan.write_text(content.replace('## Журнал изменений\n\n', '## Журнал изменений\n\n' + entry, 1), encoding='utf-8')
    print(json.dumps({'step': checkpoint['step'], 'report': out.relative_to(ROOT).as_posix()}))


if __name__ == '__main__':
    main()
