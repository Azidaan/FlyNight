"""Record an actual compiled controller launch after its admission check."""
from datetime import datetime
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/z13_triton_hota_v1_20261014/full'


def main():
    status = json.loads((RUN / 'status.json').read_text(encoding='utf-8'))
    assert status['state'] == 'running'
    assert json.loads((RUN / 'technical/report.json').read_text(encoding='utf-8'))['passed']
    assert any(json.loads(p.read_text(encoding='utf-8'))['state'] == 'armed'
               for p in RUN.glob('deadline_guard_*/started.json'))
    assert (RUN / 'inference_identity.json').is_file()
    auth = json.loads((RUN / 'full_authorization.json').read_text(encoding='utf-8'))
    proof = {'verified': datetime.now().astimezone().isoformat(), 'status_snapshot': status,
             'deadline': auth['deadline'], 'guard_armed': True, 'technical_probe_passed': True,
             'source_identity': str(RUN / 'inference_identity.json'), 'hota_measured': False}
    (RUN / 'launch_verification.json').write_text(json.dumps(proof, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    path = ROOT / 'experiments/project_history.json'
    cat = json.loads(path.read_text(encoding='utf-8'))
    card = next(c for c in cat['experiments'] if c['id'] == 'fly_spatial_hota_policy_z13_v1')
    card['evidence'].append({'kind': 'json', 'ref': 'runs/z13_triton_hota_v1_20261014/full/launch_verification.json',
                            'claim': 'По просьбе автора тот же протокол продолжен отдельным скомпилированным исполнением после проверки совпадения; старые выходы сохранены'})
    path.write_text(json.dumps(cat, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    plan = ROOT / 'dissertation/ЕДИНЫЙ_ПЛАН.md'
    text = plan.read_text(encoding='utf-8')
    now = datetime.now().astimezone()
    entry = f'''**{now:%d.%m.%Y, %H:%M} MSK** — отдельный HOTA-прогон на Triton запущен

`runs/z13_triton_hota_v1_20261014/full/launch_verification.json`: после технического допуска запущен последовательный контроллер B/C/D с прежним протоколом, моделями, порогами и MASA. Срок {auth['deadline']} не продлён; независимый ограничитель включён. Из технической пробы переиспользованы только два выхода скомпилированной версии, остальные её видео считаются отдельно от обычного исполнения. Шесть готовых видео обычного B сохранены; смешения кэшей нет. `inference_identity.json` фиксирует исходники, модели, данные, описание окружения и RECORD установленного Triton. Новая HOTA ещё не измерена; ускорение не объявляется улучшением качества.

'''
    plan.write_text(text.replace('## Журнал изменений\n\n', '## Журнал изменений\n\n' + entry, 1), encoding='utf-8')
    print(json.dumps(proof, ensure_ascii=False))


if __name__ == '__main__':
    main()
