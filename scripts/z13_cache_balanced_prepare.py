"""Prepare a reviewable short probe; author chooses its timing before GPU use."""
from datetime import datetime
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/z13_cache_balanced_v1_20261015'


def main():
    now = datetime.now().astimezone()
    RUN.mkdir(exist_ok=True)
    assert not (RUN / 'protocol.json').exists()
    base = json.loads((ROOT / 'runs/z13_spatial_hota_v1_20261014/protocol.json').read_text(encoding='utf-8'))
    protocol = {'id': 'z13_cache_balanced_v1', 'created': now.isoformat(),
        'train_videos': base['threshold_videos'][:1], 'repeats': 24,
        'batches': [1, 4], 'modes': ['triton', 'triton_repeat', 'cache', 'cache_without_checks'],
        'gates': {'probability_atol': 1e-5, 'same_argmax': True,
                  'benefit': 'Paired block interval for sequence time saving above zero for both batch sizes.'},
        'measurement': 'One D model, warmed for each mode and batch; same inputs; rotate/reverse order; synchronize CUDA before and after each timing.',
        'mechanism': 'Separately time preparation/dereference of actual parameters and CPU cost of cached return with versus without version/gradient checks.',
        'limits': 'No end-to-end/video/HOTA claim. Timing blocks are repeats of fixed inputs on one GPU, not independent videos. No production code changes.',
        'cost_cap_seconds': 300,
        'probe_sha256': hashlib.sha256((ROOT / 'scripts/z13_cache_balanced_probe.py').read_bytes()).hexdigest(),
        'decision': 'Keep cache disabled in current frozen HOTA. If supported here, retain as candidate for matched end-to-end timing; otherwise disable this implementation without deleting artifacts.'}
    (RUN / 'protocol.json').write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (RUN / 'full_authorization.json').write_text(json.dumps({'approved': False,
        'reason': 'Main HOTA is active and author requested completion by 07:40; await preference before interrupting it for up to 5 minutes.'}, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    path = ROOT / 'experiments/project_history.json'
    catalog = json.loads(path.read_text(encoding='utf-8'))
    previous = next(c for c in catalog['experiments'] if c['id'] == 'z13_cached_params_speed_v1')
    assert not any(c['id'] == protocol['id'] for c in catalog['experiments'])
    catalog['experiments'].append({'id': protocol['id'], 'title': 'З13 — повторный замер вклада кэша на прогретой модели',
        'area': previous['area'], 'state': 'planned', 'started': now.strftime('%Y-%m-%d'),
        'question': 'Помогает ли кэш в вычислительном пути FlyVis и объясняет ли проверка весов часть затрат?',
        'hypothesis': 'Повторная подготовка параметров может быть заметной, но её экономия ограничена; прежний замер видео не отделил эффект от временных колебаний.',
        'expected': 'Чередующиеся повторные замеры одного ядра, время API параметров и проверки версий отдельно; без нового видео или обучения.',
        'acceptance': [{'criterion': 'Вероятности в пределах 1e-5, аргмаксимум совпадает.', 'met': None},
                       {'criterion': 'Парный интервал экономии времени последовательности выше нуля при пакетах 1 и 4.', 'met': None}],
        'cannot_claim': 'Это не время полного видео, не независимые видео и не результат HOTA. Причины нагрузки компьютера по прежним логам не установлены.',
        'decision': 'Подготовлено без запуска GPU; выбор времени короткой пробы оставлен автору. Исходный Triton HOTA продолжает работать до 07:40.',
        'evidence': [{'kind': 'json', 'ref': 'runs/z13_cache_balanced_v1_20261015/protocol.json',
                      'claim': 'До повторного замера: прогрев, порядок, повторы, допуски, ограничение выводов и бюджет'}]})
    path.write_text(json.dumps(catalog, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    plan = ROOT / 'dissertation/ЕДИНЫЙ_ПЛАН.md'
    text = plan.read_text(encoding='utf-8')
    entry = f'''**{now:%d.%m.%Y, %H:%M} MSK** — подготовка проверки механизма кэша без запуска GPU

*Постановка (`z13_cache_balanced_v1`).* Автор хочет установить пользу Triton с кэшем параметров и причину различий. `runs/z13_cache_balanced_v1_20261015/protocol.json`: одна прогретая модель D, один ранее использованный train-источник, пакеты 1 и 4, 24 блока с поворотом и разворотом порядка четырёх режимов. Контроль — повтор исходного Triton без изменений. Кэш сравнивается с проверками неизменности и без них только внутри изолированной пробы. Перед и после замера синхронизация CUDA, отдельно API параметров и стоимость возврата кэша. Допуск вероятностей 1e-5, одинаковые аргмаксимумы; интервал экономии выше нуля в обоих размерах пакета. Повторы фиксированных входов не объявляются независимыми видео. Код `scripts/z13_cache_balanced_probe.py` проверен на синтаксис, GPU не запускался.

*Решение.* Карточка planned. До 5 минут GPU, только после выбора автора о временной паузе основного HOTA; готовые видео сохраняются, после пробы возобновляется тот же Triton без нового кэша. Предел 07:40 не переносится. Сейчас основной прогон продолжает работать. По чтению исходников гипотеза «сам кэш меняет вычисления B» отвергается: он в B не применяется; конкретные причины временной нагрузки компьютера неизвестны.

'''
    plan.write_text(text.replace('## Журнал изменений\n\n', '## Журнал изменений\n\n' + entry, 1), encoding='utf-8')
    print(json.dumps(protocol, ensure_ascii=False))


if __name__ == '__main__':
    main()
