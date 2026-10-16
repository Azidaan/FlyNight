"""Register one bounded matched speed test, not a new tracking hypothesis."""
from datetime import datetime, timedelta
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tracking.z13_checkpoint import atomic_json
RUN = ROOT / 'runs/z13_cache_speed_v1_20261014'
FULL = ROOT / 'runs/z13_triton_hota_v1_20261014/full'


def main():
    assert json.loads((FULL / 'status.json').read_text(encoding='utf-8'))['state'] == 'paused'
    now = datetime.now().astimezone()
    window = json.loads((FULL / 'full_authorization.json').read_text(encoding='utf-8'))
    deadline = min(now + timedelta(minutes=20), datetime.fromisoformat(window['deadline']))
    RUN.mkdir(exist_ok=True)
    protocol = {'id': 'z13_cached_params_speed_v1', 'created': now.isoformat(),
         'video': '00067cfb-e535423e', 'arms': ['B', 'D'],
         'comparison': 'Current Triton versus same Triton plus immutable FlyVis parameter cache',
         'B_control': 'No changed mathematical or execution operation in B; report timing fluctuation honestly',
         'scope': 'One identical 1206-frame video, fresh outputs for both modes and arms; one GPU process; no training, HOTA, heldout or new data',
         'gates': {'state_atol': 1e-4, 'probability_atol': 1e-5,
                   'output_atol_px': 1e-3, 'track_ids_and_event_counts_equal': True},
         'timing': 'Use signatures.json seconds to match the previous table; also report outer wall time including model construction',
         'cpu_npz_check': 'Same full ordered MASA frame/box task list; repeated unpacking versus arrays loaded once; no GPU MASA run',
         'decision': 'Resume the existing frozen Triton full run after the probe regardless of sign; new cache not inserted into its fixed identity',
         'maximum_seconds': 1200, 'deadline': deadline.isoformat()}
    atomic_json(RUN / 'protocol.json', protocol)
    atomic_json(RUN / 'full_authorization.json', {'approved': True,
         'human_request': 'Я хочу увидеть результат такой таблицы с новой оптимизацией',
         'maximum_seconds': 1200, 'deadline': deadline.isoformat(), 'scope': protocol['scope']})
    atomic_json(RUN / 'status.json', {'state': 'preparing', 'updated': now.isoformat()})
    catalog_path = ROOT / 'experiments/project_history.json'
    catalog = json.loads(catalog_path.read_text(encoding='utf-8'))
    previous = next(c for c in catalog['experiments'] if c['id'] == 'z13_triton_execution_v1')
    assert not any(c['id'] == protocol['id'] for c in catalog['experiments'])
    catalog['experiments'].append({'id': protocol['id'], 'title': 'З13 — время с однократной подготовкой параметров',
        'area': previous['area'], 'state': 'running', 'started': now.strftime('%Y-%m-%d'),
        'question': 'Даёт ли однократная подготовка неизменных параметров дополнительное ускорение поверх Triton?',
        'hypothesis': 'Исключение повторной сборки параметров может сократить время D; у B правки нет.',
        'expected': 'Парная таблица B/D на одном полном видео, совпадение выходов и отдельная CPU-проверка чтения npz.',
        'acceptance': [{'criterion': 'Последовательное состояние/вероятности и полный выход сохраняются в записанном допуске, номера и счётчики событий совпадают.', 'met': None},
                       {'criterion': 'Время обработки D с кэшем меньше времени того же Triton без кэша.', 'met': None}],
        'cannot_claim': 'Одно видео не даёт надёжной оценки скорости на всём наборе. HOTA и независимая проверка не считаются. B — контроль разброса, в нём эта правка отсутствует.',
        'evidence': [{'kind': 'json', 'ref': 'runs/z13_cache_speed_v1_20261014/protocol.json', 'claim': 'До замера: один GPU-процесс, неизменные модели/входы, допуски и отдельные этапы времени'}]})
    catalog_path.write_text(json.dumps(catalog, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    plan = ROOT / 'dissertation/ЕДИНЫЙ_ПЛАН.md'
    text = plan.read_text(encoding='utf-8')
    entry = f'''**{now:%d.%m.%Y, %H:%M} MSK** — автор запросил таблицу времени новой оптимизации

*Постановка (`z13_cached_params_speed_v1`).* Текущий HOTA через Triton временно остановлен с сохранением готовых видео. Проверяется отдельный кэш неизменных параметров FlyVis поверх прежнего Triton, без изменения математики, весов и порога. B не меняется и служит контролем колебаний времени. `runs/z13_cache_speed_v1_20261014/protocol.json`: одно одинаковое полное видео, свежие выходы B/D до и после, допуски состояния/вероятности/рамок и совпадение номеров/счётчиков событий. До 20 минут, один GPU-процесс. Чтение архива MASA проверяется отдельно на CPU, не складывается со временем прежней таблицы. Затем возобновляется прежняя зафиксированная версия HOTA; кэш в неё не вставляется, общий предел до 15.10 08:46:23 MSK не продлевается.

'''
    plan.write_text(text.replace('## Журнал изменений\n\n', '## Журнал изменений\n\n' + entry, 1), encoding='utf-8')
    print(json.dumps(protocol, ensure_ascii=False))


if __name__ == '__main__':
    main()
