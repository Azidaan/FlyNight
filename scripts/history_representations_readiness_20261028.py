"""После завершённой Б1: инвентаризация исходных данных для матрицы1/3/4."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from pathlib import Path
import sys, time
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.recovery_research_service_20261027 import RUN, read, write, sha, matching, now


def prerequisite():
    active = read(RUN / 'b1_active_stage.json')
    source = RUN / active['source']; audit_folder = RUN / active['audit']
    if not (source / 'report.json').exists() or not (source / 'independent_audit.json').exists():
        return False, 'Нет полного отчёта Б1 и независимого аудита', source
    r = read(source / 'report.json'); a = read(source / 'independent_audit.json')
    if not r.get('complete') or not a.get('passed') or not a.get('full_series'):
        return False, 'Б1 либо аудит неполны; это не отрицательный результат представлений', source
    for folder in [source, audit_folder]:
        if not (folder / 'controller_result.json').exists():
            return False, 'Контроллер предыдущей ступени ещё не завершён', source
        c = read(folder / 'controller_result.json')
        if c['exit_code'] != 0 or not c['all_owned_ended'] or any(matching(k, v) for k, v in c['owned'].items()):
            return False, 'Предыдущее собственное дерево не подтвердило успешный конец', source
    return True, '', source


def main():
    folder = RUN / 'history_representations_readiness'
    p = read(folder / 'protocol.json')
    for path, digest in p['sha256'].items(): assert sha(ROOT / path) == digest, path
    ok, reason, source = prerequisite()
    if not ok:
        write(folder / 'report.json', dict(preparation_completed=False, prerequisite_passed=False, reason=reason,
            scientific_verdict=None, training_started=False, hota_measured=False, updated=now()))
        return
    old = read(ROOT / 'runs/b1_real_inputs_20261026/protocol.json')
    rows = []
    for split in ['fit84', 'fit20']:
        for video in old[split]:
            paths = dict(video=f'data/bdd100k/videos/train/{video}.mov',
                detections=f'runs/train_cache/{video}/detections/raw_video.npz',
                retinal=f'runs/z13_data/{video}.npz',
                observed_queries=f'runs/b1_real_inputs_20261026/inputs/{video}.json',
                appearance=f'runs/z13_masa_regions_20261026/recovery1/videos/{video}.npz')
            rows.append(dict(video=video, historical_split=split, sources=paths,
                exists={k: (ROOT / v).is_file() for k, v in paths.items()}))
            write(folder / 'status.json', dict(state='running', stage='Наличие исходных наблюдений, без вычисления моделей',
                done=len(rows), total=104, updated=now()), required=False)
    counts = {k: sum(r['exists'][k] for r in rows) for k in rows[0]['exists']}
    report = dict(preparation_completed=True, prerequisite_passed=True, b1_source=source.relative_to(ROOT).as_posix(),
        b1_report_sha256=sha(source / 'report.json'), inventory=rows, available_video_counts=counts,
        ready_for_training=False, new_representations_exported=False, models_trained=0, hota_measured=False,
        next_required=['Фиксировать причинную систему координат/движение камеры для изображения истории',
            'Реализовать временные банки внешности и реальных наблюдений с неизвестными метками отдельно',
            'Сверить отличие от прежних сигнатур/EMA/буфера/Б1/Б3',
            'Реализовать общий интерфейс8сочетаний и точный нулевой контроль',
            'Зафиксировать train/dev/test, критерии, бюджеты, измерить стоимость до обучения/полнойHOTA'], updated=now())
    write(folder / 'report.json', report)
    path = RUN / 'HANDOFF.md'
    path.write_text(path.read_text(encoding='utf-8')+'\n\nИнвентаризацияhistory_representations_readiness/report.jsonготова после полногоБ1и аудита. ЭтоНЕготовоеобучение:ready_for_training=false, моделей0,HOTAнет. НужнареализацияАГЕНТОМтрёхпричинныхвходов/общегоинтерфейса8сочетанийи отдельнаярегистрациястоимости/обучения. Не останавливатьсянаинвентаризацииприготовомразрешённомокне.\n', encoding='utf-8')


if __name__ == '__main__': main()
