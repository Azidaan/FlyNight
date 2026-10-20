"""Record and publish only the prepared context-transfer experiment."""
import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.z13_parallel_next import read, write, sha
from scripts import z13_record_next as rec

RUN = ROOT / 'runs/z13_context_transfer_20261020'
OWN = ['scripts/z13_context_transfer_20261020.py',
       'scripts/z13_context_transfer_progress.ps1',
       'scripts/z13_record_context_transfer_20261020.py',
       'scripts/z13_context_transfer_service_20261020.py']


def main(start, publish):
    catalog_path = ROOT / 'experiments/project_history.json'
    cat = read(catalog_path)
    card = next(c for c in cat['experiments'] if c['id'] == 'z13_context_transfer_v1')
    marker = RUN / ('start_recorded.json' if start else 'finish_recorded.json')
    if not marker.exists():
        technical = read(RUN / 'technical.json')
        end_technical = read(RUN / 'technical_controller_result.json')
        assert technical['passed'] and end_technical['worker_ended'] and end_technical['exit_code'] == 0
        if start:
            auth = read(RUN / 'full_authorization.json')
            launch = read(RUN / 'controller.json')
            fields = dict(state='running')
            text = ('*Проба.* runs/z13_context_transfer_20261020/technical.json: '+json.dumps(technical,ensure_ascii=False)+
                '\n\n*Продолжение.* Автор20.10попросил продолжить работу; выполняется обычный CPU-пилот по существующему плану, без GPU/обучения ядра. '
                'Самостоятельно установлен конечный лимит '+str(auth['budget_seconds'])+'с плюс30с завершения; источник полномочий и расчёт цены — full_authorization.json. '
                'Фактический запуск и сроки — controller.json: '+json.dumps(launch,ensure_ascii=False)+
                '. Кандидат C выбран по вспомогательным данным; исходный отрицательный критерий D не пересматривается. '
                'Train_rationale.json: парные интервалы C на train26 включают ноль, поэтому перенос остаётся проверкой неопределённой гипотезы. '
                'Проверка не является независимой итоговой оценкой, не запускает HOTA,75или чат-автоматизации.')
        else:
            end = read(RUN / 'controller_result.json')
            assert end['worker_ended']
            good = end['exit_code'] == 0 and not end['hard_timeout'] and (RUN / 'report.json').exists()
            report = read(RUN / 'report.json') if good else None
            source = 'runs/z13_context_transfer_20261020/'+('report.json' if good else 'controller_result.json')
            fields = dict(state='completed_positive' if good and report['proceed'] else 'inconclusive',
                done=datetime.now(timezone(timedelta(hours=3))).date().isoformat(),
                obtained=source+': '+json.dumps({k:v for k,v in report.items() if k!='coverage'} if good else end,ensure_ascii=False),
                interpretation=('Оба заранее заданных критерия переноса выполнены; правило применения в трекере ещё не проверено.' if good and report['proceed'] else
                    'Критерии этой реализации не выполнены; это не закрывает FlyVis.' if good else 'Сохранён частичный расчёт без вывода об эффективности.'),
                decision=report['decision'] if good else 'save_partial_without_efficiency_verdict',
                cannot_claim='Нет HOTA,75отложенных, слепогоdev или гарантии безопасности неизвестных исходов. Это новая версия считывания C, не доказательство пользы дообучения D.')
            met = [True, report['localization_pass'], report['selection_pass']] if good else [True,None,None]
            for item, value in zip(card['acceptance'], met): item.update(met=value,evidence=source)
            card['evidence'].append(dict(kind='json',ref=source,claim='Полный либо частичный первичный итог'))
            text = '*Получено.* '+fields['obtained']+'\n\n*Вывод.* '+fields['interpretation']+'\n\n*Решение.* '+fields['decision']+'. HOTA не измерялась.'
        rec.update(card,fields,'Зафиксирован запуск либо завершение отдельного переноса пространственного считывания C')
        card['files'] = OWN
        card['evidence'].append(dict(kind='json',ref='runs/z13_context_transfer_20261020/technical.json',claim='Точное воспроизведение направления и прежнего контроля; это техническая готовность'))
        catalog_path.write_text(json.dumps(cat,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
        rec.journal(('запуск ' if start else 'итог/сохранение ')+'z13_context_transfer_v1',text)
        write(marker,dict(recorded=True,phase='start' if start else 'finish'))
    write(RUN / 'administrative_sources.json',dict(record_script_sha256=sha(Path(__file__).resolve())))
    rec.BASE = RUN
    rec.OWN_FILES = OWN
    blocked = publish and (ROOT / '.git/index.lock').exists()
    if blocked:
        write(RUN / 'publication_pending.json',dict(published=False,publication_attempted=False,
            reason='Existing index.lock preserved under author prohibition; recording and validation continue locally',
            previous_failure_log='runs/z13_context_transfer_20261020/start_publication.log'))
    rec.validate_and_publish(RUN,publish and not blocked,False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser();parser.add_argument('--start',action='store_true');parser.add_argument('--publish',action='store_true')
    args = parser.parse_args();main(args.start,args.publish)
