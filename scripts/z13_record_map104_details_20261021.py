"""Attach sourced partial comparisons and execution provenance; no science rerun."""
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_candidate_reserve_20261021 import read,write,sha,now


def main():
    path=ROOT/'experiments/project_history.json';cat=read(path);card=next(c for c in cat['experiments'] if c['id']=='z13_map_only104_v1')
    report_path=ROOT/'runs/z13_map_only104_20261021/report.json';report=read(report_path);assert report['primary_pass'] and read(ROOT/'runs/z13_map_only104_20261021/finish_recorded.json')['primary_pass']
    if card.get('confirmed_results'):raise RuntimeError('Comparisons already attached')
    card['confirmed_results']=[];src=report_path.relative_to(ROOT).as_posix();digest=sha(report_path)
    for arm in ('C','D'):
        key=arm+'_map-'+arm+'_top1';assert report['paired'][key]['ci95'][0]>0
        card['confirmed_results'].append(dict(id=arm+'_map104_vs_top1',title='Фиксированное чтение карты104 против максимума — '+arm,category='readout',metric='useful_center_fraction',unit='fraction',
            evidence=dict(source=src,locator=['paired',key],sha256=digest),baseline=dict(source=src,locator=['summary',arm+'_top1','useful_mean_video'],sha256=digest),
            subset='train26;26 многократно использованных видео',comparison='Те же3 пика карты; выбор фиксированным ансамблем3 сидов, обученным на104train. Дополнительные сырые векторы обнулены, карта FlyVis сохранена.',
            conditions='Ветвь зарегистрирована до обучения104, все основные критерии C выполнены; D — вторичное сравнение.',limitations='Не слепая проверка. R1 с тем же модулем тоже улучшается, весь прирост не приписывается мухе. D не доказала преимущество над ближайшим к прогнозу пиком. Нет HOTA и безопасного отказа.'))
    transfer=next(c for c in cat['experiments'] if c['id']=='z13_map_transfer104_v1');assert read(ROOT/'runs/z13_map_transfer104_20261021/technical.json')['passed'];transfer['state']='running'
    for name in ('technical.json','technical_controller_result.json','technical_launch.json','cache_launch.json'):
        transfer['evidence'].append(dict(kind='file',ref='runs/z13_map_transfer104_20261021/'+name,claim='Фактический старт и техническая проверка без изменения допусков'))
    write(path,cat)
    plan=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';text=plan.read_text(encoding='utf-8');anchor='## Журнал изменений\n'
    text=text.replace(anchor,anchor+f'\n**{now():%d.%m.%Y, %H:%M} MSK** — сохранение карты104 и запуск фиксированного переноса\n\n*Проверено.* `runs/z13_map_only104_20261021/report.json` пересчитан из исходных строк при записи: C карта−максимум +3,874п.п. [1,531;6,091], C−ближайший к прогнозу +2,074п.п. [0,448;3,805]; C−R1 +13,570п.п. [9,210;18,207]. Все основные критерии C выполнены;3 индивидуальные точки положительны, но их интервалы не все выше нуля. D−максимум +4,780п.п. [0,906;9,551] — вторичное сравнение, D−ближайший не подтверждено. R1−максимум +7,219п.п. [4,343;11,462]; разность эффектов C−R1 −3,345п.п. [−8,574;0,817]. Поэтому общий выигрыш считывателя не объявляется специфическим преимуществом мухи. Сравнения добавлены в подборку со всеми ограничениями.\n\n*Исполнение.* CPU-обучение104 завершилось `worker_ended=true, exit_code=0` в03:54:32; записано в `runs/z13_candidate_data104_20261021/result_audit.json`. Первые8 эпизодов каждого C/D/R1 воспроизведены новым переносом: `runs/z13_map_transfer104_20261021/technical.json`; максимум расхождения центра0,000105px и балла0,000028, исходные допуски0,5px/0,001 не менялись. Полный кэш35dev стартовал03:59:56 после завершения технического дерева; окно PowerShell29920 проверено по процессу, дубликат не создавался. Результата dev ещё нет.\n\n*Служебные события.* При запуске короткого анализа карты путь PowerShell в Program Files не найден; контроллер уже завершился, его не перезапускали. Путь взят из работающего процесса, отдельное окно17660 подтверждено ненулевым дескриптором. Пустая Git-блокировка03:40:22 сохранена копией в `runs/z13_local_peak_20261021/index_lock_034022.backup` и убрана после проверки отсутствия процессов Git по ранее данному разрешению. Отправка коммита1e0e6ac отклонена серверным pre-receive обработчиком GitLab (вызов его API); начата одна повторная отправка вместе с f986993, исходы локально сохранены. Это не ошибка расчёта.\n',1)
    plan.write_text(text,encoding='utf-8')
    for name in ('z13_local_peak_20261021','z13_candidate_data104_20261021','z13_map_only104_20261021'):
        write(ROOT/'runs'/name/'night_finished.json',dict(compute_ended=True,result_recorded=True,at=now().isoformat(),publication_pending=True))


if __name__=='__main__':main()
