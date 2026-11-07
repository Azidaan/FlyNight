"""Сохранить проверенный экспорт и прежний отказ; группу гипотез не менять."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import json

ROOT=Path(__file__).resolve().parents[1]


def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))


def main():
    folder='runs/b123_20261027/b1_base_export_vectorized/'
    report=read(ROOT/folder/'recovery1/outputs/report.json')
    control=read(ROOT/folder/'recovery1/export_controller.json')
    assert report['passed'] and control['exit_code']==0 and control['all_owned_ended'] and not control['timed_out']
    path=ROOT/'experiments/project_history.json';raw=path.read_bytes();data=json.loads(raw)
    card=next(c for c in data['experiments'] if c['id']=='b1_base_export_vectorized_v1')
    assert card['obtained']==''
    card.setdefault('revisions',[]).append(dict(date='2026-10-27',commit='',field='summary',old=card['summary'],
        reason='Подготовительная запись заменена подтверждённым результатом отдельного служебного восстановления; коммит пока блокирован прежним index.lock.'))
    card.update(state='infrastructure',summary='Наблюдаемые признаки и сохранённые прогнозы базы восстановлены; работа полного трекера ещё не проверена.',
        done='Пакетное вычисление сравнений и кэш NPZ; восстановлен прежний формат нулевого внешнего дополнения по исходным ширинам частей. Сохранены исходный отказ, диагностика и отдельная служебная версия; срок не продлён.',
        obtained='Источник '+folder+'recovery1/outputs/report.json: 28970 запросов, максимальная конечная разность признаков2,384185791015625e-7; прогнозы личности, признаки и прогнозы выбора действия совпали точно. Контроллер '+folder+'recovery1/export_controller.json: exit_code0, timed_out=false, all_owned_ended=true. HOTA не измерялась.',
        interpretation='Пройдено восстановление сохранённой базы для последующего применения. Первый отказ был вызван нулями внешнего дополнения Gtag в прежнем merge(), а не расхождением реальных признаков. Полный трекер и извлечение MASA в нём пока не приняты.',
        decision='Использовать сохранённую PCA и наблюдаемые преобразования при реализации полного трекера. Научная Б1 остаётся непроверенной. Отдельную группу «Гипотезы» не менять.')
    card['acceptance'][0].update(met=True,evidence=folder+'recovery1/outputs/report.json')
    for name,claim in [('controller_result.json','Первый отказ сохранён'),('inspection.json','Расхождение локализовано во внешнем дополнении'),
                       ('recovery1/protocol.json','Отдельная служебная версия с тем же сроком'),
                       ('recovery1/outputs/report.json','Строгое воспроизведение признаков и прогнозов'),
                       ('recovery1/export_controller.json','Конец собственного дерева до исходного срока')]:
        card['evidence'].append(dict(kind='file',ref=folder+name,claim=claim))
    assert path.read_bytes()==raw
    path.write_bytes((json.dumps(data,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
    plan=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';raw=plan.read_bytes();text=raw.decode('utf-8')
    stamp=datetime.now(timezone(timedelta(hours=3))).strftime('%d.%m.%Y, %H:%M MSK')
    note=f'''**{stamp}** — Б1: восстановлена база; Google Scholar включён в литературный поиск

*Получено (`b1_base_export_vectorized_v1`).* `{folder}recovery1/outputs/report.json`: 28970 запросов, максимальная конечная разность признаков2,384185791015625e-7, точное совпадение прогнозов личности, признаков и прогнозов выбора действия. `{folder}recovery1/export_controller.json`: код0, без превышения срока, собственное дерево завершено. Первый отказ сохранён в controller_result.json; inspection.json локализовал несовпадение только во внешних невалидных слотах Gtag. Старый merge() дополнял более узкую часть нулями. Отдельная служебная версия воспроизводит это по исходным ширинам частей, не меняя реальные признаки и исходные строгие проверки. Исходный срок01:32:43 МСК не продлён. Прежняя сводка карточки сохранена в revisions. Это подготовка, а не HOTA: извлечение MASA и нулевая поправка в полном трекере ещё не проверены.

*Литературный доступ.* Автор подключил вкладку Google Scholar через Edge и прямо поручил искать работы также через Scholar. Поиск BUSCA и переход к цитирующим работам проверены; подтверждение `runs/b123_20261027/scholar_access_20261027.png`. Через эту выдачу найден и прочитан первичный текст LTR с масками; дополнение и границы переноса сохранены в `dissertation/Литературная проверка Б1–Б3 — ограничения и восстановление пропусков.md`. Scholar используется для обнаружения работ; научные числа сверяются с первоисточниками. Выдача и цитирования сами по себе доказательствами не считаются. В граф «Гипотезы» ничего не добавлено.

*Сохранность.* Каталог и журнал обновлены; обязательные проверки выполняются после записи. Автоматической пересборки отдельной группы не было по решению автора. Существующий `.git/index.lock` сохранён, коммит и push пока недоступны. Б1 D продолжает обучение, готовые этапы не повторяются.

'''
    assert plan.read_bytes()==raw
    plan.write_bytes(text.replace('## Журнал изменений\n','## Журнал изменений\n\n'+note,1).encode('utf-8'))
    with (ROOT/'runs/b123_20261027/HANDOFF.md').open('a',encoding='utf-8') as f:
        f.write('\n## '+stamp+' — экспорт принят как подготовка\n\n')
        f.write(folder+'recovery1/outputs/report.json passed=true, контроллер код0/all_owned_ended. PCA лежит рядом pca.npz. Реальные признаки и прогнозы базы восстановлены; нули внешнего дополнения Gtag учтены отдельно, старый отказ сохранён. Полное применение MASA и нулевая поправка в трекере ещё не проверены. Отдельный граф не обновлять без команды автора.\n')


if __name__=='__main__':main()
